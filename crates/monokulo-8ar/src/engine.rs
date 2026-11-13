//! The one GPU a process has, what state it's in, and the arenas that use it.

use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicU8, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use zeroize::Zeroizing;

use crate::{
    backend::{self, Device},
    gpu::{GpuError, GpuErrorKind},
    vectors,
};

/// What's doing the work: the GPU, or the CPU, scanning every block with
/// monero-oxide's `Scanner` because the GPU can't be used. For a setting to show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Engine {
    Gpu { name: String },
    Cpu { reason: Reason },
}

/// Why the GPU isn't being used. Only `NotStarted` and `Starting` can change
/// before the process restarts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// [`warm_up`] hasn't been called, nor anything that calls it.
    NotStarted,
    /// It's setting up the GPU and running the test vectors.
    Starting,
    /// No Vulkan loader or driver. (Metal is part of Apple's systems.)
    NoVulkan,
    /// Vulkan, but no hardware GPU that can do the work.
    NoGpu,
    /// The GPU gave a wrong answer to the test vectors.
    FailedSelfTest,
    /// The GPU failed: while setting up, on the vectors, or later on a batch.
    Failed,
}

const NOT_STARTED: u8 = 0;
const STARTING: u8 = 1;
const READY: u8 = 2;
const NO_VULKAN: u8 = 3;
const NO_GPU: u8 = 4;
const FAILED_SELF_TEST: u8 = 5;
const FAILED: u8 = 6;

/// One of the above. Once it's past `STARTING` it changes only from `READY` to
/// `FAILED`: a GPU that's failed isn't tried again until the process restarts.
static STATE: AtomicU8 = AtomicU8::new(NOT_STARTED);
static CACHE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
/// The device, once it's passed its vectors, until it's retired. Arenas hold it
/// too, so it goes once the last of them lets go.
static DEVICE: Mutex<Option<Arc<Device>>> = Mutex::new(None);
/// The arena `Prefilter`, `check_wallet` and a `Keyring` made with `new` use.
static DEFAULT: OnceLock<Arena> = OnceLock::new();
/// The ready GPU's name, readable without waiting for a batch to finish.
static NAME: OnceLock<String> = OnceLock::new();
/// The warm-up thread, to join once it's done.
static WARM_UP: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
/// Set by the warm-up thread as its last act, for [`settle`] to wait on.
static DONE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
/// Whether the exit handler that waits for a warm-up is registered.
#[cfg(unix)]
static AT_EXIT: std::sync::Once = std::sync::Once::new();
/// How long a process exiting waits for a warm-up still inside the driver.
/// Only the first run of a cold cache compiles the shader (20–30 s); after
/// that a warm-up is over in well under a second.
const EXIT_WAIT: Duration = Duration::from_secs(60);

/// Where compiled shaders are kept: an app's cache folder, say.
pub fn set_cache_dir(dir: PathBuf) {
    *CACHE_DIR.lock().expect("cache lock") = Some(dir);
}

/// Starts the GPU on a thread of its own, if it's not been: compiles the shader,
/// then has it pass the test vectors before anything else is asked of it. It
/// starts once per process, so this is the check on each cold start of an app.
/// An app in the background on iOS can't use the GPU; a warm-up that runs into
/// that ends as if it hadn't started, and the next batch starts it again.
pub fn warm_up() {
    if STATE.compare_exchange(NOT_STARTED, STARTING, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    *DONE.0.lock().unwrap_or_else(|e| e.into_inner()) = false;
    let dir = CACHE_DIR.lock().expect("cache lock").clone();
    let started = std::thread::Builder::new().name("monokulo-8ar-gpu".into()).spawn(move || {
        let state = match Device::start_for_device(dir, false) {
            // The vectors run in an arena of their own, freed after.
            Ok(device) => match backend::Arena::new(&device).and_then(|arena| vectors::passes(&arena)) {
                Ok(true) => {
                    let _ = NAME.set(device.name().to_owned());
                    *DEVICE.lock().unwrap_or_else(|e| e.into_inner()) = Some(device);
                    READY
                }
                Ok(false) => FAILED_SELF_TEST,
                Err(e) => after_failure(e.kind),
            },
            Err(e) => after_failure(e.kind),
        };
        STATE.store(state, Ordering::Relaxed);
        finished();
    });
    match started {
        Ok(handle) => {
            *WARM_UP.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
            // A process that returns from main while this thread is still in
            // the driver can crash in the driver's own teardown: wait for it.
            #[cfg(unix)]
            AT_EXIT.call_once(|| {
                extern "C" fn at_exit() {
                    settle(EXIT_WAIT);
                }
                // SAFETY: atexit takes a plain C function pointer and
                // registers it; the handler touches only this module's
                // statics, which outlive it.
                unsafe {
                    libc::atexit(at_exit);
                }
            });
        }
        Err(_) => {
            STATE.store(FAILED, Ordering::Relaxed);
            finished();
        }
    }
}

/// The state a warm-up that failed in this way leaves the GPU in. Only iOS
/// refusing an app in the background the GPU leaves it to be started again.
fn after_failure(kind: GpuErrorKind) -> u8 {
    match kind {
        GpuErrorKind::NoVulkan => NO_VULKAN,
        GpuErrorKind::NoGpu => NO_GPU,
        GpuErrorKind::Background => NOT_STARTED,
        GpuErrorKind::Failed => FAILED,
    }
}

fn finished() {
    let (lock, signal) = &DONE;
    *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
    signal.notify_all();
}

/// Waits for the warm-up to finish, up to `timeout`, and joins its thread.
/// True once it's finished, or if it was never started; false if it's still
/// going when the time is up. A test binary or a server shutting down calls
/// this first, so the process doesn't exit with the thread inside the
/// driver. A process returning from `main` does the same on its own, for up
/// to a minute.
pub fn settle(timeout: Duration) -> bool {
    if STATE.load(Ordering::Relaxed) == NOT_STARTED {
        return true;
    }
    let (lock, signal) = &DONE;
    let done = lock.lock().unwrap_or_else(|e| e.into_inner());
    let (done, result) = signal.wait_timeout_while(done, timeout, |done| !*done).unwrap_or_else(|e| e.into_inner());
    if result.timed_out() && !*done {
        return false;
    }
    drop(done);
    if let Some(handle) = WARM_UP.lock().unwrap_or_else(|e| e.into_inner()).take() {
        let _ = handle.join();
    }
    true
}

/// What's doing the work now.
pub fn engine() -> Engine {
    let reason = match STATE.load(Ordering::Relaxed) {
        READY => return Engine::Gpu { name: NAME.get().cloned().unwrap_or_default() },
        NOT_STARTED => Reason::NotStarted,
        STARTING => Reason::Starting,
        NO_VULKAN => Reason::NoVulkan,
        NO_GPU => Reason::NoGpu,
        FAILED_SELF_TEST => Reason::FailedSelfTest,
        _ => Reason::Failed,
    };
    Engine::Cpu { reason }
}

/// Whether the GPU has failed or can't be used, so every block is scanned
/// without the prefilter. It's false while the GPU is still starting.
pub fn gpu_failed() -> bool {
    STATE.load(Ordering::Relaxed) > READY
}

/// Whether the GPU has started, passed its vectors and hasn't failed.
pub fn gpu_ready() -> bool {
    STATE.load(Ordering::Relaxed) == READY && device().is_some()
}

fn device() -> Option<Arc<Device>> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The GPU and driver that's ready to scan, if one is.
pub fn gpu_identity() -> Option<String> {
    if STATE.load(Ordering::Relaxed) != READY {
        return None;
    }
    device().map(|d| d.identity())
}

/// Retires the GPU after a failure, until the process restarts. Arenas let go
/// of it as they next look.
fn retire() {
    STATE.store(FAILED, Ordering::Relaxed);
    DEVICE.lock().unwrap_or_else(|e| e.into_inner()).take();
}

/// A share of the GPU for one caller: its own buffers, command buffer and
/// fence, about 43 MB once it's been used. Several arenas work at once, so two
/// keyrings with an arena each (a server's lanes, say) don't wait on each
/// other's batches. A clone is the same arena: whoever holds it has it for a
/// whole `flag_many` at a time. Nothing is made until the GPU is ready and a
/// batch needs it.
#[derive(Clone, Default)]
pub struct Arena(Arc<Mutex<Option<backend::Arena>>>);

impl Arena {
    pub fn new() -> Arena {
        Arena::default()
    }

    /// The arena most callers share: one, as a phone needs.
    pub fn shared() -> Arena {
        DEFAULT.get_or_init(Arena::new).clone()
    }

    /// Runs `work` in this arena, making it if need be. `None` if the GPU isn't
    /// ready, or `work` failed, which retires the GPU, unless it failed only
    /// because the app is in the background.
    pub(crate) fn with<T>(&self, work: impl FnOnce(&backend::Arena) -> Result<T, GpuError>) -> Option<T> {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let device = match STATE.load(Ordering::Relaxed) {
            READY => device(),
            _ => None,
        };
        let Some(device) = device else {
            // A retired device goes once its arenas let go.
            held.take();
            return None;
        };
        if held.as_ref().is_none_or(|arena| !Arc::ptr_eq(arena.device(), &device)) {
            match backend::Arena::new(&device) {
                Ok(arena) => *held = Some(arena),
                Err(_) => {
                    retire();
                    return None;
                }
            }
        }
        match work(held.as_ref().expect("just made")) {
            Ok(done) => Some(done),
            // iOS refused the work, as the app isn't in the foreground: this
            // batch goes to the CPU, and the GPU is there for the next once it
            // is. The arena has already let go of its memory.
            Err(e) if e.kind == GpuErrorKind::Background => None,
            Err(_) => {
                held.take();
                retire();
                None
            }
        }
    }

    /// Gives back the memory the arena kept between batches. One in use is left
    /// alone, as whoever's using it still wants that memory.
    pub fn release(&self) {
        if let Ok(held) = self.0.try_lock()
            && let Some(arena) = held.as_ref()
        {
            arena.release();
        }
    }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Arena")
    }
}

/// Has the ready GPU work with a wallet's own view key, for that wallet to keep
/// the result. `None` if there's no GPU to ask, or it failed, which retires it.
pub fn check_wallet(view: &Zeroizing<[u8; 32]>) -> Option<bool> {
    Arena::shared().with(|arena| vectors::passes_for_wallet(arena, view))
}

/// Whether the GPU can take a batch now. One that hasn't been started is started,
/// for the batches after this one.
pub(crate) fn usable() -> bool {
    match STATE.load(Ordering::Relaxed) {
        READY => device().is_some(),
        NOT_STARTED => {
            warm_up();
            false
        }
        _ => false,
    }
}

/// Scanning has caught up: gives back the memory the shared arena kept between
/// batches. Other arenas are released by whoever holds them.
pub fn rest() {
    Arena::shared().release();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With or without a GPU: before a warm-up there is nothing to wait for,
    /// and after one `settle` returns once the thread is done, leaving the
    /// engine in a state past `Starting`.
    #[test]
    fn settle_waits_for_the_warm_up_and_no_longer() {
        if STATE.load(Ordering::Relaxed) == NOT_STARTED {
            assert!(settle(Duration::from_millis(1)), "nothing started, nothing to wait for");
        }
        warm_up();
        assert!(settle(Duration::from_secs(120)), "the warm-up thread ended");
        assert_ne!(engine(), Engine::Cpu { reason: Reason::Starting });
        assert_ne!(engine(), Engine::Cpu { reason: Reason::NotStarted });
        assert!(settle(Duration::from_millis(1)), "and asking again costs nothing");
        assert!(WARM_UP.lock().unwrap().is_none(), "the thread was joined");
    }

    /// A GPU that failed is given up on, but not one iOS refused only because
    /// the app was in the background: that one is started again.
    #[test]
    fn only_the_background_leaves_the_gpu_to_be_started_again() {
        assert_eq!(after_failure(GpuErrorKind::Background), NOT_STARTED);
        assert_eq!(after_failure(GpuErrorKind::Failed), FAILED);
        assert_eq!(after_failure(GpuErrorKind::NoGpu), NO_GPU);
        assert_eq!(after_failure(GpuErrorKind::NoVulkan), NO_VULKAN);
    }
}
