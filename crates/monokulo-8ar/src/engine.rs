//! The one GPU a process has, and what state it's in.

use std::{
    path::PathBuf,
    sync::{
        Condvar, Mutex, OnceLock,
        atomic::{AtomicU8, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use zeroize::Zeroizing;

use crate::{
    gpu::{Digits, Gpu, GpuErrorKind, Points, TUNING},
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
    /// No Vulkan loader or driver.
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
static GPU: OnceLock<Mutex<Option<Gpu>>> = OnceLock::new();
/// The ready GPU's name, readable without waiting for a batch to finish.
static NAME: OnceLock<String> = OnceLock::new();
/// The warm-up thread, to join once it's done.
static WARM_UP: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
/// Set by the warm-up thread as its last act, for [`settle`] to wait on.
static DONE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
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
pub fn warm_up() {
    if STATE.compare_exchange(NOT_STARTED, STARTING, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let dir = CACHE_DIR.lock().expect("cache lock").clone();
    let started = std::thread::Builder::new().name("monokulo-8ar-gpu".into()).spawn(move || {
        let state = match Gpu::start_for_device(dir, false) {
            Ok(gpu) => match vectors::passes(&gpu) {
                Ok(true) => {
                    let _ = NAME.set(gpu.name().to_owned());
                    let _ = GPU.set(Mutex::new(Some(gpu)));
                    READY
                }
                Ok(false) => FAILED_SELF_TEST,
                Err(_) => FAILED,
            },
            Err(e) => match e.kind {
                GpuErrorKind::NoVulkan => NO_VULKAN,
                GpuErrorKind::NoGpu => NO_GPU,
                GpuErrorKind::Failed => FAILED,
            },
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
            {
                extern "C" fn at_exit() {
                    settle(EXIT_WAIT);
                }
                // SAFETY: atexit takes a plain C function pointer and
                // registers it; the handler touches only this module's
                // statics, which outlive it.
                unsafe {
                    libc::atexit(at_exit);
                }
            }
        }
        Err(_) => {
            STATE.store(FAILED, Ordering::Relaxed);
            finished();
        }
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
    STATE.load(Ordering::Relaxed) == READY && GPU.get().is_some()
}

/// The GPU and driver that's ready to scan, if one is.
pub fn gpu_identity() -> Option<String> {
    if !gpu_ready() {
        return None;
    }
    GPU.get()?.lock().ok()?.as_ref().map(Gpu::identity)
}

/// Retires the GPU after a failure, until the process restarts.
fn retire(gpu: &mut Option<Gpu>) {
    *gpu = None;
    STATE.store(FAILED, Ordering::Relaxed);
}

/// Has the ready GPU work with a wallet's own view key, for that wallet to keep
/// the result. `None` if there's no GPU to ask, or it failed, which retires it.
pub fn check_wallet(view: &Zeroizing<[u8; 32]>) -> Option<bool> {
    let slot = GPU.get()?;
    let mut gpu = slot.lock().ok()?;
    match vectors::passes_for_wallet(gpu.as_ref()?, view) {
        Ok(passed) => Some(passed),
        Err(_) => {
            retire(&mut gpu);
            None
        }
    }
}

/// Whether the GPU can take a batch now. One that hasn't been started is started,
/// for the batches after this one.
pub(crate) fn usable() -> bool {
    match STATE.load(Ordering::Relaxed) {
        READY => GPU.get().is_some(),
        NOT_STARTED => {
            warm_up();
            false
        }
        _ => false,
    }
}

/// The most keys to give [`derive`] at once.
pub(crate) fn batch() -> usize {
    TUNING.batch
}

/// Scanning has caught up: gives back the memory the GPU kept between batches. A
/// GPU in use is left alone, as whoever's using it still wants that memory.
pub fn rest() {
    if let Some(Ok(gpu)) = GPU.get().map(Mutex::try_lock)
        && let Some(gpu) = gpu.as_ref()
    {
        gpu.release();
    }
}

/// 8aR on the GPU, or `None` once it's failed, which retires it.
pub(crate) fn derive(digits: &Digits, keys: &[[u8; 32]]) -> Option<Points> {
    let slot = GPU.get()?;
    let mut gpu = slot.lock().ok()?;
    match gpu.as_ref()?.derive(digits, keys) {
        Ok(points) => Some(points),
        Err(_) => {
            retire(&mut gpu);
            None
        }
    }
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
}
