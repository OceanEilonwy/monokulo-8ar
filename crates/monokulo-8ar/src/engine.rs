//! Which engine does the arithmetic: the setting, and the one GPU a process has.

use std::{
    path::PathBuf,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

use zeroize::Zeroizing;

use crate::{
    fe::Digits,
    gpu::{Gpu, Points, TUNING},
    vectors,
};

/// What scans. A GPU that's still starting, that failed its test vectors, or
/// that's failed or absent, leaves the CPU doing the work in every mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The GPU, for everything.
    Gpu,
    /// Whatever suits the size of the batch: the default. The caller says from how
    /// many keys the GPU takes over ([`Prefilter::flag`](crate::Prefilter::flag)).
    Auto,
    /// Only the CPU's cores.
    Cpu,
}

static MODE: AtomicU8 = AtomicU8::new(1);
static CACHE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
/// Set once a GPU has failed, its test vectors included: it isn't tried again
/// until the process next starts.
static BROKEN: AtomicBool = AtomicBool::new(false);
static STARTING: AtomicBool = AtomicBool::new(false);
static GPU: OnceLock<Mutex<Option<Gpu>>> = OnceLock::new();

pub fn set_mode(mode: Mode) {
    MODE.store(
        match mode {
            Mode::Gpu => 0,
            Mode::Auto => 1,
            Mode::Cpu => 2,
        },
        Ordering::Relaxed,
    );
    if mode != Mode::Cpu {
        warm_up();
    }
}

pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        0 => Mode::Gpu,
        1 => Mode::Auto,
        _ => Mode::Cpu,
    }
}

/// Where compiled shaders are kept: an app's cache folder, say.
pub fn set_cache_dir(dir: PathBuf) {
    *CACHE_DIR.lock().expect("cache lock") = Some(dir);
}

/// Starts the GPU on a thread of its own, if it's not been: compiles the shader,
/// then has it pass the test vectors before anything else is asked of it. It
/// starts once per process, so this is the check on each cold start of an app.
pub fn warm_up() {
    if BROKEN.load(Ordering::Relaxed) || GPU.get().is_some() || STARTING.swap(true, Ordering::Relaxed) {
        return;
    }
    let dir = CACHE_DIR.lock().expect("cache lock").clone();
    let started = std::thread::Builder::new().name("monokulo-8ar-gpu".into()).spawn(move || match Gpu::start_for_device(dir, false) {
        Ok(gpu) if matches!(vectors::passes(&gpu), Ok(true)) => {
            let _ = GPU.set(Mutex::new(Some(gpu)));
        }
        _ => BROKEN.store(true, Ordering::Relaxed),
    });
    if started.is_err() {
        BROKEN.store(true, Ordering::Relaxed);
    }
}

/// Whether the GPU has failed or can't be used, so the CPU is scanning whatever
/// the mode says. It's false while the GPU is still starting.
pub fn gpu_failed() -> bool {
    BROKEN.load(Ordering::Relaxed)
}

/// Whether the GPU has started, passed its vectors and hasn't failed.
pub fn gpu_ready() -> bool {
    !BROKEN.load(Ordering::Relaxed) && GPU.get().is_some()
}

/// The GPU and driver that's ready to scan, if one is.
pub fn gpu_identity() -> Option<String> {
    if !gpu_ready() {
        return None;
    }
    GPU.get()?.lock().ok()?.as_ref().map(Gpu::identity)
}

/// Has the ready GPU work with a wallet's own view key, for that wallet to keep
/// the result. `None` if there's no GPU to ask, or it failed, which retires it.
pub fn check_wallet(view: &Zeroizing<[u8; 32]>) -> Option<bool> {
    let slot = GPU.get()?;
    let mut gpu = slot.lock().ok()?;
    match vectors::passes_for_wallet(gpu.as_ref()?, view) {
        Ok(passed) => Some(passed),
        Err(_) => {
            *gpu = None;
            BROKEN.store(true, Ordering::Relaxed);
            None
        }
    }
}

/// Where a batch of keys goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    Cpu,
    Gpu,
}

/// Where `keys` go. `gpu_allowed` is the wallet's say; the GPU still warms up
/// without it. In [`Mode::Auto`] the GPU takes over from `gpu_from` keys.
pub(crate) fn route(keys: usize, gpu_allowed: bool, gpu_from: usize) -> Route {
    route_for(mode(), keys, usable() && gpu_allowed, gpu_from)
}

/// What each mode does with a batch of keys, given a GPU that's ready.
fn policy(mode: Mode, keys: usize, gpu_from: usize) -> Route {
    match mode {
        Mode::Gpu => Route::Gpu,
        Mode::Auto if keys >= gpu_from => Route::Gpu,
        Mode::Auto | Mode::Cpu => Route::Cpu,
    }
}

fn usable() -> bool {
    if BROKEN.load(Ordering::Relaxed) {
        return false;
    }
    if GPU.get().is_none() {
        if mode() != Mode::Cpu {
            warm_up();
        }
        return false;
    }
    true
}

fn route_for(mode: Mode, keys: usize, gpu_ready: bool, gpu_from: usize) -> Route {
    if gpu_ready { policy(mode, keys, gpu_from) } else { Route::Cpu }
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
            *gpu = None;
            BROKEN.store(true, Ordering::Relaxed);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fob's choice, as an example.
    const GPU_FROM: usize = 512;

    #[test]
    fn each_mode_goes_where_it_says() {
        assert_eq!(route_for(Mode::Cpu, 100_000, true, GPU_FROM), Route::Cpu);
        assert_eq!(route_for(Mode::Gpu, 1, true, GPU_FROM), Route::Gpu, "the GPU, even for a little");
        assert_eq!(route_for(Mode::Gpu, 100_000, true, GPU_FROM), Route::Gpu, "and for a lot");
        assert_eq!(route_for(Mode::Auto, 1, true, GPU_FROM), Route::Cpu, "a little: the CPU costs less");
        assert_eq!(route_for(Mode::Auto, GPU_FROM, true, GPU_FROM), Route::Gpu);
        assert_eq!(route_for(Mode::Auto, 100_000, true, GPU_FROM), Route::Gpu, "a lot of blocks: still only the GPU");
        for mode in [Mode::Auto, Mode::Gpu, Mode::Cpu] {
            assert_eq!(route_for(mode, 100_000, false, GPU_FROM), Route::Cpu, "no GPU, no choice");
        }
    }

    #[test]
    fn auto_goes_by_the_size_of_the_batch() {
        assert_eq!(policy(Mode::Auto, 0, 100), Route::Cpu);
        assert_eq!(policy(Mode::Auto, 99, 100), Route::Cpu);
        assert_eq!(policy(Mode::Auto, 100, 100), Route::Gpu);
        assert_eq!(policy(Mode::Auto, 1_000_000, 100), Route::Gpu);
    }

    #[test]
    fn the_others_go_one_way_at_any_size() {
        for keys in [0, 1, 1_000_000] {
            assert_eq!(policy(Mode::Gpu, keys, 100), Route::Gpu);
            assert_eq!(policy(Mode::Cpu, keys, 100), Route::Cpu);
        }
    }
}
