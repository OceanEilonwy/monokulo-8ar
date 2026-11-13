//! 8aR on the GPU in Metal, for Apple's platforms, which have no Vulkan. The
//! shader is `gpu.rs`'s: `shaders/scan.metal` is what SPIRV-Cross writes from
//! `scan.comp` (`shaders/build.sh`), so the two backends run the same code.
//! Everything around it is the same too: a [`Device`] set up once a process, and
//! [`Arena`]s that each have their own buffers, with the methods `gpu.rs` has,
//! handing back the same [`Points`] for the caller to check.
//!
//! Apple's GPUs share memory with the CPU. The keys, the accounts' digits and the
//! results are in shared buffers, mapped for as long as they're kept; the working
//! state is in the GPU's private memory. As in `gpu.rs`, the state is cleared at
//! the end of each batch, the results once read, and the digits after each load
//! and before the memory goes back.
//!
//! iOS lets only an app in the foreground use the GPU: a command buffer an app
//! in the background commits fails with `MTLCommandBufferErrorNotPermitted`.
//! That's [`GpuErrorKind::Background`], which the engine doesn't retire the GPU
//! for: the batch goes to the CPU, and the next uses the GPU again. Any other
//! failure is an error for the caller, as in `gpu.rs`. Nothing here panics.

use std::{
    cell::RefCell,
    ffi::c_void,
    path::PathBuf,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSProcessInfo, NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferError, MTLCommandBufferErrorDomain, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};
use zeroize::{Zeroize, Zeroizing};

pub use crate::gpu::{DISCRETE_TUNING, Derived, Digits, GpuError, GpuErrorKind, Points, TUNING, Tuning};

/// The shader, as SPIRV-Cross wrote it from `scan.comp`.
const MSL: &str = include_str!("../shaders/scan.metal");
/// What SPIRV-Cross calls the shader's `main`.
const ENTRY: &str = "main0";
/// The shader's arguments, as SPIRV-Cross numbered them in `scan.metal`'s
/// `main0`: the push constants, then the four buffers. A test holds these to the
/// file.
pub(crate) const PUSH_INDEX: usize = 0;
pub(crate) const ACCOUNTS_INDEX: usize = 1;
pub(crate) const STATE_INDEX: usize = 2;
pub(crate) const INPUT_INDEX: usize = 3;
pub(crate) const OUT_INDEX: usize = 4;
/// The specialisation constants, as `scan.comp` numbers them: function
/// constants in Metal.
const GROUP_CONSTANT: usize = 0;
const STAGE_CONSTANT: usize = 1;
const STEPS_CONSTANT: usize = 2;
/// A work item's state on the GPU: 300 words.
const STATE_BYTES: u64 = 300 * 4;
/// A work item's result: the compressed point, then a flag.
const OUT_WORDS: usize = 9;
/// An account's digits on the GPU: 64 signed bytes.
const ACCOUNT_BYTES: usize = 64;
/// The push constants: tx_count, first_item, count, digit_at and op.
const PUSH_BYTES: usize = 20;
/// The most command queues made, for arenas to share round-robin.
const MOST_QUEUES: usize = 4;
/// The most keys one load takes, as in `gpu.rs`.
const MOST_KEYS: usize = 1 << 20;
/// The most accounts one load takes.
const MOST_ACCOUNTS: usize = 1 << 20;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

fn failed(what: &str) -> GpuError {
    GpuError::failed(what.to_owned())
}

/// A Metal error, for the log: never key material.
fn described(what: &str, error: &NSError) -> GpuError {
    GpuError::failed(format!("{what}: {}", error.localizedDescription()))
}

/// The GPU, set up once: shared by every [`Arena`].
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Metal's queues may be used from any thread, so they need no lock.
    queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    next_queue: AtomicUsize,
    init: Pipeline,
    window: Pipeline,
    tail: Option<(usize, Pipeline)>,
    last: Pipeline,
    check: Pipeline,
    tuning: Tuning,
    /// The longest buffer the GPU takes.
    most_bytes: u64,
    name: String,
    /// The system's version: Metal's compiler and driver come with it.
    system: String,
}

// A Metal device, its queues and its pipeline states may be used from any
// thread; what an arena records is its own.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Sets up the GPU, compiling the shader. Metal keeps what it compiles in a
    /// cache of its own, so `cache_dir` isn't used, and there's only ever a
    /// hardware GPU, so neither is `any_kind`: both are there to match `gpu.rs`.
    pub fn start(cache_dir: Option<PathBuf>, tuning: Tuning, any_kind: bool) -> Result<Arc<Device>, GpuError> {
        Self::start_tuned(cache_dir, Some(tuning), any_kind)
    }

    /// [`start`](Device::start), tuned for the GPU it finds: what [`crate::warm_up`] uses.
    pub fn start_for_device(cache_dir: Option<PathBuf>, any_kind: bool) -> Result<Arc<Device>, GpuError> {
        Self::start_tuned(cache_dir, None, any_kind)
    }

    fn start_tuned(_cache_dir: Option<PathBuf>, tuning: Option<Tuning>, _any_kind: bool) -> Result<Arc<Device>, GpuError> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| GpuError::new(GpuErrorKind::NoGpu, "no Metal device".into()))?;
        let name = device.name().to_string();
        // An iPhone's or iPad's GPU is tuned as a phone's is; a Mac's has no
        // watchdog near a batch, so a batch is one submit, as on a discrete GPU.
        let tuning = tuning.unwrap_or(if cfg!(target_os = "macos") { DISCRETE_TUNING } else { TUNING });
        let most_bytes = device.maxBufferLength() as u64;
        if most_bytes < tuning.batch as u64 * STATE_BYTES {
            return Err(GpuError::new(GpuErrorKind::NoGpu, format!("{name} is too small for this work")));
        }
        let queues: Vec<_> = (0..MOST_QUEUES).map_while(|_| device.newCommandQueue()).collect();
        if queues.is_empty() {
            return Err(failed("command queue"));
        }

        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_0);
        let library = device.newLibraryWithSource_options_error(&NSString::from_str(MSL), Some(&options)).map_err(|e| described("shader", &e))?;
        let group = tuning.group;
        let windows = tuning.windows;
        let pipeline = |stage: i32, steps: i32| -> Result<Pipeline, GpuError> {
            let constants = MTLFunctionConstantValues::new();
            // SAFETY: each pointer is to a value of the type given, which Metal
            // copies before the call returns.
            unsafe {
                constants.setConstantValue_type_atIndex(NonNull::from(&group).cast(), MTLDataType::UInt, GROUP_CONSTANT);
                constants.setConstantValue_type_atIndex(NonNull::from(&stage).cast(), MTLDataType::Int, STAGE_CONSTANT);
                constants.setConstantValue_type_atIndex(NonNull::from(&steps).cast(), MTLDataType::Int, STEPS_CONSTANT);
            }
            let function = library.newFunctionWithName_constantValues_error(&NSString::from_str(ENTRY), &constants).map_err(|e| described("function", &e))?;
            let made = device.newComputePipelineStateWithFunction_error(&function).map_err(|e| described("pipeline", &e))?;
            if made.maxTotalThreadsPerThreadgroup() < group as usize {
                return Err(GpuError::new(GpuErrorKind::NoGpu, format!("{name} takes fewer than {group} threads a group")));
            }
            Ok(made)
        };
        let short = 63 % windows;
        let tail = match short {
            0 => None,
            _ => Some((short, pipeline(1, short as i32)?)),
        };
        Ok(Arc::new(Device {
            init: pipeline(0, 1)?,
            window: pipeline(1, windows as i32)?,
            tail,
            last: pipeline(2, 1)?,
            check: pipeline(3, 1)?,
            device,
            queues,
            next_queue: AtomicUsize::new(0),
            tuning,
            most_bytes,
            system: NSProcessInfo::processInfo().operatingSystemVersionString().to_string(),
            name,
        }))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// This GPU and the system it runs under, whose update brings a new Metal
    /// compiler: a check holds for these and no others.
    pub fn identity(&self) -> String {
        format!("{} {}", self.name, self.system)
    }

    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// How many command queues the arenas share.
    pub fn queues(&self) -> usize {
        self.queues.len()
    }

    /// The stages of one slab: (pipeline, digit_at), the top digit first.
    fn stages(&self) -> Vec<(&Pipeline, u32)> {
        let windows = self.tuning.windows;
        let mut stages = vec![(&self.init, 63)];
        let mut at = 62usize;
        loop {
            let take = windows.min(at + 1);
            let pipeline = match &self.tail {
                Some((short, tail)) if take == *short => tail,
                _ => &self.window,
            };
            stages.push((pipeline, at as u32));
            if at + 1 == take {
                break;
            }
            at -= take;
        }
        stages.push((&self.last, 0));
        stages
    }
}

/// An arena's memory, kept between batches. Freed by [`Arena::release`], by a
/// failure and with the arena.
struct Buffers {
    input: Buffer,
    accounts: Buffer,
    state: Buffer,
    out: Buffer,
    /// The most items, input bytes and account bytes it takes.
    items: u32,
    input_bytes: usize,
    account_bytes: usize,
}

impl Buffers {
    fn in_map(&self) -> *mut u8 {
        self.input.contents().as_ptr().cast()
    }

    fn accounts_map(&self) -> *mut u8 {
        self.accounts.contents().as_ptr().cast()
    }

    fn out_map(&self) -> *mut u32 {
        self.out.contents().as_ptr().cast()
    }
}

/// What's loaded for [`Arena::slab`]: how many keys, and how many items in all.
#[derive(Clone, Copy)]
struct Loaded {
    keys: u32,
    items: u64,
}

/// One caller's share of the GPU: its buffers, and a queue it takes in turn
/// with others. Not shared between threads: a caller that wants it from
/// several holds it behind a lock.
pub struct Arena {
    gpu: Arc<Device>,
    queue: usize,
    buffers: RefCell<Option<Buffers>>,
    loaded: RefCell<Option<Loaded>>,
}

// Its buffers are its own; moving it to another thread is fine, sharing it isn't (RefCell).
unsafe impl Send for Arena {}

impl Drop for Arena {
    fn drop(&mut self) {
        self.release();
    }
}

impl Arena {
    /// A new arena on `gpu`, taking the next of its queues. Its buffers are made
    /// as the first batch needs them.
    pub fn new(gpu: &Arc<Device>) -> Result<Arena, GpuError> {
        Ok(Arena {
            gpu: gpu.clone(),
            queue: gpu.next_queue.fetch_add(1, Ordering::Relaxed) % gpu.queues.len(),
            buffers: RefCell::new(None),
            loaded: RefCell::new(None),
        })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.gpu
    }

    pub fn name(&self) -> &str {
        self.gpu.name()
    }

    pub fn tuning(&self) -> Tuning {
        self.gpu.tuning
    }

    /// The most keys one [`load`](Arena::load) takes: what a buffer can hold,
    /// and at most a million.
    pub fn most_keys(&self) -> usize {
        usize::try_from(self.gpu.most_bytes / 32).unwrap_or(usize::MAX).min(MOST_KEYS)
    }

    /// The most accounts one [`load`](Arena::load) takes, in the same way.
    pub fn most_accounts(&self) -> usize {
        usize::try_from(self.gpu.most_bytes / ACCOUNT_BYTES as u64).unwrap_or(usize::MAX).min(MOST_ACCOUNTS)
    }

    /// Gives back the memory kept between batches, for when scanning rests. The next
    /// batch makes it again.
    pub fn release(&self) {
        self.loaded.borrow_mut().take();
        if let Some(buffers) = self.buffers.borrow_mut().take() {
            Self::wipe(&buffers);
        }
    }

    /// Every command buffer is waited for before it returns, so nothing of this
    /// arena's is still running. The results may hold points multiplied by a
    /// key, and the account buffer holds view keys' digits: both are wiped before
    /// the memory goes back. (The input is public keys only, and the state is
    /// cleared by each batch's last command buffer.)
    fn wipe(buffers: &Buffers) {
        // SAFETY: both are shared buffers of at least these lengths, mapped
        // while they live, and no command buffer is using them.
        unsafe {
            std::slice::from_raw_parts_mut(buffers.out_map(), buffers.items as usize * OUT_WORDS).zeroize();
            std::slice::from_raw_parts_mut(buffers.accounts_map(), buffers.account_bytes).zeroize();
        }
    }

    /// Room for at least `items` items, `input_bytes` of input and `account_bytes`
    /// of digits, made again larger if what's kept is short. Sizes go up in powers
    /// of two, so a run of growing batches makes it only a few times.
    fn ensure(&self, items: u32, input_bytes: usize, account_bytes: usize) -> Result<(), GpuError> {
        let mut kept = self.buffers.borrow_mut();
        if kept.as_ref().is_some_and(|b| b.items >= items && b.input_bytes >= input_bytes && b.account_bytes >= account_bytes) {
            return Ok(());
        }
        let (items, input_bytes, account_bytes) = match kept.take() {
            Some(old) => {
                Self::wipe(&old);
                (items.max(old.items), input_bytes.max(old.input_bytes), account_bytes.max(old.account_bytes))
            }
            None => (items, input_bytes, account_bytes),
        };
        *kept = Some(self.new_buffers(items.next_power_of_two(), input_bytes.next_power_of_two(), account_bytes.next_power_of_two())?);
        Ok(())
    }

    fn new_buffers(&self, items: u32, input_bytes: usize, account_bytes: usize) -> Result<Buffers, GpuError> {
        let device = &self.gpu.device;
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer = |bytes: usize, options: MTLResourceOptions| device.newBufferWithLength_options(bytes.max(4), options).ok_or_else(|| GpuError::failed(format!("no memory for {bytes} bytes")));
        let out_bytes = items as usize * OUT_WORDS * 4;
        Ok(Buffers {
            input: buffer(input_bytes, shared)?,
            accounts: buffer(account_bytes, shared)?,
            state: buffer(items as usize * STATE_BYTES as usize, MTLResourceOptions::StorageModePrivate)?,
            out: buffer(out_bytes, shared)?,
            items,
            input_bytes,
            account_bytes,
        })
    }

    /// Commits a command buffer that `record` fills and waits for it.
    fn submit(&self, record: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), GpuError>) -> Result<(), GpuError> {
        let command = self.gpu.queues[self.queue].commandBuffer().ok_or_else(|| failed("command buffer"))?;
        record(&command)?;
        command.commit();
        command.waitUntilCompleted();
        if command.status() == MTLCommandBufferStatus::Completed {
            return Ok(());
        }
        Err(match command.error() {
            Some(error) if not_permitted(&error) => GpuError::new(GpuErrorKind::Background, format!("not permitted: {}", error.localizedDescription())),
            Some(error) => described("command buffer", &error),
            None => failed("command buffer didn't complete"),
        })
    }

    /// Runs `stages` (a pipeline and its digit_at) over `count` items from
    /// `first_item`, reading back each item's words through `read`.
    fn run_on(&self, keys: u32, first_item: u32, count: u32, op: u32, stages: &[(&Pipeline, u32)], read: &mut dyn FnMut(&[u32])) -> Result<(), GpuError> {
        let kept = self.buffers.borrow();
        let buffers = kept.as_ref().ok_or_else(|| failed("no buffers"))?;
        let group = self.gpu.tuning.group as usize;
        let groups = (count as usize).div_ceil(group);
        let used = count as usize * STATE_BYTES as usize;
        let every = self.gpu.tuning.submit_every.max(1);
        let parts = stages.len().div_ceil(every);
        for (at, part) in stages.chunks(every).enumerate() {
            self.submit(|command| {
                let encoder = command.computeCommandEncoder().ok_or_else(|| failed("compute encoder"))?;
                // SAFETY: the buffers outlive the command buffer, which is
                // waited for, and are as long as the shader reads for `count`
                // items. The push constants are copied by Metal.
                unsafe {
                    for (buffer, index) in [(&buffers.input, INPUT_INDEX), (&buffers.state, STATE_INDEX), (&buffers.out, OUT_INDEX), (&buffers.accounts, ACCOUNTS_INDEX)] {
                        encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                    }
                    // A serial encoder, as this one is, runs each dispatch after
                    // the last, and sees what it wrote.
                    for (pipeline, digit_at) in part {
                        encoder.setComputePipelineState(pipeline);
                        let mut bytes = [0u8; PUSH_BYTES];
                        for (i, v) in [keys.max(1), first_item, count, *digit_at, op].iter().enumerate() {
                            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
                        }
                        encoder.setBytes_length_atIndex(NonNull::from(&bytes).cast::<c_void>(), PUSH_BYTES, PUSH_INDEX);
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: groups, height: 1, depth: 1 }, MTLSize { width: group, height: 1, depth: 1 });
                    }
                }
                encoder.endEncoding();
                // The working state is points multiplied by a key. It's cleared in
                // the last command buffer, for no wait of its own.
                if at + 1 == parts {
                    let blit = command.blitCommandEncoder().ok_or_else(|| failed("blit encoder"))?;
                    blit.fillBuffer_range_value(&buffers.state, NSRange { location: 0, length: used }, 0);
                    blit.endEncoding();
                }
                Ok(())
            })?;
        }
        // SAFETY: a shared buffer of at least this many words, and the command
        // buffers that wrote it have completed.
        let words = unsafe { std::slice::from_raw_parts_mut(buffers.out_map(), count as usize * OUT_WORDS) };
        read(words);
        // The results are points multiplied by a key too.
        words.zeroize();
        Ok(())
    }

    /// Any failure frees the arena's memory: what it held may not have been cleared.
    fn or_release<T>(&self, result: Result<T, GpuError>) -> Result<T, GpuError> {
        if result.is_err() {
            self.release();
        }
        result
    }

    /// Puts `keys` and the accounts' digits on the GPU, for [`slab`](Arena::slab)
    /// to take items of the grid from, as `gpu.rs`'s does: item `g` is key
    /// `g % keys.len()` with account `g / keys.len()`. [`unload`](Arena::unload)
    /// wipes the digits again.
    pub fn load(&self, accounts: &[&Digits], keys: &[[u8; 32]]) -> Result<(), GpuError> {
        let items = keys.len() as u64 * accounts.len() as u64;
        // The shader indexes keys in 32 bits.
        if items >= 1 << 31 || keys.len() > self.most_keys() || accounts.len() > self.most_accounts() {
            return Err(GpuError::failed(format!("{} keys and {} accounts in one load", keys.len(), accounts.len())));
        }
        let batch = (items as usize).min(self.gpu.tuning.batch).max(1) as u32;
        let result = self.ensure(batch, keys.len() * 32, accounts.len() * ACCOUNT_BYTES).map(|()| {
            let kept = self.buffers.borrow();
            let buffers = kept.as_ref().expect("just made");
            let input = keys.as_flattened();
            // SAFETY: shared buffers at least this long, which no command
            // buffer is using.
            unsafe {
                std::ptr::copy_nonoverlapping(input.as_ptr(), buffers.in_map(), input.len());
                let digits = std::slice::from_raw_parts_mut(buffers.accounts_map(), accounts.len() * ACCOUNT_BYTES);
                for (account, out) in accounts.iter().zip(digits.chunks_mut(ACCOUNT_BYTES)) {
                    for (i, byte) in out.iter_mut().enumerate() {
                        *byte = account.get(i) as u8;
                    }
                }
            }
        });
        *self.loaded.borrow_mut() = result.is_ok().then_some(Loaded { keys: keys.len() as u32, items });
        self.or_release(result)
    }

    /// Wipes the loaded digits from the GPU's memory.
    pub fn unload(&self) {
        self.loaded.borrow_mut().take();
        if let Some(buffers) = self.buffers.borrow().as_ref() {
            // SAFETY: a shared buffer this long, which no command buffer is using.
            unsafe { std::slice::from_raw_parts_mut(buffers.accounts_map(), buffers.account_bytes).zeroize() };
        }
    }

    /// 8aR for the `count` items of the loaded grid from `first_item`: at most
    /// [`Tuning::batch`], and within what was loaded.
    pub fn slab(&self, first_item: u64, count: usize) -> Result<Points, GpuError> {
        let loaded = (*self.loaded.borrow()).ok_or_else(|| failed("nothing loaded"))?;
        if count == 0 {
            return Ok(Points::from_words(vec![]));
        }
        if count > self.gpu.tuning.batch || first_item + count as u64 > loaded.items {
            return Err(GpuError::failed(format!("{count} items from {first_item} of {}", loaded.items)));
        }
        let stages = self.gpu.stages();
        let mut points = Zeroizing::new(Vec::with_capacity(count * OUT_WORDS));
        let result = self.run_on(loaded.keys, first_item as u32, count as u32, 0, &stages, &mut |words| points.extend_from_slice(words));
        self.or_release(result).map(|()| Points::from_words(std::mem::take(&mut *points)))
    }

    /// 8aR for every key with every account, account by account, wiping the
    /// digits after: the whole grid at once, for the vectors and the test tools.
    pub fn derive(&self, accounts: &[&Digits], keys: &[[u8; 32]]) -> Result<Points, GpuError> {
        if keys.is_empty() || accounts.is_empty() {
            return Ok(Points::from_words(vec![]));
        }
        self.load(accounts, keys)?;
        let items = keys.len() * accounts.len();
        let mut all = Zeroizing::new(Vec::with_capacity(items * OUT_WORDS));
        let batch = self.gpu.tuning.batch;
        let mut result = Ok(());
        for first in (0..items).step_by(batch) {
            match self.slab(first as u64, batch.min(items - first)) {
                Ok(points) => all.extend_from_slice(points.words()),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.unload();
        result.map(|()| Points::from_words(std::mem::take(&mut *all)))
    }

    /// One field operation on each pair: for the tests. 0 adds, 1 subtracts,
    /// 2 multiplies, 3 squares, 4 negates and 5 only reduces. At most
    /// [`Tuning::batch`] pairs.
    pub fn check_field(&self, op: u32, pairs: &[([u32; 8], [u32; 8])]) -> Result<Vec<[u32; 8]>, GpuError> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }
        if pairs.len() > self.gpu.tuning.batch {
            return Err(GpuError::failed(format!("{} pairs at once", pairs.len())));
        }
        let mut input = Vec::with_capacity(pairs.len() * 64);
        for (a, b) in pairs {
            for limb in a.iter().chain(b) {
                input.extend_from_slice(&limb.to_le_bytes());
            }
        }
        let mut out = Vec::with_capacity(pairs.len());
        self.loaded.borrow_mut().take();
        let result = self.ensure(pairs.len() as u32, input.len(), ACCOUNT_BYTES).and_then(|()| {
            {
                let kept = self.buffers.borrow();
                let buffers = kept.as_ref().expect("just made");
                // SAFETY: a shared buffer at least this long, which no command
                // buffer is using.
                unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), buffers.in_map(), input.len()) };
            }
            self.run_on(1, 0, pairs.len() as u32, op, &[(&self.gpu.check, 0)], &mut |words| {
                for word in words.as_chunks::<OUT_WORDS>().0 {
                    let mut limbs = [0; 8];
                    limbs.copy_from_slice(&word[..8]);
                    out.push(limbs);
                }
            })
        });
        self.or_release(result).map(|()| out)
    }
}

/// Whether Metal refused a command buffer because the app isn't in the foreground.
fn not_permitted(error: &NSError) -> bool {
    // SAFETY: a static Metal defines, read only.
    let domain = unsafe { MTLCommandBufferErrorDomain };
    *error.domain() == *domain && error.code() == MTLCommandBufferError::NotPermitted.0 as isize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPIRV-Cross numbers the shader's arguments itself: the constants above
    /// must be what it wrote, or a buffer would be bound in another's place.
    #[test]
    fn the_arguments_are_where_scan_metal_has_them() {
        let entry = MSL.lines().find(|l| l.starts_with(&format!("kernel void {ENTRY}("))).expect("the entry point");
        for (name, index) in [("pc", PUSH_INDEX), ("input_keys", INPUT_INDEX), ("state", STATE_INDEX), ("output_keys", OUT_INDEX), ("accounts", ACCOUNTS_INDEX)] {
            assert!(entry.contains(&format!("& {name} [[buffer({index})]]")), "{name} at {index} in {entry}");
        }
        for (name, index) in [("STAGE", STAGE_CONSTANT), ("STEPS", STEPS_CONSTANT)] {
            assert!(MSL.contains(&format!("constant int {name}_tmp [[function_constant({index})]];")), "{name} is function constant {index}");
        }
        assert!(MSL.contains(&format!("_tmp [[function_constant({GROUP_CONSTANT})]];")), "the group size is function constant {GROUP_CONSTANT}");
    }

    /// Only Metal's refusal of an app in the background is taken for one.
    #[test]
    fn only_not_permitted_is_the_background() {
        // SAFETY: a static Metal defines, read only.
        let domain = unsafe { MTLCommandBufferErrorDomain };
        let error = |domain: &NSString, code: MTLCommandBufferError| NSError::new(code.0 as isize, domain);
        assert!(not_permitted(&error(domain, MTLCommandBufferError::NotPermitted)));
        assert!(!not_permitted(&error(domain, MTLCommandBufferError::Timeout)));
        assert!(!not_permitted(&error(domain, MTLCommandBufferError::Internal)));
        assert!(!not_permitted(&error(&NSString::from_str("Elsewhere"), MTLCommandBufferError::NotPermitted)));
    }
}
