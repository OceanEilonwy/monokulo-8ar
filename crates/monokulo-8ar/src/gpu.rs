//! 8aR on the GPU, in Vulkan compute (`shaders/scan.comp`).
//!
//! The GPU does the curve arithmetic, inverts each point's Z and hands back the
//! compressed points; the CPU does everything after. A key that isn't on the
//! curve comes back marked, and so does a point the GPU's own check finds off
//! it, which only a fault makes. Any Vulkan failure is an error for the caller,
//! which then scans without the prefilter. Nothing here panics: a driver's behaviour
//! is outside what the wallet can vouch for.
//!
//! A [`Device`] is the GPU, set up once a process: the instance, the logical
//! device and its compute queues, the pipelines and their cache. An [`Arena`] is
//! what one caller works in: its buffers, command buffer and fence. Arenas run
//! side by side; they share only a queue's lock, for the moment of a submit,
//! and not even that where the GPU has a queue each for them.
//!
//! The view keys' digits are in an arena's account buffer while it works, and
//! wiped from it after (and before the memory goes back). They, the shader's
//! working state and the results are in memory the driver manages: the state is
//! cleared after each batch, the results once read, and the CPU's copies of the
//! digits are wiped, but a driver's own copies can't be reached.

use std::{
    cell::RefCell,
    ffi::CStr,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::vk;
use zeroize::{Zeroize, Zeroizing};

/// A secret scalar, the view key, as signed base-16 digits: what the GPU is given.
#[derive(Clone)]
pub struct Digits([i8; 64]);

impl Drop for Digits {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl Digits {
    /// Digit `i`, the least significant first.
    pub fn get(&self, i: usize) -> i8 {
        self.0[i]
    }

    pub fn new(scalar: &[u8; 32]) -> Digits {
        assert!(scalar[31] <= 127);
        let mut d = [0i8; 64];
        for i in 0..32 {
            d[2 * i] = (scalar[i] & 15) as i8;
            d[2 * i + 1] = ((scalar[i] >> 4) & 15) as i8;
        }
        for i in 0..63 {
            let c = (d[i] + 8) >> 4;
            d[i] -= c << 4;
            d[i + 1] += c;
        }
        Digits(d)
    }
}

const SPIRV: &[u8] = include_bytes!("../shaders/scan.spv");
/// A work item's state on the GPU: 300 words.
const STATE_BYTES: u64 = 300 * 4;
/// A work item's result: the compressed point, then a flag: 1 if the key was off
/// the curve, 2 if the point failed the GPU's own check.
const OUT_WORDS: usize = 9;
/// An account's digits on the GPU: 64 signed bytes.
const ACCOUNT_BYTES: usize = 64;
const CACHE_FILE: &str = "scan-gpu.cache";
/// A submit taking this long means something's wrong: the driver's watchdog should have acted.
const FENCE_TIMEOUT_NANOS: u64 = 60_000_000_000;
/// The push constants: tx_count, first_item, count, digit_at and op.
const PUSH_BYTES: u32 = 20;
/// The most compute queues taken from the GPU, for arenas to share round-robin.
const MOST_QUEUES: u32 = 4;
/// Storage buffers: keys, state, results, account digits.
const BINDINGS: u32 = 4;
/// The most keys one load takes, 32 MB of them, however many more a GPU's
/// buffers could hold: the caller splits a bigger load.
const MOST_KEYS: usize = 1 << 20;
/// The most accounts one load takes: 64 MB of digits.
const MOST_ACCOUNTS: usize = 1 << 20;

/// How the work is cut up.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    /// The most work items in one go. An item's state is 1,200 bytes on the GPU.
    pub batch: usize,
    /// Digits of the view key taken per dispatch.
    pub windows: usize,
    /// Dispatches per submit.
    pub submit_every: usize,
    /// Items per workgroup.
    pub group: u32,
}

/// Tuned on a Mali-G715: 16 windows hold a key's state in the shader longest, and a dispatch to about 60 ms.
pub const TUNING: Tuning = Tuning { batch: 32_768, windows: 16, submit_every: 1, group: 64 };

/// A discrete GPU has no dispatch near its watchdog, so a batch is one submit. On an
/// RX 6800 XT a batch costs at least 3.7 ms however small: 8,192 keys were slower
/// overall, and 16,384 no faster.
pub const DISCRETE_TUNING: Tuning = Tuning { submit_every: 8, ..TUNING };

/// What went wrong, for the log. Never holds key material.
#[derive(Debug)]
pub struct GpuError {
    pub what: String,
    pub kind: GpuErrorKind,
}

/// Whether there was a GPU to use at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuErrorKind {
    /// No Vulkan loader or driver.
    NoVulkan,
    /// Vulkan, but no hardware GPU that can do this work.
    NoGpu,
    /// A GPU that went wrong.
    Failed,
}

impl GpuError {
    fn new(kind: GpuErrorKind, what: String) -> GpuError {
        GpuError { what, kind }
    }

    fn failed(what: String) -> GpuError {
        GpuError::new(GpuErrorKind::Failed, what)
    }
}

fn failed(what: &str) -> impl Fn(vk::Result) -> GpuError + '_ {
    move |e| GpuError::failed(format!("{what}: {e}"))
}

/// One item of [`Points`].
pub enum Derived {
    /// 8aR, compressed. Wipe it when done.
    Point([u8; 32]),
    /// The key wasn't on the curve, the GPU says: Monero takes it as the identity.
    OffCurve,
    /// Not an answer the GPU could have got right: it's faulty.
    Wrong,
}

/// What the GPU worked out for a batch of items. Copied out whole first, as
/// reading the GPU's memory a word at a time is slow on a phone, and decoded by
/// whichever thread takes each point.
pub struct Points(Zeroizing<Vec<u32>>);

impl Points {
    #[cfg(test)]
    pub(crate) fn from_words(words: Vec<u32>) -> Points {
        Points(Zeroizing::new(words))
    }

    /// Item `i`'s point, compressed, or `None` if its key wasn't on the curve or
    /// the GPU says it went wrong. Wipe it when done: it's the key's shared
    /// secret with the wallet.
    pub fn get(&self, i: usize) -> Option<[u8; 32]> {
        match self.derived(i) {
            Derived::Point(bytes) => Some(bytes),
            Derived::OffCurve | Derived::Wrong => None,
        }
    }

    /// Item `i` as the GPU handed it back, with what the CPU can check cheaply
    /// held against it: a flag it knows, and a point that's a canonical encoding
    /// and not y = 0. That point has order 4, which 8aR never has; it's what a
    /// zero Z comes out as.
    pub fn derived(&self, i: usize) -> Derived {
        let word = &self.0[i * OUT_WORDS..(i + 1) * OUT_WORDS];
        match word[8] {
            0 => {
                let mut bytes = [0; 32];
                for (limb, out) in word[..8].iter().zip(bytes.as_chunks_mut::<4>().0) {
                    *out = limb.to_le_bytes();
                }
                let y_is_zero = bytes[..31].iter().all(|b| *b == 0) && bytes[31] & 0x7f == 0;
                if crate::canonical(&bytes) && !y_is_zero {
                    Derived::Point(bytes)
                } else {
                    bytes.zeroize();
                    Derived::Wrong
                }
            }
            1 => Derived::OffCurve,
            _ => Derived::Wrong,
        }
    }

    pub fn len(&self) -> usize {
        self.0.len() / OUT_WORDS
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The GPU, set up once: shared by every [`Arena`].
pub struct Device {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    family: u32,
    /// Vulkan wants a queue's submits one at a time; the lock is held for nothing else.
    queues: Vec<Mutex<vk::Queue>>,
    next_queue: AtomicUsize,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    cache: vk::PipelineCache,
    init: vk::Pipeline,
    window: vk::Pipeline,
    tail: Option<(usize, vk::Pipeline)>,
    last: vk::Pipeline,
    check: vk::Pipeline,
    tuning: Tuning,
    /// The most bytes a storage buffer's binding may span.
    range: u64,
    name: String,
    driver: u32,
}

// Vulkan's device-level handles may be used from any thread; what needs one at a
// time (a queue) is behind a lock, and what an arena records is its own.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Every arena holds the device, so none is left to use a queue.
            let _ = self.device.device_wait_idle();
            for pipeline in [self.init, self.window, self.last, self.check] {
                self.device.destroy_pipeline(pipeline, None);
            }
            if let Some((_, tail)) = self.tail {
                self.device.destroy_pipeline(tail, None);
            }
            self.device.destroy_pipeline_cache(self.cache, None);
            self.device.destroy_shader_module(self.module, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Device {
    /// Sets up the GPU, compiling the shader: up to about 20 s on a phone the
    /// first time, and little once `cache_dir` holds the result.
    ///
    /// `any_kind` also takes software renderers, which only the test tool wants:
    /// they'd be a slower CPU.
    pub fn start(cache_dir: Option<PathBuf>, tuning: Tuning, any_kind: bool) -> Result<Arc<Device>, GpuError> {
        Self::start_tuned(cache_dir, Some(tuning), any_kind)
    }

    /// [`start`](Device::start), tuned for the GPU it finds: what [`crate::warm_up`] uses.
    pub fn start_for_device(cache_dir: Option<PathBuf>, any_kind: bool) -> Result<Arc<Device>, GpuError> {
        Self::start_tuned(cache_dir, None, any_kind)
    }

    fn start_tuned(cache_dir: Option<PathBuf>, tuning: Option<Tuning>, any_kind: bool) -> Result<Arc<Device>, GpuError> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| GpuError::new(GpuErrorKind::NoVulkan, format!("no Vulkan: {e}")))?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
            // Without a driver, the loader can't make an instance.
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(|e| GpuError::new(GpuErrorKind::NoVulkan, format!("instance: {e}")))?;
            Self::on_instance(&entry, instance, cache_dir, tuning, any_kind).map(Arc::new)
        }
    }

    /// Takes `instance`, destroying it on failure as well.
    unsafe fn on_instance(entry: &ash::Entry, instance: ash::Instance, cache_dir: Option<PathBuf>, tuning: Option<Tuning>, any_kind: bool) -> Result<Device, GpuError> {
        unsafe {
            let found = Self::choose(&instance, tuning.unwrap_or(TUNING), any_kind);
            let (physical, family, queue_count, name) = match found {
                Ok(found) => found,
                Err(e) => {
                    instance.destroy_instance(None);
                    return Err(e);
                }
            };
            let properties = instance.get_physical_device_properties(physical);
            let driver = properties.driver_version;
            let range = u64::from(properties.limits.max_storage_buffer_range);
            let tuning = tuning.unwrap_or(match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DISCRETE_TUNING,
                _ => TUNING,
            });
            let queue_count = queue_count.clamp(1, MOST_QUEUES);
            let priorities = vec![1.0; queue_count as usize];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let device = match instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues), None) {
                Ok(device) => device,
                Err(e) => {
                    instance.destroy_instance(None);
                    return Err(failed("device")(e));
                }
            };
            let queues = (0..queue_count).map(|i| Mutex::new(device.get_device_queue(family, i))).collect();
            let memory = instance.get_physical_device_memory_properties(physical);

            // From here the Device owns it all; its drop frees what's been made, as
            // Vulkan's destroy calls ignore null handles.
            let mut gpu = Device {
                _entry: entry.clone(),
                instance,
                device,
                memory,
                family,
                queues,
                next_queue: AtomicUsize::new(0),
                set_layout: vk::DescriptorSetLayout::null(),
                layout: vk::PipelineLayout::null(),
                module: vk::ShaderModule::null(),
                cache: vk::PipelineCache::null(),
                init: vk::Pipeline::null(),
                window: vk::Pipeline::null(),
                tail: None,
                last: vk::Pipeline::null(),
                check: vk::Pipeline::null(),
                tuning,
                range,
                name,
                driver,
            };
            gpu.build(cache_dir)?;
            Ok(gpu)
        }
    }

    /// The GPU to use, its compute queue family, how many queues that has, and its name.
    unsafe fn choose(instance: &ash::Instance, tuning: Tuning, any_kind: bool) -> Result<(vk::PhysicalDevice, u32, u32, String), GpuError> {
        unsafe {
            let rank = |kind: vk::PhysicalDeviceType| match kind {
                vk::PhysicalDeviceType::DISCRETE_GPU => Some(0),
                vk::PhysicalDeviceType::INTEGRATED_GPU => Some(1),
                _ if any_kind => Some(2),
                _ => None,
            };
            let physical = instance
                .enumerate_physical_devices()
                .map_err(failed("devices"))?
                .into_iter()
                .filter_map(|d| rank(instance.get_physical_device_properties(d).device_type).map(|r| (r, d)))
                .min_by_key(|(r, _)| *r)
                .map(|(_, d)| d)
                .ok_or_else(|| GpuError::new(GpuErrorKind::NoGpu, "no GPU".into()))?;
            let properties = instance.get_physical_device_properties(physical);
            let name = CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy().into_owned();
            let limits = properties.limits;
            if u64::from(limits.max_storage_buffer_range) < tuning.batch as u64 * STATE_BYTES
                || limits.max_compute_work_group_invocations < tuning.group
                || limits.max_compute_work_group_size[0] < tuning.group
                || limits.max_push_constants_size < PUSH_BYTES
                || limits.max_per_stage_descriptor_storage_buffers < BINDINGS
            {
                return Err(GpuError::new(GpuErrorKind::NoGpu, format!("{name} is too small for this work")));
            }
            let families = instance.get_physical_device_queue_family_properties(physical);
            let family = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| GpuError::new(GpuErrorKind::NoGpu, format!("{name} has no compute queue")))?;
            Ok((physical, family as u32, families[family].queue_count, name))
        }
    }

    unsafe fn build(&mut self, cache_dir: Option<PathBuf>) -> Result<(), GpuError> {
        unsafe {
            let device = &self.device;
            let bindings: Vec<_> = (0..BINDINGS)
                .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            self.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(failed("set layout"))?;
            let layouts = [self.set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: PUSH_BYTES }];
            self.layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(failed("pipeline layout"))?;
            let code: Vec<u32> = SPIRV.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            self.module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(failed("shader"))?;

            // A cache the driver finds stale or foreign it ignores.
            let cache_path = cache_dir.as_ref().map(|dir| dir.join(CACHE_FILE));
            let cached = cache_path.as_ref().and_then(|path| std::fs::read(path).ok()).unwrap_or_default();
            self.cache = device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(&cached), None)
                .or_else(|_| device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None))
                .map_err(failed("pipeline cache"))?;

            let group = self.tuning.group;
            let windows = self.tuning.windows;
            self.init = self.pipeline(group, 0, 1)?;
            self.window = self.pipeline(group, 1, windows as u32)?;
            let short = 63 % windows;
            if short != 0 {
                self.tail = Some((short, self.pipeline(group, 1, short as u32)?));
            }
            self.last = self.pipeline(group, 2, 1)?;
            self.check = self.pipeline(group, 3, 1)?;

            if let (Some(path), Ok(data)) = (cache_path, self.device.get_pipeline_cache_data(self.cache)) {
                // Compiled shader code, with no key material: digits are in a buffer.
                let spare = path.with_extension("tmp");
                if std::fs::write(&spare, data).is_ok() {
                    let _ = std::fs::rename(&spare, &path);
                }
            }
            Ok(())
        }
    }

    unsafe fn pipeline(&self, group: u32, stage: u32, steps: u32) -> Result<vk::Pipeline, GpuError> {
        unsafe {
            let bytes: Vec<u8> = [group, stage, steps].iter().flat_map(|v| v.to_ne_bytes()).collect();
            let entries: Vec<_> = (0..3).map(|i| vk::SpecializationMapEntry { constant_id: i, offset: i * 4, size: 4 }).collect();
            let spec = vk::SpecializationInfo::default().map_entries(&entries).data(&bytes);
            let shader = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(self.module).name(c"main").specialization_info(&spec);
            let info = vk::ComputePipelineCreateInfo::default().stage(shader).layout(self.layout);
            let made = self.device.create_compute_pipelines(self.cache, &[info], None).map_err(|e| GpuError::failed(format!("pipeline: {}", e.1)))?;
            Ok(made[0])
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// This GPU and its driver: a check holds for these and no others.
    pub fn identity(&self) -> String {
        format!("{} {:#x}", self.name, self.driver)
    }

    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// How many compute queues the arenas share.
    pub fn queues(&self) -> usize {
        self.queues.len()
    }

    /// The stages of one slab: (pipeline, digit_at), the top digit first.
    fn stages(&self) -> Vec<(vk::Pipeline, u32)> {
        let windows = self.tuning.windows;
        let mut stages = vec![(self.init, 63)];
        let mut at = 62usize;
        loop {
            let take = windows.min(at + 1);
            let pipeline = match self.tail {
                Some((short, tail)) if take == short => tail,
                _ => self.window,
            };
            stages.push((pipeline, at as u32));
            if at + 1 == take {
                break;
            }
            at -= take;
        }
        stages.push((self.last, 0));
        stages
    }
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// An arena's memory, kept between batches. Freed by [`Arena::release`], by a
/// failure and with the arena.
struct Buffers {
    input: Buffer,
    accounts: Buffer,
    state: Buffer,
    out: Buffer,
    set: vk::DescriptorSet,
    /// The most items, input bytes and account bytes it takes.
    items: u32,
    input_bytes: usize,
    account_bytes: usize,
    in_map: *mut u8,
    accounts_map: *mut u8,
    out_map: *mut u32,
}

/// What's loaded for [`Arena::slab`]: how many keys, and how many items in all.
#[derive(Clone, Copy)]
struct Loaded {
    keys: u32,
    items: u64,
}

/// One caller's share of the GPU: buffers, a command buffer and a fence. It
/// works while other arenas do. Not shared between threads: a caller that
/// wants it from several holds it behind a lock.
pub struct Arena {
    gpu: Arc<Device>,
    queue: usize,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorPool,
    buffers: RefCell<Option<Buffers>>,
    loaded: RefCell<Option<Loaded>>,
}

// Its handles are its own; moving it to another thread is fine, sharing it isn't (RefCell).
unsafe impl Send for Arena {}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            self.release();
            let device = &self.gpu.device;
            device.destroy_descriptor_pool(self.descriptors, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

impl Arena {
    /// A new arena on `gpu`, taking the next of its queues. Its buffers are made
    /// as the first batch needs them.
    pub fn new(gpu: &Arc<Device>) -> Result<Arena, GpuError> {
        unsafe {
            let device = &gpu.device;
            let mut arena = Arena {
                gpu: gpu.clone(),
                queue: gpu.next_queue.fetch_add(1, Ordering::Relaxed) % gpu.queues.len(),
                pool: vk::CommandPool::null(),
                command: vk::CommandBuffer::null(),
                fence: vk::Fence::null(),
                descriptors: vk::DescriptorPool::null(),
                buffers: RefCell::new(None),
                loaded: RefCell::new(None),
            };
            // On an error, the arena's drop frees what's been made.
            arena.pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(gpu.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .map_err(failed("command pool"))?;
            arena.command = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(arena.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .map_err(failed("command buffer"))?[0];
            arena.fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(failed("fence"))?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: BINDINGS }];
            arena.descriptors = device
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET).pool_sizes(&sizes), None)
                .map_err(failed("descriptor pool"))?;
            Ok(arena)
        }
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

    /// The most keys one [`load`](Arena::load) takes: what a buffer binding can
    /// span, and at most a million.
    pub fn most_keys(&self) -> usize {
        usize::try_from(self.gpu.range / 32).unwrap_or(usize::MAX).min(MOST_KEYS)
    }

    /// The most accounts one [`load`](Arena::load) takes, in the same way.
    pub fn most_accounts(&self) -> usize {
        usize::try_from(self.gpu.range / ACCOUNT_BYTES as u64).unwrap_or(usize::MAX).min(MOST_ACCOUNTS)
    }

    /// A buffer in the first memory type with all of one of `wanted`'s flags.
    unsafe fn buffer(&self, size: u64, wanted: &[vk::MemoryPropertyFlags]) -> Result<Buffer, GpuError> {
        unsafe {
            let device = &self.gpu.device;
            let memory = &self.gpu.memory;
            let info = vk::BufferCreateInfo::default().size(size.max(4)).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device.create_buffer(&info, None).map_err(failed("buffer"))?;
            let needs = device.get_buffer_memory_requirements(buffer);
            for flags in wanted {
                for i in 0..memory.memory_type_count {
                    let kind = memory.memory_types[i as usize];
                    if needs.memory_type_bits & (1 << i) != 0 && kind.property_flags.contains(*flags) {
                        let alloc = vk::MemoryAllocateInfo::default().allocation_size(needs.size).memory_type_index(i);
                        if let Ok(memory) = device.allocate_memory(&alloc, None) {
                            if let Err(e) = device.bind_buffer_memory(buffer, memory, 0) {
                                device.destroy_buffer(buffer, None);
                                device.free_memory(memory, None);
                                return Err(failed("binding memory")(e));
                            }
                            return Ok(Buffer { buffer, memory });
                        }
                    }
                }
            }
            device.destroy_buffer(buffer, None);
            Err(GpuError::failed(format!("no memory for {size} bytes")))
        }
    }

    unsafe fn free(&self, buffer: &Buffer) {
        unsafe {
            self.gpu.device.destroy_buffer(buffer.buffer, None);
            self.gpu.device.free_memory(buffer.memory, None);
        }
    }

    /// Gives back the memory kept between batches, for when scanning rests. The next
    /// batch makes it again.
    pub fn release(&self) {
        self.loaded.borrow_mut().take();
        if let Some(buffers) = self.buffers.borrow_mut().take() {
            unsafe { self.free_buffers(buffers) };
        }
    }

    unsafe fn free_buffers(&self, buffers: Buffers) {
        unsafe {
            // Every submit is waited for before it returns, so nothing of this
            // arena's is still running. Results may hold points multiplied by a
            // key, and the account buffer holds view keys' digits: both are wiped
            // before the memory goes back. (The input is public keys only.)
            // Freeing unmaps.
            if !buffers.out_map.is_null() {
                std::slice::from_raw_parts_mut(buffers.out_map, buffers.items as usize * OUT_WORDS).zeroize();
            }
            if !buffers.accounts_map.is_null() {
                std::slice::from_raw_parts_mut(buffers.accounts_map, buffers.account_bytes).zeroize();
            }
            let _ = self.gpu.device.free_descriptor_sets(self.descriptors, &[buffers.set]);
            for buffer in [&buffers.input, &buffers.accounts, &buffers.state, &buffers.out] {
                self.free(buffer);
            }
        }
    }

    /// Room for at least `items` items, `input_bytes` of input and `account_bytes`
    /// of digits, made again larger if what's kept is short. Sizes go up in powers
    /// of two, so a run of growing batches makes it only a few times.
    unsafe fn ensure(&self, items: u32, input_bytes: usize, account_bytes: usize) -> Result<(), GpuError> {
        unsafe {
            let mut kept = self.buffers.borrow_mut();
            if kept.as_ref().is_some_and(|b| b.items >= items && b.input_bytes >= input_bytes && b.account_bytes >= account_bytes) {
                return Ok(());
            }
            let (items, input_bytes, account_bytes) = match kept.take() {
                Some(old) => {
                    let wanted = (items.max(old.items), input_bytes.max(old.input_bytes), account_bytes.max(old.account_bytes));
                    self.free_buffers(old);
                    wanted
                }
                None => (items, input_bytes, account_bytes),
            };
            *kept = Some(self.new_buffers(items.next_power_of_two(), input_bytes.next_power_of_two(), account_bytes.next_power_of_two())?);
            Ok(())
        }
    }

    unsafe fn new_buffers(&self, items: u32, input_bytes: usize, account_bytes: usize) -> Result<Buffers, GpuError> {
        unsafe {
            let device = &self.gpu.device;
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let cached = visible | vk::MemoryPropertyFlags::HOST_CACHED;
            let made = [
                self.buffer(input_bytes as u64, &[visible]),
                self.buffer(account_bytes as u64, &[visible]),
                self.buffer(u64::from(items) * STATE_BYTES, &[local, vk::MemoryPropertyFlags::empty()]),
                self.buffer(u64::from(items) * OUT_WORDS as u64 * 4, &[cached, visible]),
            ];
            if made.iter().any(Result::is_err) {
                let mut problem = None;
                for buffer in made {
                    match buffer {
                        Ok(buffer) => self.free(&buffer),
                        Err(e) => problem = Some(e),
                    }
                }
                return Err(problem.expect("one of them failed"));
            }
            let [input, accounts, state, out] = made.map(|b| b.expect("all made"));
            let layouts = [self.gpu.set_layout];
            let set = match device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.descriptors).set_layouts(&layouts)) {
                Ok(sets) => sets[0],
                Err(e) => {
                    for buffer in [&input, &accounts, &state, &out] {
                        self.free(buffer);
                    }
                    return Err(GpuError::failed(format!("descriptor set: {e}")));
                }
            };
            // Bindings in the shader's order: keys, state, results, accounts.
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [&input, &state, &out, &accounts].iter().map(|b| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }]).collect();
            let writes: Vec<_> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            device.update_descriptor_sets(&writes, &[]);
            let mut buffers = Buffers { input, accounts, state, out, set, items, input_bytes, account_bytes, in_map: std::ptr::null_mut(), accounts_map: std::ptr::null_mut(), out_map: std::ptr::null_mut() };
            // Mapped for as long as they're kept: the rest is the same memory each batch.
            let map = |b: &Buffer| device.map_memory(b.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty());
            match (map(&buffers.input), map(&buffers.accounts), map(&buffers.out)) {
                (Ok(i), Ok(a), Ok(o)) => {
                    buffers.in_map = i as *mut u8;
                    buffers.accounts_map = a as *mut u8;
                    buffers.out_map = o as *mut u32;
                    Ok(buffers)
                }
                (i, a, o) => {
                    let e = [i, a, o].into_iter().find_map(Result::err).expect("one failed");
                    self.free_buffers(buffers);
                    Err(failed("mapping")(e))
                }
            }
        }
    }

    unsafe fn submit(&self, record: impl Fn(vk::CommandBuffer)) -> Result<(), GpuError> {
        unsafe {
            let device = &self.gpu.device;
            device.begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(failed("recording"))?;
            record(self.command);
            device.end_command_buffer(self.command).map_err(failed("recording"))?;
            let commands = [self.command];
            let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
            {
                // The queue's lock, only for the submit itself.
                let queue = self.gpu.queues[self.queue].lock().map_err(|_| GpuError::failed("queue lock".into()))?;
                device.queue_submit(*queue, &submit, self.fence).map_err(failed("submit"))?;
            }
            if let Err(e) = device.wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NANOS) {
                // Still running, or lost: wait the queue out before anything of
                // this arena's is freed.
                if let Ok(queue) = self.gpu.queues[self.queue].lock() {
                    let _ = device.queue_wait_idle(*queue);
                }
                return Err(failed("waiting")(e));
            }
            device.reset_fences(&[self.fence]).map_err(failed("fence"))?;
            device.reset_command_buffer(self.command, vk::CommandBufferResetFlags::RELEASE_RESOURCES).map_err(failed("reset"))?;
            Ok(())
        }
    }

    /// Runs `stages` (a pipeline and its digit_at) over `count` items from
    /// `first_item`, reading back each item's words through `read`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_on(&self, keys: u32, first_item: u32, count: u32, op: u32, stages: &[(vk::Pipeline, u32)], read: &mut dyn FnMut(&[u32])) -> Result<(), GpuError> {
        unsafe {
            let device = &self.gpu.device;
            let kept = self.buffers.borrow();
            let buffers = kept.as_ref().ok_or_else(|| GpuError::failed("no buffers".into()))?;
            let groups = count.div_ceil(self.gpu.tuning.group);
            let sets = [buffers.set];
            let used = u64::from(count) * STATE_BYTES;
            let every = self.gpu.tuning.submit_every.max(1);
            let parts = stages.len().div_ceil(every);
            for (at, part) in stages.chunks(every).enumerate() {
                self.submit(|command| {
                    device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, self.gpu.layout, 0, &sets, &[]);
                    for (pipeline, digit_at) in part {
                        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, *pipeline);
                        let mut bytes = [0u8; PUSH_BYTES as usize];
                        for (i, v) in [keys.max(1), first_item, count, *digit_at, op].iter().enumerate() {
                            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
                        }
                        device.cmd_push_constants(command, self.gpu.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                        device.cmd_dispatch(command, groups, 1, 1);
                        let barrier = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_READ)];
                        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                    }
                    // The working state is points multiplied by a key. It's cleared in the
                    // last submit, for no wait of its own, and the next batch starts after that.
                    if at + 1 == parts {
                        device.cmd_fill_buffer(command, buffers.state.buffer, 0, used, 0);
                        let barrier = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                    }
                })?;
            }
            let words = std::slice::from_raw_parts_mut(buffers.out_map, count as usize * OUT_WORDS);
            read(words);
            // The results are points multiplied by a key too.
            words.fill(0);
            Ok(())
        }
    }

    /// Any failure frees the arena's memory: what it held may not have been cleared.
    fn or_release<T>(&self, result: Result<T, GpuError>) -> Result<T, GpuError> {
        if result.is_err() {
            self.release();
        }
        result
    }

    /// Puts `keys` (canonical; fewer than 2^31 items in all, at most
    /// [`most_keys`](Arena::most_keys) keys and [`most_accounts`](Arena::most_accounts)
    /// accounts) and the accounts' digits on the GPU, for [`slab`](Arena::slab)
    /// to take items of the grid from: item `g` is key `g % keys.len()` with
    /// account `g / keys.len()`. [`unload`](Arena::unload) wipes the digits again.
    pub fn load(&self, accounts: &[&Digits], keys: &[[u8; 32]]) -> Result<(), GpuError> {
        let items = keys.len() as u64 * accounts.len() as u64;
        // A binding past the GPU's range is undefined, and the shader indexes keys in 32 bits.
        if items >= 1 << 31 || keys.len() > self.most_keys() || accounts.len() > self.most_accounts() {
            return Err(GpuError::failed(format!("{} keys and {} accounts in one load", keys.len(), accounts.len())));
        }
        let result = unsafe {
            let batch = (items as usize).min(self.gpu.tuning.batch).max(1) as u32;
            self.ensure(batch, keys.len() * 32, accounts.len() * ACCOUNT_BYTES).map(|()| {
                let kept = self.buffers.borrow();
                let buffers = kept.as_ref().expect("just made");
                let input = keys.as_flattened();
                std::ptr::copy_nonoverlapping(input.as_ptr(), buffers.in_map, input.len());
                let digits = std::slice::from_raw_parts_mut(buffers.accounts_map, accounts.len() * ACCOUNT_BYTES);
                for (account, out) in accounts.iter().zip(digits.chunks_mut(ACCOUNT_BYTES)) {
                    for (i, byte) in out.iter_mut().enumerate() {
                        *byte = account.get(i) as u8;
                    }
                }
            })
        };
        *self.loaded.borrow_mut() = result.is_ok().then_some(Loaded { keys: keys.len() as u32, items });
        self.or_release(result)
    }

    /// Wipes the loaded digits from the GPU's memory.
    pub fn unload(&self) {
        self.loaded.borrow_mut().take();
        if let Some(buffers) = self.buffers.borrow().as_ref() {
            unsafe { std::slice::from_raw_parts_mut(buffers.accounts_map, buffers.account_bytes).zeroize() };
        }
    }

    /// 8aR for the `count` items of the loaded grid from `first_item`: at most
    /// [`Tuning::batch`], and within what was loaded.
    pub fn slab(&self, first_item: u64, count: usize) -> Result<Points, GpuError> {
        let loaded = (*self.loaded.borrow()).ok_or_else(|| GpuError::failed("nothing loaded".into()))?;
        if count == 0 {
            return Ok(Points(Zeroizing::new(vec![])));
        }
        if count > self.gpu.tuning.batch || first_item + count as u64 > loaded.items {
            return Err(GpuError::failed(format!("{count} items from {first_item} of {}", loaded.items)));
        }
        let stages = self.gpu.stages();
        let mut points = Zeroizing::new(Vec::with_capacity(count * OUT_WORDS));
        let result = unsafe { self.run_on(loaded.keys, first_item as u32, count as u32, 0, &stages, &mut |words| points.extend_from_slice(words)) };
        self.or_release(result).map(|()| Points(points))
    }

    /// 8aR for every key with every account, account by account, wiping the
    /// digits after: the whole grid at once, for the vectors and the test tools.
    pub fn derive(&self, accounts: &[&Digits], keys: &[[u8; 32]]) -> Result<Points, GpuError> {
        if keys.is_empty() || accounts.is_empty() {
            return Ok(Points(Zeroizing::new(vec![])));
        }
        self.load(accounts, keys)?;
        let items = keys.len() * accounts.len();
        let mut all = Zeroizing::new(Vec::with_capacity(items * OUT_WORDS));
        let batch = self.gpu.tuning.batch;
        let mut result = Ok(());
        for first in (0..items).step_by(batch) {
            match self.slab(first as u64, batch.min(items - first)) {
                Ok(points) => all.extend_from_slice(&points.0),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.unload();
        result.map(|()| Points(all))
    }

    /// One field operation on each pair: for the tests. 0 adds, 1 subtracts,
    /// 2 multiplies, 3 squares, 4 negates and 5 only reduces. At most
    /// [`Tuning::batch`] pairs.
    pub fn check_field(&self, op: u32, pairs: &[([u32; 8], [u32; 8])]) -> Result<Vec<[u32; 8]>, GpuError> {
        if pairs.is_empty() {
            // Nothing to dispatch, and clearing no bytes of state isn't allowed.
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
        let result = unsafe {
            self.ensure(pairs.len() as u32, input.len(), ACCOUNT_BYTES).and_then(|()| {
                {
                    let kept = self.buffers.borrow();
                    let buffers = kept.as_ref().expect("just made");
                    std::ptr::copy_nonoverlapping(input.as_ptr(), buffers.in_map, input.len());
                }
                self.run_on(1, 0, pairs.len() as u32, op, &[(self.gpu.check, 0)], &mut |words| {
                    for word in words.as_chunks::<OUT_WORDS>().0 {
                        let mut limbs = [0; 8];
                        limbs.copy_from_slice(&word[..8]);
                        out.push(limbs);
                    }
                })
            })
        };
        self.or_release(result).map(|()| out)
    }
}
