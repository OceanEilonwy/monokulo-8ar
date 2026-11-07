//! 8aR on the GPU, in Vulkan compute (`shaders/scan.comp`).
//!
//! The GPU does the curve arithmetic, inverts each point's Z and hands back the
//! compressed points; the CPU does everything after. A key that isn't on the
//! curve comes back marked. Any Vulkan failure is an error for the caller, which
//! then scans without the prefilter. Nothing here panics: a driver's behaviour
//! is outside what the wallet can vouch for.
//!
//! The view key's digits reach the GPU as push constants. They and the shader's
//! working state are in memory the driver manages: the buffers are cleared after
//! each batch (the mapped results again before they're freed) and freed once
//! scanning rests, and the CPU's copies of the digits are wiped, but a driver's own
//! copies can't be reached.

use std::{cell::RefCell, ffi::CStr, path::PathBuf};

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

/// A batch's pipelines and their push constants, the view key's digits. Wiped on drop.
struct Stages(Vec<(vk::Pipeline, [i32; 16])>);

impl Drop for Stages {
    fn drop(&mut self) {
        for (_, digits) in &mut self.0 {
            digits.zeroize();
        }
    }
}

const SPIRV: &[u8] = include_bytes!("../shaders/scan.spv");
/// A key's working state on the GPU: 300 words.
const STATE_BYTES: u64 = 300 * 4;
/// A key's result: the compressed point, then whether the key was off the curve.
const OUT_WORDS: usize = 9;
const CACHE_FILE: &str = "scan-gpu.cache";
/// A submit taking this long means something's wrong: the driver's watchdog should have acted.
const FENCE_TIMEOUT_NANOS: u64 = 60_000_000_000;
/// The push constants: sixteen digits and the number of keys.
const PUSH_BYTES: u32 = 68;

/// How the work is cut up.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    /// The most keys in one go. A key's state is 1,200 bytes on the GPU.
    pub batch: usize,
    /// Digits of the view key taken per dispatch.
    pub windows: usize,
    /// Dispatches per submit.
    pub submit_every: usize,
    /// Keys per workgroup.
    pub group: u32,
}

/// Tuned on a Mali-G715: 16 windows hold a key's state in the shader longest, and a dispatch to about 60 ms.
pub const TUNING: Tuning = Tuning { batch: 32_768, windows: 16, submit_every: 1, group: 64 };

/// A discrete GPU has no dispatch near its watchdog, so a batch is one submit. Smaller
/// batches would let the cores finish more while it works, but on an RX 6800 XT a batch
/// costs at least 3.7 ms however small: 8,192 keys were slower overall, and 16,384 no faster.
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

/// What the GPU worked out for a batch. Copied out whole first, as reading the
/// GPU's memory a word at a time is slow on a phone, and decoded by whichever
/// thread takes each point.
pub struct Points(Zeroizing<Vec<u32>>);

impl Points {
    #[cfg(test)]
    pub(crate) fn from_words(words: Vec<u32>) -> Points {
        Points(Zeroizing::new(words))
    }

    /// Key `i`'s point, compressed, or `None` if the key wasn't on the curve.
    /// Wipe it when done: it's the key's shared secret with the wallet.
    pub fn get(&self, i: usize) -> Option<[u8; 32]> {
        let word = &self.0[i * OUT_WORDS..(i + 1) * OUT_WORDS];
        (word[8] == 0).then(|| {
            let mut bytes = [0; 32];
            for (limb, out) in word[..8].iter().zip(bytes.as_chunks_mut::<4>().0) {
                *out = limb.to_le_bytes();
            }
            bytes
        })
    }

    pub fn len(&self) -> usize {
        self.0.len() / OUT_WORDS
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// What a batch runs in, kept for the next one. Cleared after each batch, as `run_on`
/// does, and freed by [`Gpu::release`], by a failure and with the GPU.
struct Arena {
    input: Buffer,
    state: Buffer,
    out: Buffer,
    set: vk::DescriptorSet,
    /// The most items and input bytes it takes.
    items: u32,
    input_bytes: usize,
    in_map: *mut u8,
    out_map: *mut u32,
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    descriptors: vk::DescriptorPool,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    cache: vk::PipelineCache,
    init: vk::Pipeline,
    window: vk::Pipeline,
    tail: Option<(usize, vk::Pipeline)>,
    last: vk::Pipeline,
    check: vk::Pipeline,
    tuning: Tuning,
    name: String,
    driver: u32,
    arena: RefCell<Option<Arena>>,
}

// Vulkan's handles may move between threads; they're only used under the one lock.
unsafe impl Send for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.release();
            for pipeline in [self.init, self.window, self.last, self.check] {
                self.device.destroy_pipeline(pipeline, None);
            }
            if let Some((_, tail)) = self.tail {
                self.device.destroy_pipeline(tail, None);
            }
            self.device.destroy_pipeline_cache(self.cache, None);
            self.device.destroy_shader_module(self.module, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.descriptors, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Gpu {
    /// Sets up the GPU, compiling the shader: up to about 20 s on a phone the
    /// first time, and little once `cache_dir` holds the result.
    ///
    /// `any_kind` also takes software renderers, which only the test tool wants:
    /// they'd be a slower CPU.
    pub fn start(cache_dir: Option<PathBuf>, tuning: Tuning, any_kind: bool) -> Result<Gpu, GpuError> {
        Self::start_tuned(cache_dir, Some(tuning), any_kind)
    }

    /// [`start`](Gpu::start), tuned for the GPU it finds: what [`crate::warm_up`] uses.
    pub fn start_for_device(cache_dir: Option<PathBuf>, any_kind: bool) -> Result<Gpu, GpuError> {
        Self::start_tuned(cache_dir, None, any_kind)
    }

    fn start_tuned(cache_dir: Option<PathBuf>, tuning: Option<Tuning>, any_kind: bool) -> Result<Gpu, GpuError> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| GpuError::new(GpuErrorKind::NoVulkan, format!("no Vulkan: {e}")))?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
            // Without a driver, the loader can't make an instance.
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(|e| GpuError::new(GpuErrorKind::NoVulkan, format!("instance: {e}")))?;
            Self::on_instance(&entry, instance, cache_dir, tuning, any_kind)
        }
    }

    /// Takes `instance`, destroying it on failure as well.
    unsafe fn on_instance(entry: &ash::Entry, instance: ash::Instance, cache_dir: Option<PathBuf>, tuning: Option<Tuning>, any_kind: bool) -> Result<Gpu, GpuError> {
        unsafe {
            let found = Self::choose(&instance, tuning.unwrap_or(TUNING), any_kind);
            let (physical, family, name) = match found {
                Ok(found) => found,
                Err(e) => {
                    instance.destroy_instance(None);
                    return Err(e);
                }
            };
            let properties = instance.get_physical_device_properties(physical);
            let driver = properties.driver_version;
            let tuning = tuning.unwrap_or(match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DISCRETE_TUNING,
                _ => TUNING,
            });
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let device = match instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues), None) {
                Ok(device) => device,
                Err(e) => {
                    instance.destroy_instance(None);
                    return Err(failed("device")(e));
                }
            };
            let queue = device.get_device_queue(family, 0);
            let memory = instance.get_physical_device_memory_properties(physical);

            // From here the Gpu owns it all; its drop frees what's been made, as
            // Vulkan's destroy calls ignore null handles.
            let mut gpu = Gpu {
                _entry: entry.clone(),
                instance,
                device,
                memory,
                queue,
                pool: vk::CommandPool::null(),
                command: vk::CommandBuffer::null(),
                fence: vk::Fence::null(),
                set_layout: vk::DescriptorSetLayout::null(),
                descriptors: vk::DescriptorPool::null(),
                layout: vk::PipelineLayout::null(),
                module: vk::ShaderModule::null(),
                cache: vk::PipelineCache::null(),
                init: vk::Pipeline::null(),
                window: vk::Pipeline::null(),
                tail: None,
                last: vk::Pipeline::null(),
                check: vk::Pipeline::null(),
                tuning,
                name,
                driver,
                arena: RefCell::new(None),
            };
            gpu.build(family, cache_dir)?;
            Ok(gpu)
        }
    }

    /// The GPU to use, its compute queue family and its name.
    unsafe fn choose(instance: &ash::Instance, tuning: Tuning, any_kind: bool) -> Result<(vk::PhysicalDevice, u32, String), GpuError> {
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
            {
                return Err(GpuError::new(GpuErrorKind::NoGpu, format!("{name} is too small for this work")));
            }
            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| GpuError::new(GpuErrorKind::NoGpu, format!("{name} has no compute queue")))? as u32;
            Ok((physical, family, name))
        }
    }

    unsafe fn build(&mut self, family: u32, cache_dir: Option<PathBuf>) -> Result<(), GpuError> {
        unsafe {
            let device = &self.device;
            self.pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .map_err(failed("command pool"))?;
            self.command = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .map_err(failed("command buffer"))?[0];
            self.fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(failed("fence"))?;

            let bindings: Vec<_> = (0..3)
                .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            self.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(failed("set layout"))?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }];
            self.descriptors = device
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET).pool_sizes(&sizes), None)
                .map_err(failed("descriptor pool"))?;
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
                // Compiled shader code, with no key material: digits are push constants.
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

    /// A buffer in the first memory type with all of one of `wanted`'s flags.
    unsafe fn buffer(&self, size: u64, wanted: &[vk::MemoryPropertyFlags]) -> Result<(Buffer, vk::MemoryPropertyFlags), GpuError> {
        unsafe {
            let info = vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(failed("buffer"))?;
            let needs = self.device.get_buffer_memory_requirements(buffer);
            for flags in wanted {
                for i in 0..self.memory.memory_type_count {
                    let kind = self.memory.memory_types[i as usize];
                    if needs.memory_type_bits & (1 << i) != 0 && kind.property_flags.contains(*flags) {
                        let alloc = vk::MemoryAllocateInfo::default().allocation_size(needs.size).memory_type_index(i);
                        if let Ok(memory) = self.device.allocate_memory(&alloc, None) {
                            self.device.bind_buffer_memory(buffer, memory, 0).map_err(failed("binding memory"))?;
                            return Ok((Buffer { buffer, memory }, kind.property_flags));
                        }
                    }
                }
            }
            self.device.destroy_buffer(buffer, None);
            Err(GpuError::failed(format!("no memory for {size} bytes")))
        }
    }

    unsafe fn free(&self, buffer: &Buffer) {
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    /// Gives back the memory kept between batches, for when scanning rests. The next
    /// batch makes it again.
    pub fn release(&self) {
        if let Some(arena) = self.arena.borrow_mut().take() {
            unsafe { self.free_arena(arena) };
        }
    }

    unsafe fn free_arena(&self, arena: Arena) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // A failed batch may have left points multiplied by the key in the results:
            // they're wiped before the memory goes back. (The input is the
            // transactions' public keys only.) Freeing unmaps.
            if !arena.out_map.is_null() {
                let words = std::slice::from_raw_parts_mut(arena.out_map, arena.items as usize * OUT_WORDS);
                words.zeroize();
            }
            let _ = self.device.free_descriptor_sets(self.descriptors, &[arena.set]);
            self.free(&arena.input);
            self.free(&arena.state);
            self.free(&arena.out);
        }
    }

    /// Room for at least `items` items with `input_bytes` of input. Sizes go up in
    /// powers of two, so a run of growing batches makes it only a few times.
    unsafe fn new_arena(&self, items: u32, input_bytes: usize) -> Result<Arena, GpuError> {
        unsafe {
            let device = &self.device;
            let items = items.next_power_of_two();
            let input_bytes = input_bytes.next_power_of_two();
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let cached = visible | vk::MemoryPropertyFlags::HOST_CACHED;
            let (input, _) = self.buffer(input_bytes as u64, &[visible])?;
            let state = self.buffer(u64::from(items) * STATE_BYTES, &[local, vk::MemoryPropertyFlags::empty()]);
            let out = self.buffer(u64::from(items) * OUT_WORDS as u64 * 4, &[cached, visible]);
            let (state, out) = match (state, out) {
                (Ok((state, _)), Ok((out, _))) => (state, out),
                (state, out) => {
                    self.free(&input);
                    let mut problem = None;
                    for made in [state, out] {
                        match made {
                            Ok((buffer, _)) => self.free(&buffer),
                            Err(e) => problem = Some(e),
                        }
                    }
                    return Err(problem.expect("one of them failed"));
                }
            };
            let layouts = [self.set_layout];
            let made = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.descriptors).set_layouts(&layouts));
            let set = match made {
                Ok(sets) => sets[0],
                Err(e) => {
                    for buffer in [&input, &state, &out] {
                        self.free(buffer);
                    }
                    return Err(GpuError::failed(format!("descriptor set: {e}")));
                }
            };
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [&input, &state, &out].iter().map(|b| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }]).collect();
            let writes: Vec<_> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            device.update_descriptor_sets(&writes, &[]);
            // Mapped for as long as they're kept: the rest is the same memory each batch.
            let maps = device.map_memory(input.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).and_then(|in_map| Ok((in_map, device.map_memory(out.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?)));
            let arena = Arena { input, state, out, set, items, input_bytes, in_map: std::ptr::null_mut(), out_map: std::ptr::null_mut() };
            match maps {
                Ok((in_map, out_map)) => Ok(Arena { in_map: in_map as *mut u8, out_map: out_map as *mut u32, ..arena }),
                Err(e) => {
                    self.free_arena(arena);
                    Err(failed("mapping")(e))
                }
            }
        }
    }

    unsafe fn submit(&self, record: impl Fn(vk::CommandBuffer)) -> Result<(), GpuError> {
        unsafe {
            let device = &self.device;
            device.begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(failed("recording"))?;
            record(self.command);
            device.end_command_buffer(self.command).map_err(failed("recording"))?;
            let commands = [self.command];
            let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
            device.queue_submit(self.queue, &submit, self.fence).map_err(failed("submit"))?;
            device.wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NANOS).map_err(failed("waiting"))?;
            device.reset_fences(&[self.fence]).map_err(failed("fence"))?;
            // Takes the digits recorded in it off the driver's hands.
            device.reset_command_buffer(self.command, vk::CommandBufferResetFlags::RELEASE_RESOURCES).map_err(failed("reset"))?;
            Ok(())
        }
    }

    /// Runs `stages` (a pipeline and its push constants) over `count` items whose
    /// input is in `input`, reading back each item's words through `read`.
    unsafe fn run(&self, input: &[u8], count: u32, stages: &[(vk::Pipeline, [i32; 16])], read: &mut dyn FnMut(&[u32])) -> Result<(), GpuError> {
        unsafe {
            let mut kept = self.arena.borrow_mut();
            if kept.as_ref().is_none_or(|a| a.items < count || a.input_bytes < input.len()) {
                if let Some(small) = kept.take() {
                    self.free_arena(small);
                }
                *kept = Some(self.new_arena(count, input.len())?);
            }
            let arena = kept.as_ref().expect("just made");
            let result = self.run_on(arena, input, count, stages, read);
            if result.is_err() {
                // What it held may not have been cleared, and the GPU is being retired.
                if let Some(arena) = kept.take() {
                    self.free_arena(arena);
                }
            }
            result
        }
    }

    unsafe fn run_on(&self, arena: &Arena, input: &[u8], count: u32, stages: &[(vk::Pipeline, [i32; 16])], read: &mut dyn FnMut(&[u32])) -> Result<(), GpuError> {
        unsafe {
            let device = &self.device;
            std::ptr::copy_nonoverlapping(input.as_ptr(), arena.in_map, input.len());

            let groups = count.div_ceil(self.tuning.group);
            let sets = [arena.set];
            let used = u64::from(count) * STATE_BYTES;
            let every = self.tuning.submit_every.max(1);
            let parts = stages.len().div_ceil(every);
            for (at, part) in stages.chunks(every).enumerate() {
                self.submit(|command| {
                    device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &sets, &[]);
                    for (pipeline, digits) in part {
                        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, *pipeline);
                        let mut bytes = [0u8; PUSH_BYTES as usize];
                        for (i, d) in digits.iter().enumerate() {
                            bytes[i * 4..i * 4 + 4].copy_from_slice(&d.to_ne_bytes());
                        }
                        bytes[64..68].copy_from_slice(&count.to_ne_bytes());
                        device.cmd_push_constants(command, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                        // The view key's digits: the driver has copied them now.
                        bytes.zeroize();
                        device.cmd_dispatch(command, groups, 1, 1);
                        let barrier = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_READ)];
                        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                    }
                    // The working state is points multiplied by the key. It's cleared in the
                    // last submit, for no wait of its own, and the next batch starts after that.
                    if at + 1 == parts {
                        device.cmd_fill_buffer(command, arena.state.buffer, 0, used, 0);
                        let barrier = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                    }
                })?;
            }
            let words = std::slice::from_raw_parts_mut(arena.out_map, count as usize * OUT_WORDS);
            read(words);

            // The results are points multiplied by the key too.
            words.fill(0);
            Ok(())
        }
    }

    /// 8aR for each of `keys`, at most [`Tuning::batch`] of them, canonical.
    pub fn derive(&self, digits: &Digits, keys: &[[u8; 32]]) -> Result<Points, GpuError> {
        if keys.is_empty() {
            return Ok(Points(Zeroizing::new(vec![])));
        }
        let windows = self.tuning.windows;
        // Every copy of the view key's digits made here is wiped when it goes, on
        // an error too: `Stages` on drop, the rest as `Zeroizing`. Sized up front so
        // no copy is left behind by a reallocation.
        let mut stages = Stages(Vec::with_capacity(66));
        let mut first = Zeroizing::new([0i32; 16]);
        first[0] = i32::from(digits.get(63));
        stages.0.push((self.init, *first));
        let mut order = Zeroizing::new(Vec::with_capacity(63));
        order.extend((0..63).rev().map(|i| i32::from(digits.get(i))));
        for part in order.chunks(windows) {
            let mut pc = Zeroizing::new([0i32; 16]);
            pc[..part.len()].copy_from_slice(part);
            let pipeline = match self.tail {
                Some((short, tail)) if part.len() == short => tail,
                _ => self.window,
            };
            stages.0.push((pipeline, *pc));
        }
        stages.0.push((self.last, [0; 16]));

        let mut points = Zeroizing::new(vec![]);
        unsafe {
            self.run(keys.as_flattened(), keys.len() as u32, &stages.0, &mut |words| points.extend_from_slice(words))?;
        }
        Ok(Points(points))
    }

    /// One field operation on each pair: for the tests. 0 adds, 1 subtracts,
    /// 2 multiplies, 3 squares, 4 negates and 5 only reduces.
    pub fn check_field(&self, op: i32, pairs: &[([u32; 8], [u32; 8])]) -> Result<Vec<[u32; 8]>, GpuError> {
        let mut input = Vec::with_capacity(pairs.len() * 64);
        for (a, b) in pairs {
            for limb in a.iter().chain(b) {
                input.extend_from_slice(&limb.to_le_bytes());
            }
        }
        let mut digits = [0i32; 16];
        digits[0] = op;
        let mut out = Vec::with_capacity(pairs.len());
        unsafe {
            self.run(&input, pairs.len() as u32, &[(self.check, digits)], &mut |words| {
                for word in words.as_chunks::<OUT_WORDS>().0 {
                    let mut limbs = [0; 8];
                    limbs.copy_from_slice(&word[..8]);
                    out.push(limbs);
                }
            })?;
        }
        Ok(out)
    }
}
