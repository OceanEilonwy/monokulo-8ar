//! 8aR on the GPU for every transaction key in a scan-bench fixture:
//!
//!   scan-vulkan <file> [--batch N] [--steps K] [--submit-every M] [--group W] [--threads T] [--runs R]
//!
//! View tags and everything after them stay on the CPU, as in scan-bench. A run
//! that doesn't end in digest e0c4c620 for the 2000-block fixture is wrong.

use std::{
    collections::HashMap,
    ffi::CStr,
    io::Read,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use ash::vk;
use curve25519_dalek::{EdwardsPoint, Scalar, constants::ED25519_BASEPOINT_TABLE, edwards::CompressedEdwardsY};
use sha3::{Digest, Keccak256};

const SPIRV: &[u8] = include_bytes!("../shaders/scan.spv");
/// One key's state in the shader, in bytes: 300 uints.
const STATE_BYTES: u64 = 300 * 4;
const OUT_WORDS: usize = 9;

struct Job {
    r: [u8; 32],
    outs: Vec<Out>,
}

struct Out {
    index: u32,
    tag: u8,
    key: [u8; 32],
}

struct Work {
    blocks: u64,
    jobs: Vec<Job>,
    outputs: usize,
}

fn read_work(path: &str) -> Work {
    let mut f = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut u64s = [0u8; 8];
    f.read_exact(&mut u64s).unwrap();
    let blocks = u64::from_le_bytes(u64s);
    f.read_exact(&mut u64s).unwrap();
    let count = u64::from_le_bytes(u64s);
    let mut jobs = Vec::with_capacity(count as usize);
    let mut outputs = 0;
    for _ in 0..count {
        let mut r = [0; 32];
        f.read_exact(&mut r).unwrap();
        let mut n = [0; 4];
        f.read_exact(&mut n).unwrap();
        let n = u32::from_le_bytes(n);
        let mut outs = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let mut index = [0; 4];
            f.read_exact(&mut index).unwrap();
            let mut tag = [0; 1];
            f.read_exact(&mut tag).unwrap();
            let mut key = [0; 32];
            f.read_exact(&mut key).unwrap();
            outs.push(Out { index: u32::from_le_bytes(index), tag: tag[0], key });
        }
        outputs += outs.len();
        jobs.push(Job { r, outs });
    }
    Work { blocks, jobs, outputs }
}

/// scan-bench's view key: nobody's.
fn view_key() -> [u8; 32] {
    Scalar::from_bytes_mod_order(Keccak256::digest(b"scan-bench view key").into()).to_bytes()
}

/// The scalar's signed base-16 digits, least significant first.
fn digits(scalar: &[u8; 32]) -> [i8; 64] {
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
    d
}

fn varint(mut n: u32, out: &mut Vec<u8>) {
    while n >= 0x80 {
        out.push(n as u8 | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

#[derive(Default)]
struct Tally {
    digest: [u8; 32],
    matched: u64,
    found: u64,
    invalid: u64,
}

impl Tally {
    fn merge(&mut self, o: &Tally) {
        for i in 0..32 {
            self.digest[i] ^= o.digest[i];
        }
        self.matched += o.matched;
        self.found += o.found;
        self.invalid += o.invalid;
    }
}

/// scan-bench's after_derivation.
fn after_derivation(derivation: &[u8; 32], job: &Job, spend: &HashMap<[u8; 32], u32>, tally: &mut Tally) {
    for i in 0..32 {
        tally.digest[i] ^= derivation[i];
    }
    let mut message = Vec::with_capacity(48);
    for out in &job.outs {
        message.clear();
        message.extend_from_slice(b"view_tag");
        message.extend_from_slice(derivation);
        varint(out.index, &mut message);
        if Keccak256::digest(&message)[0] != out.tag {
            continue;
        }
        tally.matched += 1;
        let Some(key) = CompressedEdwardsY(out.key).decompress() else { continue };
        message.clear();
        message.extend_from_slice(derivation);
        varint(out.index, &mut message);
        let shared = Scalar::from_bytes_mod_order(Keccak256::digest(&message).into());
        let candidate: EdwardsPoint = key - (&shared * ED25519_BASEPOINT_TABLE);
        if spend.contains_key(candidate.compress().as_bytes()) {
            tally.found += 1;
        }
    }
}

struct Options {
    batch: usize,
    steps: usize,
    submit_every: usize,
    group: u32,
    threads: usize,
    runs: usize,
}

struct Gpu {
    device: ash::Device,
    queue: vk::Queue,
    memory: vk::PhysicalDeviceMemoryProperties,
    pool: vk::CommandPool,
}

impl Gpu {
    /// A buffer in the first memory type holding every flag of one of `wanted`.
    fn buffer(&self, size: u64, wanted: &[vk::MemoryPropertyFlags]) -> (vk::Buffer, vk::DeviceMemory, vk::MemoryPropertyFlags) {
        unsafe {
            let info = vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).unwrap();
            let needs = self.device.get_buffer_memory_requirements(buffer);
            for flags in wanted {
                for i in 0..self.memory.memory_type_count {
                    let kind = self.memory.memory_types[i as usize];
                    if needs.memory_type_bits & (1 << i) != 0 && kind.property_flags.contains(*flags) {
                        let alloc = vk::MemoryAllocateInfo::default().allocation_size(needs.size).memory_type_index(i);
                        if let Ok(memory) = self.device.allocate_memory(&alloc, None) {
                            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
                            return (buffer, memory, kind.property_flags);
                        }
                    }
                }
            }
            panic!("no memory for a buffer of {size} bytes");
        }
    }
}

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter().position(|a| a == name).map_or(default, |i| args[i + 1].parse().unwrap())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = &args[0];
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let options = Options {
        batch: arg(&args, "--batch", 8192),
        steps: arg(&args, "--steps", 1),
        submit_every: arg(&args, "--submit-every", 1000),
        group: arg(&args, "--group", 64) as u32,
        threads: arg(&args, "--threads", cores.saturating_sub(1).clamp(1, 8)),
        runs: arg(&args, "--runs", 1),
    };
    assert!(options.steps >= 1 && options.steps <= 16);
    let work = read_work(path);
    println!(
        "{} blocks, {} keys, {} outputs; batch {}, {} windows per dispatch, a submit every {} stages, group {}, {} CPU threads",
        work.blocks, work.jobs.len(), work.outputs, options.batch, options.steps, options.submit_every, options.group, options.threads
    );
    if args.iter().any(|a| a == "--cpu") {
        return cpu(&work, &options);
    }
    unsafe { run(&work, &options) }
}

/// The same work on the CPU's threads with curve25519-dalek, to compare power.
fn cpu(work: &Work, o: &Options) {
    let view = Scalar::from_bytes_mod_order(view_key());
    let spend: HashMap<[u8; 32], u32> = (0..200u32).map(|i| ((&Scalar::from(i as u64 + 1) * ED25519_BASEPOINT_TABLE).compress().to_bytes(), i)).collect();
    for run in 1..=o.runs {
        let next = AtomicUsize::new(0);
        let started = Instant::now();
        let tallies: Vec<Tally> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..o.threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut tally = Tally::default();
                        loop {
                            let at = next.fetch_add(16, Ordering::Relaxed);
                            if at >= work.jobs.len() {
                                return tally;
                            }
                            for job in &work.jobs[at..(at + 16).min(work.jobs.len())] {
                                let Some(r) = CompressedEdwardsY(job.r).decompress() else { continue };
                                after_derivation(&(view * r).mul_by_cofactor().compress().to_bytes(), job, &spend, &mut tally);
                            }
                        }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let seconds = started.elapsed().as_secs_f64();
        let mut all = Tally::default();
        for t in &tallies {
            all.merge(t);
        }
        println!("run {run}: CPU {seconds:.3} s {:.0} keys/s {:.0} blocks/s digest {}", work.jobs.len() as f64 / seconds, work.blocks as f64 / seconds, hex::encode(&all.digest[..4]));
    }
}

unsafe fn run(work: &Work, o: &Options) {
    unsafe {
        let entry = ash::Entry::load().expect("no Vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).unwrap();
        let mut devices = instance.enumerate_physical_devices().unwrap();
        devices.sort_by_key(|d| match instance.get_physical_device_properties(*d).device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        });
        let physical = devices[0];
        let properties = instance.get_physical_device_properties(physical);
        let features = instance.get_physical_device_features(physical);
        println!(
            "device: {} (driver {:#x}), shaderInt64 {}, max group {}, max storage range {} MB, timestamp period {} ns",
            CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy(),
            properties.driver_version,
            features.shader_int64,
            properties.limits.max_compute_work_group_invocations,
            properties.limits.max_storage_buffer_range / (1 << 20),
            properties.limits.timestamp_period,
        );
        let family = instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let device = instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues), None).unwrap();
        let queue = device.get_device_queue(family, 0);
        let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
        let gpu = Gpu { device, queue, memory: instance.get_physical_device_memory_properties(physical), pool };
        let device = &gpu.device;

        let batch = o.batch.min(work.jobs.len());
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached = visible | vk::MemoryPropertyFlags::HOST_CACHED;
        let (in_buffer, in_memory, _) = gpu.buffer(batch as u64 * 32, &[visible]);
        let (state_buffer, _, state_flags) = gpu.buffer(batch as u64 * STATE_BYTES, &[local, vk::MemoryPropertyFlags::empty()]);
        let (out_buffer, out_memory, out_flags) = gpu.buffer((batch * OUT_WORDS * 4) as u64, &[cached, visible]);
        println!("state memory {state_flags:?}, output memory {out_flags:?}, state {} MB", batch as u64 * STATE_BYTES / (1 << 20));
        let in_map = device.map_memory(in_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
        let out_map = device.map_memory(out_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *const u32;

        let bindings: Vec<_> = (0..3)
            .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }];
        let descriptors = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).unwrap();
        let layouts = [set_layout];
        let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptors).set_layouts(&layouts)).unwrap()[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [in_buffer, state_buffer, out_buffer].iter().map(|b| [vk::DescriptorBufferInfo { buffer: *b, offset: 0, range: vk::WHOLE_SIZE }]).collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info))
            .collect();
        device.update_descriptor_sets(&writes, &[]);

        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 64 }];
        let pipeline_layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).unwrap();
        let code: Vec<u32> = SPIRV.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).unwrap();
        let compile = Instant::now();
        let pipelines: Vec<vk::Pipeline> = (0..3u32)
            .map(|stage| {
                let values = [o.group, stage, o.steps as u32];
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let entries: Vec<_> = (0..3).map(|i| vk::SpecializationMapEntry { constant_id: i, offset: i * 4, size: 4 }).collect();
                let spec = vk::SpecializationInfo::default().map_entries(&entries).data(&bytes);
                let shader = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main").specialization_info(&spec);
                let info = vk::ComputePipelineCreateInfo::default().stage(shader).layout(pipeline_layout);
                let made = Instant::now();
                let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|e| e.1).unwrap()[0];
                println!("pipeline {stage} compiled in {:.2} s", made.elapsed().as_secs_f64());
                pipeline
            })
            .collect();
        println!("pipelines compiled in {:.2} s", compile.elapsed().as_secs_f64());

        let command = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(gpu.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();

        // Stages, most significant digit first: INIT takes the top digit.
        let d = digits(&view_key());
        let mut stages: Vec<(usize, [i32; 16])> = vec![];
        let mut first = [0i32; 16];
        first[0] = d[63] as i32;
        stages.push((0, first));
        let windows: Vec<i32> = (0..63).rev().map(|i| d[i] as i32).collect();
        for part in windows.chunks(o.steps) {
            let mut pc = [0i32; 16];
            pc[..part.len()].copy_from_slice(part);
            // The pipeline does exactly `steps`, so a short last chunk runs alone.
            stages.push((if part.len() == o.steps { 1 } else { 3 + part.len() }, pc));
        }
        stages.push((2, [0; 16]));
        // A short tail needs a pipeline of its own length.
        let mut tail_pipeline = vk::Pipeline::null();
        if let Some(short) = stages.iter().find(|s| s.0 > 3).map(|s| s.0 - 3) {
            let values = [o.group, 1, short as u32];
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let entries: Vec<_> = (0..3).map(|i| vk::SpecializationMapEntry { constant_id: i, offset: i * 4, size: 4 }).collect();
            let spec = vk::SpecializationInfo::default().map_entries(&entries).data(&bytes);
            let shader = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main").specialization_info(&spec);
            let info = vk::ComputePipelineCreateInfo::default().stage(shader).layout(pipeline_layout);
            tail_pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|e| e.1).unwrap()[0];
        }

        let spend: HashMap<[u8; 32], u32> = (0..200u32).map(|i| ((&Scalar::from(i as u64 + 1) * ED25519_BASEPOINT_TABLE).compress().to_bytes(), i)).collect();

        for run in 1..=o.runs {
            let whole = Instant::now();
            let (mut gpu_seconds, mut cpu_seconds, mut longest_submit) = (0f64, 0f64, 0f64);
            let mut all = Tally::default();
            for jobs in work.jobs.chunks(batch) {
                for (i, job) in jobs.iter().enumerate() {
                    std::ptr::copy_nonoverlapping(job.r.as_ptr(), in_map.add(i * 32), 32);
                }
                let groups = (jobs.len() as u32).div_ceil(o.group);
                let started = Instant::now();
                for part in stages.chunks(o.submit_every) {
                    device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                    device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[]);
                    for (stage, pc) in part {
                        let pipeline = if *stage > 3 { tail_pipeline } else { pipelines[*stage] };
                        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                        let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_ne_bytes()).collect();
                        device.cmd_push_constants(command, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                        device.cmd_dispatch(command, groups, 1, 1);
                        let barrier = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::HOST_READ)];
                        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                    }
                    device.end_command_buffer(command).unwrap();
                    let commands = [command];
                    let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
                    let submitted = Instant::now();
                    device.queue_submit(gpu.queue, &submit, fence).unwrap();
                    device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                    device.reset_fences(&[fence]).unwrap();
                    longest_submit = longest_submit.max(submitted.elapsed().as_secs_f64());
                }
                gpu_seconds += started.elapsed().as_secs_f64();

                // Readback and the CPU's part, on its threads.
                let started = Instant::now();
                let results = std::slice::from_raw_parts(out_map, jobs.len() * OUT_WORDS).to_vec();
                let next = AtomicUsize::new(0);
                let tallies: Vec<Tally> = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..o.threads)
                        .map(|_| {
                            scope.spawn(|| {
                                let mut tally = Tally::default();
                                loop {
                                    let at = next.fetch_add(64, Ordering::Relaxed);
                                    if at >= jobs.len() {
                                        return tally;
                                    }
                                    for i in at..(at + 64).min(jobs.len()) {
                                        let words = &results[i * OUT_WORDS..(i + 1) * OUT_WORDS];
                                        if words[8] != 0 {
                                            tally.invalid += 1;
                                            continue;
                                        }
                                        let mut derivation = [0u8; 32];
                                        for (w, word) in words[..8].iter().enumerate() {
                                            derivation[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
                                        }
                                        after_derivation(&derivation, &jobs[i], &spend, &mut tally);
                                    }
                                }
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for t in &tallies {
                    all.merge(t);
                }
                cpu_seconds += started.elapsed().as_secs_f64();
            }
            let seconds = whole.elapsed().as_secs_f64();
            println!(
                "run {run}: GPU {gpu_seconds:.3} s {:.0} keys/s; CPU part {cpu_seconds:.3} s; whole {seconds:.3} s {:.0} blocks/s; longest submit {:.3} s; digest {} tags {} found {} invalid {}",
                work.jobs.len() as f64 / gpu_seconds,
                work.blocks as f64 / seconds,
                longest_submit,
                hex::encode(&all.digest[..4]),
                all.matched,
                all.found,
                all.invalid,
            );
        }
    }
}
