//! Times the engines on a fixture written by prototypes/scan-bench:
//!
//!   scan_speed <file> [--cpu N] [--batch N] [--windows N] [--every N] [--group N] [--runs N] [--any]
//!
//! Each run does the arithmetic, the compression and every view tag, and prints
//! the XOR of all derivations: e0c4c620... for the 2000-block fixture.
//!
//!   scan_speed <file> --tip 1,10,100 [--seconds N] [--rounds N] [--period-ms N] [--cache dir]
//!
//! Compares the energy of small catch-ups on Android: for each size of catch-up,
//! rounds of idle, CPU and GPU with the GPU left open, as in the app. One batch
//! each period, long enough for the GPU to power down between them. The battery's
//! current is sampled once a second: what matters is the difference from idle in
//! the same round, as a charger may be supplying some of it.

use std::{io::Read, sync::atomic::{AtomicUsize, Ordering}, time::Instant};

use monokulo_8ar::{fe, gpu::{Gpu, TUNING, Tuning}};
use curve25519_dalek::Scalar;
use sha3::{Digest, Keccak256};

struct Job {
    r: [u8; 32],
    outs: Vec<(u32, u8)>,
}

fn read_work(path: &str) -> (u64, Vec<Job>) {
    let mut f = std::io::BufReader::new(std::fs::File::open(path).expect("the fixture"));
    let mut bytes = [0u8; 8];
    f.read_exact(&mut bytes).unwrap();
    let blocks = u64::from_le_bytes(bytes);
    f.read_exact(&mut bytes).unwrap();
    let count = u64::from_le_bytes(bytes);
    let jobs = (0..count)
        .map(|_| {
            let mut r = [0; 32];
            f.read_exact(&mut r).unwrap();
            let mut n = [0; 4];
            f.read_exact(&mut n).unwrap();
            let outs = (0..u32::from_le_bytes(n))
                .map(|_| {
                    let mut out = [0; 37];
                    f.read_exact(&mut out).unwrap();
                    (u32::from_le_bytes(out[..4].try_into().unwrap()), out[4])
                })
                .collect();
            Job { r, outs }
        })
        .collect();
    (blocks, jobs)
}

fn tags(derivation: &[u8; 32], job: &Job) -> u64 {
    let mut hits = 0;
    let mut message = Vec::with_capacity(48);
    for (index, tag) in &job.outs {
        message.clear();
        message.extend_from_slice(b"view_tag");
        message.extend_from_slice(derivation);
        let mut n = *index;
        while n >= 0x80 {
            message.push(n as u8 | 0x80);
            n >>= 7;
        }
        message.push(n as u8);
        if Keccak256::digest(&message)[0] == *tag {
            hits += 1;
        }
    }
    hits
}

fn in_parallel(count: usize, threads: usize, chunk: usize, each: impl Fn(usize, usize) -> ([u8; 32], u64) + Sync) -> ([u8; 32], u64) {
    let next = AtomicUsize::new(0);
    let parts: Vec<([u8; 32], u64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut all = ([0u8; 32], 0);
                    loop {
                        let at = next.fetch_add(chunk, Ordering::Relaxed);
                        if at >= count {
                            return all;
                        }
                        let (digest, hits) = each(at, (at + chunk).min(count));
                        all.0.iter_mut().zip(digest).for_each(|(a, d)| *a ^= d);
                        all.1 += hits;
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut all = ([0u8; 32], 0);
    for part in parts {
        all.0.iter_mut().zip(part.0).for_each(|(a, d)| *a ^= d);
        all.1 += part.1;
    }
    all
}

fn finish(points: &[(fe::Fe, fe::Fe, fe::Fe)], jobs: &[Job]) -> ([u8; 32], u64) {
    let mut compressed = Vec::with_capacity(points.len());
    fe::compress_batch(points, &mut compressed);
    let mut digest = [0u8; 32];
    let mut hits = 0;
    for (derivation, job) in compressed.iter().zip(jobs) {
        digest.iter_mut().zip(derivation).for_each(|(a, d)| *a ^= d);
        hits += tags(derivation, job);
    }
    (digest, hits)
}

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter().position(|a| a == name).map_or(default, |i| args[i + 1].parse().expect("a number"))
}

fn cpu_batch(jobs: &[Job], digits: &fe::Digits, threads: usize) {
    in_parallel(jobs.len(), threads, 64, |at, end| {
        let part = &jobs[at..end];
        let identity = fe::Ext::decompress(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let points: Vec<_> = part.iter().map(|job| fe::mul8(digits, &fe::Ext::decompress(&job.r).unwrap_or(identity))).collect();
        finish(&points, part)
    });
}

fn gpu_batch(jobs: &[Job], digits: &fe::Digits, threads: usize, gpu: &Gpu) {
    let keys: Vec<[u8; 32]> = jobs.iter().map(|j| j.r).collect();
    let points = gpu.derive(digits, &keys).expect("the GPU ran");
    in_parallel(jobs.len(), threads, 64, |at, end| {
        let points: Vec<_> = (at..end).map(|i| points.get(i).expect("on the curve")).collect();
        finish(&points, &jobs[at..end])
    });
}

fn read_sys(name: &str) -> f64 {
    std::fs::read_to_string(format!("/sys/class/power_supply/battery/{name}")).expect("the battery").trim().parse().expect("a number")
}

fn tip(args: &[String], blocks: u64, jobs: &[Job], digits: &fe::Digits, sizes: &str) {
    let seconds = arg(args, "--seconds", 40) as u64;
    let rounds = arg(args, "--rounds", 4);
    let period = std::time::Duration::from_millis(arg(args, "--period-ms", 2300) as u64);
    let gpu = Gpu::start(args.iter().position(|a| a == "--cache").map(|i| args[i + 1].clone().into()), TUNING, false).expect("a GPU");
    let per_block = jobs.len() as f64 / blocks as f64;
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(1).clamp(1, 8);
    println!("{}, {per_block:.0} keys a block, one batch each {period:?}", gpu.name());
    println!("blocks keys threads | CPU ms mJ +- | GPU ms mJ +- | batches short of the plan");

    // Which arm is running: the sampler files each reading under it.
    let arm = std::sync::atomic::AtomicUsize::new(usize::MAX);
    let samples = std::sync::Mutex::new(vec![]);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                // Current is positive while charging: a load makes it smaller.
                let watts = read_sys("current_now") / 1e6 * (read_sys("voltage_now") / 1e6);
                samples.lock().unwrap().push((arm.load(Ordering::Relaxed), watts));
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
        for size in sizes.split(',') {
            let size: usize = size.parse().expect("a number of blocks");
            let keys = ((size as f64 * per_block) as usize).clamp(1, jobs.len());
            // As `scan.rs` does: one thread until there are enough blocks for more.
            let threads = if size < 8 { 1 } else { cores };
            let mut net = [vec![], vec![]];
            let mut busy = [0.0f64; 2];
            let mut batches = [0usize; 2];
            let mut short = 0;
            let mut at = 0;
            for round in 0..rounds {
                samples.lock().unwrap().clear();
                let order = if round % 2 == 0 { [2, 0, 1] } else { [2, 1, 0] };
                for which in order {
                    // A moment for the last arm's tail to pass.
                    arm.store(usize::MAX, Ordering::Relaxed);
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    arm.store(which, Ordering::Relaxed);
                    let started = Instant::now();
                    let (mut next, mut count) = (Instant::now(), 0);
                    while started.elapsed().as_secs() < seconds {
                        if which < 2 {
                            let part = &jobs[at..(at + keys).min(jobs.len())];
                            at = if at + 2 * keys > jobs.len() { 0 } else { at + keys };
                            let t = Instant::now();
                            if which == 0 { cpu_batch(part, digits, threads) } else { gpu_batch(part, digits, threads, &gpu) }
                            busy[which] += t.elapsed().as_secs_f64();
                            batches[which] += 1;
                        }
                        count += 1;
                        next += period;
                        std::thread::sleep(next.saturating_duration_since(Instant::now()));
                    }
                    short += (seconds as f64 / period.as_secs_f64()) as usize - count.min((seconds as f64 / period.as_secs_f64()) as usize);
                }
                let mean = |of: usize| {
                    let s = samples.lock().unwrap();
                    let v: Vec<f64> = s.iter().filter(|(a, _)| *a == of).map(|(_, w)| *w).collect();
                    v.iter().sum::<f64>() / v.len() as f64
                };
                let idle = mean(2);
                for (which, net) in net.iter_mut().enumerate() {
                    net.push((idle - mean(which)) * period.as_secs_f64() * 1000.0);
                }
            }
            let line = |which: usize| {
                let n = net[which].len() as f64;
                let m = net[which].iter().sum::<f64>() / n;
                let se = (net[which].iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0).max(1.0) / n).sqrt();
                format!("{:.1} ms {m:.0} mJ +-{se:.0}", busy[which] / batches[which].max(1) as f64 * 1000.0)
            };
            println!("{size} {keys} {threads} | {} | {} | {short}", line(0), line(1));
        }
        done.store(true, Ordering::Relaxed);
    });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (blocks, jobs) = read_work(&args[0]);
    let view = Scalar::from_bytes_mod_order(Keccak256::digest(b"scan-bench view key").into()).to_bytes();
    let digits = fe::Digits::new(&view);
    if let Some(i) = args.iter().position(|a| a == "--tip") {
        return tip(&args, blocks, &jobs, &digits, &args[i + 1]);
    }
    let runs = arg(&args, "--runs", 1);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(1).clamp(1, 8);
    let threads = arg(&args, "--threads", cores);
    let cpu = arg(&args, "--cpu", 0);
    let show = |run: usize, seconds: f64, gpu: f64, (digest, hits): ([u8; 32], u64)| {
        let hex: String = digest[..4].iter().map(|b| format!("{b:02x}")).collect();
        println!("run {run}: {seconds:.3} s, {:.0} blocks/s, {:.0} keys/s, GPU part {gpu:.3} s, digest {hex}, tags {hits}", blocks as f64 / seconds, jobs.len() as f64 / seconds);
        assert_eq!(hex, "e0c4c620", "the wrong answer");
    };
    if cpu > 0 {
        for run in 1..=runs {
            let started = Instant::now();
            let result = in_parallel(jobs.len(), cpu, 64, |at, end| {
                let part = &jobs[at..end];
                let identity = fe::Ext::decompress(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                let points: Vec<_> = part.iter().map(|job| fe::mul8(&digits, &fe::Ext::decompress(&job.r).unwrap_or(identity))).collect();
                finish(&points, part)
            });
            show(run, started.elapsed().as_secs_f64(), 0.0, result);
        }
        return;
    }
    let tuning = Tuning {
        batch: arg(&args, "--batch", TUNING.batch),
        windows: arg(&args, "--windows", TUNING.windows),
        submit_every: arg(&args, "--every", TUNING.submit_every),
        group: arg(&args, "--group", TUNING.group as usize) as u32,
    };
    let started = Instant::now();
    let gpu = Gpu::start(args.iter().position(|a| a == "--cache").map(|i| args[i + 1].clone().into()), tuning, args.iter().any(|a| a == "--any")).expect("a GPU");
    println!("{}, {tuning:?}, started in {:.1} s", gpu.name(), started.elapsed().as_secs_f64());
    for run in 1..=runs {
        let started = Instant::now();
        let mut gpu_seconds = 0.0;
        let mut all = ([0u8; 32], 0);
        // As the wallet does: the GPU goes on while the cores finish the last batch.
        std::thread::scope(|scope| {
            let (done, finishing) = std::sync::mpsc::sync_channel::<(&[Job], monokulo_8ar::gpu::Points)>(1);
            let finisher = scope.spawn(move || {
                let mut all = ([0u8; 32], 0);
                for (part, points) in finishing {
                    let result = in_parallel(part.len(), threads, 64, |at, end| {
                        let points: Vec<_> = (at..end).map(|i| points.get(i).expect("on the curve")).collect();
                        finish(&points, &part[at..end])
                    });
                    all.0.iter_mut().zip(result.0).for_each(|(a, d)| *a ^= d);
                    all.1 += result.1;
                }
                all
            });
            for part in jobs.chunks(tuning.batch) {
                let keys: Vec<[u8; 32]> = part.iter().map(|j| j.r).collect();
                let t = Instant::now();
                let points = gpu.derive(&digits, &keys).expect("the GPU ran");
                gpu_seconds += t.elapsed().as_secs_f64();
                done.send((part, points)).unwrap();
            }
            drop(done);
            all = finisher.join().unwrap();
        });
        show(run, started.elapsed().as_secs_f64(), gpu_seconds, all);
    }
}
