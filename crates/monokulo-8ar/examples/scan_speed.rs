//! Times the engines on a fixture written by prototypes/scan-bench:
//!
//!   scan_speed <file> [--keys N] [--threads N] [--batch N] [--windows N] [--every N] [--group N] [--runs N] [--any]
//!
//! `--keys N` runs the fixture through a keyring of N copies of the view key, as
//! a server with N accounts would: one load of the keys and N accounts' digits,
//! the grid taken a batch at a time, and every copy's digest checked.
//!
//! Each run does the arithmetic, the compression and every view tag, and prints
//! the XOR of all derivations: e0c4c620... for the 2000-block fixture.
//!
//!   scan_speed <file> --tip 1,10,100 [--seconds N] [--rounds N] [--period-ms N] [--cache dir]
//!
//! Measures the energy of small catch-ups on Android: for each size of catch-up,
//! rounds of idle and GPU with the GPU left open, as in an app. One batch each
//! period, long enough for the GPU to power down between them. The battery's
//! current is sampled once a second: what matters is the difference from idle in
//! the same round, as a charger may be supplying some of it. (The alternative to
//! the GPU is monero-oxide's `Scanner` on every block, which this fixture, keys
//! and view tags only, can't run.)

use std::{io::Read, sync::atomic::{AtomicUsize, Ordering}, time::Instant};

use monokulo_8ar::gpu::{Arena, Device, Digits, TUNING, Tuning};
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

fn finish(derivations: &[[u8; 32]], jobs: &[Job]) -> ([u8; 32], u64) {
    let mut digest = [0u8; 32];
    let mut hits = 0;
    for (derivation, job) in derivations.iter().zip(jobs) {
        digest.iter_mut().zip(derivation).for_each(|(a, d)| *a ^= d);
        hits += tags(derivation, job);
    }
    (digest, hits)
}

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter().position(|a| a == name).map_or(default, |i| args[i + 1].parse().expect("a number"))
}

fn gpu_batch(jobs: &[Job], digits: &Digits, threads: usize, gpu: &Arena) {
    let keys: Vec<[u8; 32]> = jobs.iter().map(|j| j.r).collect();
    let points = gpu.derive(&[digits], &keys).expect("the GPU ran");
    in_parallel(jobs.len(), threads, 64, |at, end| {
        let points: Vec<_> = (at..end).map(|i| points.get(i).expect("on the curve")).collect();
        finish(&points, &jobs[at..end])
    });
}

fn read_sys(name: &str) -> f64 {
    std::fs::read_to_string(format!("/sys/class/power_supply/battery/{name}")).expect("the battery").trim().parse().expect("a number")
}

fn tip(args: &[String], blocks: u64, jobs: &[Job], digits: &Digits, sizes: &str) {
    let seconds = arg(args, "--seconds", 40) as u64;
    let rounds = arg(args, "--rounds", 4);
    let period = std::time::Duration::from_millis(arg(args, "--period-ms", 2300) as u64);
    let device = Device::start(args.iter().position(|a| a == "--cache").map(|i| args[i + 1].clone().into()), TUNING, false).expect("a GPU");
    let gpu = Arena::new(&device).expect("an arena");
    let per_block = jobs.len() as f64 / blocks as f64;
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(1).clamp(1, 8);
    println!("{}, {per_block:.0} keys a block, one batch each {period:?}", gpu.name());
    println!("blocks keys threads | GPU ms mJ +- | batches short of the plan");

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
            let mut net = vec![];
            let mut busy = 0.0f64;
            let mut batches = 0usize;
            let mut short = 0;
            let mut at = 0;
            for round in 0..rounds {
                samples.lock().unwrap().clear();
                // 1 is idle, 0 the GPU.
                let order = if round % 2 == 0 { [1, 0] } else { [0, 1] };
                for which in order {
                    // A moment for the last arm's tail to pass.
                    arm.store(usize::MAX, Ordering::Relaxed);
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    arm.store(which, Ordering::Relaxed);
                    let started = Instant::now();
                    let (mut next, mut count) = (Instant::now(), 0);
                    while started.elapsed().as_secs() < seconds {
                        if which == 0 {
                            let part = &jobs[at..(at + keys).min(jobs.len())];
                            at = if at + 2 * keys > jobs.len() { 0 } else { at + keys };
                            let t = Instant::now();
                            gpu_batch(part, digits, threads, &gpu);
                            busy += t.elapsed().as_secs_f64();
                            batches += 1;
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
                net.push((mean(1) - mean(0)) * period.as_secs_f64() * 1000.0);
            }
            let n = net.len() as f64;
            let m = net.iter().sum::<f64>() / n;
            let se = (net.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0).max(1.0) / n).sqrt();
            println!("{size} {keys} {threads} | {:.1} ms {m:.0} mJ +-{se:.0} | {short}", busy / batches.max(1) as f64 * 1000.0);
        }
        done.store(true, Ordering::Relaxed);
    });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (blocks, jobs) = read_work(&args[0]);
    let view = Scalar::from_bytes_mod_order(Keccak256::digest(b"scan-bench view key").into()).to_bytes();
    let digits = Digits::new(&view);
    if let Some(i) = args.iter().position(|a| a == "--tip") {
        return tip(&args, blocks, &jobs, &digits, &args[i + 1]);
    }
    let runs = arg(&args, "--runs", 1);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(1).clamp(1, 8);
    let threads = arg(&args, "--threads", cores);
    let copies = arg(&args, "--keys", 1);
    let tuning = Tuning {
        batch: arg(&args, "--batch", TUNING.batch),
        windows: arg(&args, "--windows", TUNING.windows),
        submit_every: arg(&args, "--every", TUNING.submit_every),
        group: arg(&args, "--group", TUNING.group as usize) as u32,
    };
    let started = Instant::now();
    let device = Device::start(args.iter().position(|a| a == "--cache").map(|i| args[i + 1].clone().into()), tuning, args.iter().any(|a| a == "--any")).expect("a GPU");
    let gpu = Arena::new(&device).expect("an arena");
    println!("{}, {tuning:?}, {} queue(s), {copies} key(s), started in {:.1} s", device.name(), device.queues(), started.elapsed().as_secs_f64());
    let keys: Vec<[u8; 32]> = jobs.iter().map(|j| j.r).collect();
    let accounts: Vec<&Digits> = vec![&digits; copies];
    let items = (jobs.len() * copies) as u64;
    for run in 1..=runs {
        let started = Instant::now();
        let mut gpu_seconds = 0.0;
        // Each copy's digest and tags, all of which must be the one key's.
        let mut each = vec![([0u8; 32], 0u64); copies];
        // As a keyring does: the GPU works out a slab, then the cores finish it,
        // one after the other and never side by side.
        gpu.load(&accounts, &keys).expect("the GPU took the keys");
        let mut first = 0;
        while first < items {
            let t = Instant::now();
            let points = gpu.slab(first, (tuning.batch as u64).min(items - first) as usize).expect("the GPU ran");
            gpu_seconds += t.elapsed().as_secs_f64();
            // The slab a piece at a time, each piece within one account's row.
            let mut at = 0;
            while at < points.len() {
                let item = first + at as u64;
                let (account, key) = ((item / jobs.len() as u64) as usize, (item % jobs.len() as u64) as usize);
                let end = (at + jobs.len() - key).min(points.len());
                let result = in_parallel(end - at, threads, 64, |from, to| {
                    let derivations: Vec<_> = (at + from..at + to).map(|i| points.get(i).expect("on the curve")).collect();
                    finish(&derivations, &jobs[key + from..key + to])
                });
                each[account].0.iter_mut().zip(result.0).for_each(|(a, d)| *a ^= d);
                each[account].1 += result.1;
                at = end;
            }
            first += tuning.batch as u64;
        }
        gpu.unload();
        let seconds = started.elapsed().as_secs_f64();
        let hex: String = each[0].0[..4].iter().map(|b| format!("{b:02x}")).collect();
        println!(
            "run {run}: {seconds:.3} s, {:.0} blocks/s, {:.0} keys/s, {:.0} account-blocks/s, GPU part {gpu_seconds:.3} s, digest {hex}, tags {}",
            blocks as f64 / seconds,
            items as f64 / seconds,
            (blocks * copies as u64) as f64 / seconds,
            each[0].1
        );
        assert!(each.iter().all(|(digest, tags)| digest[..4] == [0xe0, 0xc4, 0xc6, 0x20] && *tags == 906), "the wrong answer");
    }
}
