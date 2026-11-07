//! Scanning compared on one fixed set of blocks:
//!
//!   scan-bench capture <node> <start> <count> <file>   save the work to a file
//!   scan-bench run <file> [threads]                    time the engines on it
//!   scan-bench oxide <node> <start> <count> [threads]  fetch, time Fob's scanner
//!
//! The work is what a wallet does for outputs that aren't its own, almost all
//! of them: find 8aR for each transaction key, then a view tag for each output.
//! The 1 in 256 whose tag matches also get the shared key and the subtraction
//! from the output key. Each engine prints a digest of every 8aR, so they can't
//! get faster by getting it wrong.

mod fe;

use std::{
    collections::HashMap,
    io::{Read, Write},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use curve25519_dalek::{EdwardsPoint, Scalar as DalekScalar, constants::ED25519_BASEPOINT_TABLE, edwards::CompressedEdwardsY};
use fob_core::{fetch, transport::Transport};
use monero_wallet::{
    Scanner, ViewPair,
    address::SubaddressIndex,
    ed25519::{Point, Scalar},
    extra::Extra,
    interface::ScannableBlock,
    transaction::{Pruned, Transaction},
};
use sha3::{Digest, Keccak256};
use zeroize::Zeroizing;

/// One transaction key and the outputs to try with it.
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

fn jobs_of(tx: &Transaction<Pruned>, jobs: &mut Vec<Job>) {
    if tx.version() != 2 {
        return;
    }
    let Ok(extra) = Extra::read(&mut tx.prefix().extra.as_slice()) else { return };
    let Some((keys, additional)) = extra.keys() else { return };
    let outs = tx.prefix().outputs.iter().enumerate().filter_map(|(o, out)| Some(Out { index: o as u32, tag: out.view_tag?, key: out.key.to_bytes() }));
    let outs: Vec<Out> = outs.collect();
    for key in keys {
        let copies = outs.iter().map(|o| Out { index: o.index, tag: o.tag, key: o.key }).collect();
        jobs.push(Job { r: key.compress().to_bytes(), outs: copies });
    }
    if let Some(additional) = additional {
        for out in &outs {
            if let Some(key) = additional.get(out.index as usize) {
                jobs.push(Job { r: key.compress().to_bytes(), outs: vec![Out { index: out.index, tag: out.tag, key: out.key }] });
            }
        }
    }
}

fn write_work(work: &Work, path: &str) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    f.write_all(&work.blocks.to_le_bytes()).unwrap();
    f.write_all(&(work.jobs.len() as u64).to_le_bytes()).unwrap();
    for job in &work.jobs {
        f.write_all(&job.r).unwrap();
        f.write_all(&(job.outs.len() as u32).to_le_bytes()).unwrap();
        for out in &job.outs {
            f.write_all(&out.index.to_le_bytes()).unwrap();
            f.write_all(&[out.tag]).unwrap();
            f.write_all(&out.key).unwrap();
        }
    }
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

/// The wallet's view key and spend key in this benchmark: nobody's.
fn view_key() -> [u8; 32] {
    DalekScalar::from_bytes_mod_order(Keccak256::digest(b"scan-bench view key").into()).to_bytes()
}

fn varint(mut n: u32, out: &mut Vec<u8>) {
    while n >= 0x80 {
        out.push(n as u8 | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

struct Tally {
    digest: [u8; 32],
    matched: u64,
    found: u64,
}

impl Tally {
    fn new() -> Tally {
        Tally { digest: [0; 32], matched: 0, found: 0 }
    }
    fn merge(&mut self, other: &Tally) {
        for i in 0..32 {
            self.digest[i] ^= other.digest[i];
        }
        self.matched += other.matched;
        self.found += other.found;
    }
}

/// What follows the derivation, the same for every engine.
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
        let shared = DalekScalar::from_bytes_mod_order(Keccak256::digest(&message).into());
        let candidate: EdwardsPoint = key - (&shared * ED25519_BASEPOINT_TABLE);
        if spend.contains_key(candidate.compress().as_bytes()) {
            tally.found += 1;
        }
    }
}

/// Every thread takes the next chunk when it's done with the last, as Fob's scan does.
fn in_parallel(work: &Work, threads: usize, chunk: usize, each: impl Fn(&[Job], &mut Tally) + Sync) -> (Tally, f64) {
    let next = AtomicUsize::new(0);
    let started = Instant::now();
    let tallies: Vec<Tally> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut tally = Tally::new();
                    loop {
                        let at = next.fetch_add(chunk, Ordering::Relaxed);
                        if at >= work.jobs.len() {
                            return tally;
                        }
                        each(&work.jobs[at..(at + chunk).min(work.jobs.len())], &mut tally);
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let seconds = started.elapsed().as_secs_f64();
    let mut all = Tally::new();
    for t in &tallies {
        all.merge(t);
    }
    (all, seconds)
}

unsafe extern "C" {
    fn ge_frombytes_vartime(point: *mut u8, bytes: *const u8) -> i32;
    fn ge_scalarmult(out: *mut u8, scalar: *const u8, point: *const u8);
    fn ge_mul8(out: *mut u8, point: *const u8);
    fn ge_p1p1_to_p2(out: *mut u8, point: *const u8);
    fn ge_tobytes(out: *mut u8, point: *const u8);
}

/// Monero's generate_key_derivation, on its own curve code.
fn ref10_derivation(view: &[u8; 32], r: &[u8; 32]) -> Option<[u8; 32]> {
    #[repr(align(8))]
    struct Mem([u8; 160]);
    let (mut p3, mut p2, mut p1p1) = (Mem([0; 160]), Mem([0; 160]), Mem([0; 160]));
    let mut out = [0; 32];
    unsafe {
        if ge_frombytes_vartime(p3.0.as_mut_ptr(), r.as_ptr()) != 0 {
            return None;
        }
        ge_scalarmult(p2.0.as_mut_ptr(), view.as_ptr(), p3.0.as_ptr());
        ge_mul8(p1p1.0.as_mut_ptr(), p2.0.as_ptr());
        ge_p1p1_to_p2(p2.0.as_mut_ptr(), p1p1.0.as_ptr());
        ge_tobytes(out.as_mut_ptr(), p2.0.as_ptr());
    }
    Some(out)
}

fn run(path: &str, threads: usize) {
    let work = read_work(path);
    let view = view_key();
    let spend: HashMap<[u8; 32], u32> = (0..200u32)
        .map(|i| ((&DalekScalar::from(i as u64 + 1) * ED25519_BASEPOINT_TABLE).compress().to_bytes(), i))
        .collect();
    println!("{} blocks, {} transaction keys, {} outputs, {threads} threads", work.blocks, work.jobs.len(), work.outputs);
    let show = |name: &str, threads: usize, (tally, seconds): (Tally, f64)| {
        println!(
            "{name:<34} {threads:>2} threads {seconds:>8.3} s {:>8.0} blocks/s {:>7.0} keys/s digest {} tags {} found {}",
            work.blocks as f64 / seconds,
            work.jobs.len() as f64 / seconds,
            hex::encode(&tally.digest[..4]),
            tally.matched,
            tally.found
        );
    };
    let dalek_view = DalekScalar::from_bytes_mod_order(view);
    let dalek = |jobs: &[Job], tally: &mut Tally| {
        for job in jobs {
            let Some(r) = CompressedEdwardsY(job.r).decompress() else { continue };
            let derivation = (dalek_view * r).mul_by_cofactor().compress().to_bytes();
            after_derivation(&derivation, job, &spend, tally);
        }
    };
    let ref10 = |jobs: &[Job], tally: &mut Tally| {
        for job in jobs {
            let Some(derivation) = ref10_derivation(&view, &job.r) else { continue };
            after_derivation(&derivation, job, &spend, tally);
        }
    };
    let digits = fe::Digits::new(&view);
    let custom = |jobs: &[Job], tally: &mut Tally| {
        let mut points = Vec::with_capacity(jobs.len());
        let mut kept = Vec::with_capacity(jobs.len());
        for job in jobs {
            let Some(r) = fe::Ext::decompress(&job.r) else { continue };
            points.push(fe::mul8(&digits, &r));
            kept.push(job);
        }
        let mut compressed = Vec::with_capacity(points.len());
        fe::compress_batch(&points, &mut compressed);
        for (derivation, job) in compressed.iter().zip(kept) {
            after_derivation(derivation, job, &spend, tally);
        }
    };
    for t in [1, threads] {
        show("dalek, one 8aR per transaction key", t, in_parallel(&work, t, 16, dalek));
        show("wallet2's ref10, one per key", t, in_parallel(&work, t, 16, ref10));
        show("custom field, batch of 64", t, in_parallel(&work, t, 64, custom));
    }
}

async fn fetch_blocks(node: &str, start: u64, count: u64) -> Vec<ScannableBlock> {
    let transport = Transport::new(node, false).unwrap();
    let t = Instant::now();
    let mut blocks = Vec::with_capacity(count as usize);
    let mut from = start;
    while from < start + count {
        let end = (from + 500).min(start + count) - 1;
        blocks.extend(fetch::blocks(&transport, from, end).await.unwrap());
        from = blocks.last().map_or(from, |_| start + blocks.len() as u64);
    }
    let seconds = t.elapsed().as_secs_f64();
    println!("fetched {} blocks in {seconds:.2} s ({:.0} blocks/s)", blocks.len(), blocks.len() as f64 / seconds);
    blocks
}

fn oxide(blocks: Vec<ScannableBlock>, threads: usize) {
    let view = Scalar::hash(b"scan-bench");
    let spend = Point::from(&DalekScalar::from(7u64) * ED25519_BASEPOINT_TABLE);
    let pair = ViewPair::new(spend, Zeroizing::new(view)).unwrap();
    let mut scanner = Scanner::new(pair);
    for i in 1..=200 {
        scanner.register_subaddress(SubaddressIndex::new(0, i).unwrap());
    }
    let count = blocks.len();
    for t in [1, threads] {
        let next = AtomicUsize::new(0);
        let started = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..t {
                let mut scanner = scanner.clone();
                let (next, blocks) = (&next, &blocks);
                scope.spawn(move || {
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(block) = blocks.get(at) else { return };
                        scanner.scan(block.clone()).unwrap();
                    }
                });
            }
        });
        let seconds = started.elapsed().as_secs_f64();
        println!("{:<34} {t:>2} threads {seconds:>8.3} s {:>8.0} blocks/s", "monero-oxide Scanner (Fob today)", count as f64 / seconds);
    }
}

/// Whether a scanner still finds a wallet's money: a real block whose miner
/// transaction is rewritten to pay this wallet, in its second output.
async fn verify(node: &str, height: u64) {
    use monero_wallet::transaction::Timelock;
    let mut blocks = fetch_blocks(node, height, 1).await;
    let mut block = blocks.pop().unwrap();
    let spend_secret = DalekScalar::from(7u64);
    let spend = &spend_secret * ED25519_BASEPOINT_TABLE;
    let view = Scalar::hash(b"scan-bench");
    let dalek_view: DalekScalar = view.clone().into();
    let r = DalekScalar::from(1234567u64);
    let tx_key = &r * ED25519_BASEPOINT_TABLE;
    let derivation = (dalek_view * tx_key).mul_by_cofactor().compress().to_bytes();
    let mut tagged = b"view_tag".to_vec();
    tagged.extend_from_slice(&derivation);
    varint(1, &mut tagged);
    let tag = Keccak256::digest(&tagged)[0];
    let mut shared = derivation.to_vec();
    varint(1, &mut shared);
    let shared = DalekScalar::from_bytes_mod_order(Keccak256::digest(&shared).into());
    let key = (&shared * ED25519_BASEPOINT_TABLE) + spend;
    let mut extra = vec![1u8];
    extra.extend_from_slice(tx_key.compress().as_bytes());
    let mut miner = block.block.miner_transaction().clone();
    let Transaction::V2 { prefix, .. } = &mut miner else { panic!("a v2 miner transaction") };
    let first = prefix.outputs[0].clone();
    let mut mine = first.clone();
    mine.key = key.compress().to_bytes().into();
    mine.view_tag = Some(tag);
    mine.amount = Some(600_000_000_000);
    prefix.outputs = vec![first, mine];
    prefix.extra = extra;
    prefix.additional_timelock = Timelock::None;
    block.block = monero_wallet::block::Block::new(block.block.header.clone(), miner, block.block.transactions.clone()).unwrap();
    let pair = ViewPair::new(Point::from(spend), Zeroizing::new(view)).unwrap();
    let found = Scanner::new(pair).scan(block).unwrap().ignore_additional_timelock();
    println!("found {} output(s), amount {:?}, index in tx {:?}", found.len(), found.first().map(|f| f.commitment().amount), found.first().map(|f| f.index_in_transaction()));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].commitment().amount, 600_000_000_000);
    assert_eq!(found[0].index_in_transaction(), 1);
}

fn default_threads() -> usize {
    // As fob-core's scan: every core but one, at most eight.
    std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(1).clamp(1, 8))
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["capture", node, start, count, path] => {
            let blocks = fetch_blocks(node, start.parse().unwrap(), count.parse().unwrap()).await;
            let mut jobs = vec![];
            for block in &blocks {
                jobs_of(&Transaction::<Pruned>::from(block.block.miner_transaction().clone()), &mut jobs);
                for tx in &block.transactions {
                    jobs_of(tx, &mut jobs);
                }
            }
            let outputs = jobs.iter().map(|j| j.outs.len()).sum();
            let work = Work { blocks: blocks.len() as u64, jobs, outputs };
            println!("{} blocks, {} keys, {} outputs", work.blocks, work.jobs.len(), work.outputs);
            write_work(&work, path);
        }
        ["run", path] => run(path, default_threads()),
        ["run", path, threads] => run(path, threads.parse().unwrap()),
        ["oxide", node, start, count] => oxide(fetch_blocks(node, start.parse().unwrap(), count.parse().unwrap()).await, default_threads()),
        ["oxide", node, start, count, threads] => oxide(fetch_blocks(node, start.parse().unwrap(), count.parse().unwrap()).await, threads.parse().unwrap()),
        ["verify", node, height] => verify(node, height.parse().unwrap()).await,
        _ => eprintln!("see the top of main.rs"),
    }
}
