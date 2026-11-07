//! Computes 8·a·R for many Monero transaction keys at once, and flags the
//! blocks that may pay a wallet.
//!
//! Almost no block pays a given wallet, and proving that for one is nearly all
//! of a scan's cost: 8aR for each transaction key, a view tag for each output.
//! [`Prefilter`] does exactly that, faster than monero-oxide's `Scanner`, which
//! works out 8aR once per output rather than once per key. It decides nothing
//! about money: a block it flags goes to the `Scanner`, whole, which alone finds
//! the outputs, their amounts and their subaddresses. So its one duty is never
//! to miss, and anything it doesn't understand is flagged.
//!
//! The arithmetic runs on the CPU's cores or on the GPU ([`Mode`]). Which one is
//! a matter of speed and of power, never of the answer: a GPU passes test
//! vectors once per process before it's used, and with each wallet's view key
//! (`vectors.rs`), or the CPU does the work.
//!
//! Written for Fob, whose `docs/decisions.md` gives the reasons (decisions 331,
//! 332, 350, 351), and shared with monokulo-lws.

mod engine;
pub mod fe;
pub mod gpu;
mod vectors;

pub use engine::{Mode, check_wallet, gpu_failed, gpu_identity, gpu_ready, mode, rest, set_cache_dir, set_mode, warm_up};

use std::{
    collections::HashSet,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use curve25519_dalek::{EdwardsPoint, Scalar, constants::ED25519_BASEPOINT_TABLE};
use monero_wallet::{
    ViewPair,
    address::{Network, SubaddressIndex},
    ed25519::CompressedPoint,
    extra::Extra,
    interface::ScannableBlock,
    transaction::{Transaction, TransactionPrefix},
};
use sha3::{Digest, Keccak256};
use zeroize::Zeroizing;

/// `Scanner` refuses blocks past this fork, and must be the one to say so.
const NEWEST_FORK: u8 = 16;
/// How many keys share an inversion. A chunk is also how much a thread takes at once.
const CHUNK: usize = 64;

/// Runs the check a GPU must pass before it's used: for the test tool.
pub fn startup_vectors(gpu: &gpu::Gpu) -> Result<bool, gpu::GpuError> {
    vectors::passes(gpu)
}

/// Runs the check a wallet's own view key must pass: for the test tool.
pub fn wallet_vectors(gpu: &gpu::Gpu, view: &Zeroizing<[u8; 32]>) -> Result<bool, gpu::GpuError> {
    vectors::passes_for_wallet(gpu, view)
}

/// A transaction key and the outputs to try with it.
struct Job<'a> {
    block: usize,
    key: [u8; 32],
    outputs: Outputs<'a>,
}

enum Outputs<'a> {
    All(&'a TransactionPrefix),
    One(&'a TransactionPrefix, usize),
}

/// A transaction's keys, and its additional keys if it has them.
type TxKeys = (Vec<[u8; 32]>, Option<Vec<[u8; 32]>>);

/// The encoded form of a transaction's keys and its additional keys, as
/// `Extra::keys` gives them decompressed. Going through it would decompress and
/// compress each again for nothing; `Extra`'s fields are private, but it writes
/// them in order, keys first.
fn encoded_keys(extra: &Extra) -> Option<TxKeys> {
    let bytes = extra.serialize();
    let mut rest = bytes.as_slice();
    let mut keys = vec![];
    let mut additional = None;
    loop {
        match rest.first() {
            Some(1) => {
                keys.push(rest.get(1..33)?.try_into().ok()?);
                rest = &rest[33..];
            }
            Some(4) => {
                rest = &rest[1..];
                let (mut count, mut shift) = (0usize, 0u32);
                loop {
                    let (byte, after) = rest.split_first()?;
                    rest = after;
                    count |= usize::from(byte & 0x7f).checked_shl(shift)?;
                    shift += 7;
                    if byte & 0x80 == 0 {
                        break;
                    }
                }
                let size = count.checked_mul(32)?;
                let list: Vec<[u8; 32]> = rest.get(..size)?.as_chunks::<32>().0.to_vec();
                rest = &rest[size..];
                additional = additional.or(Some(list));
            }
            _ => break,
        }
    }
    (!keys.is_empty()).then_some((keys, additional))
}

/// 8a times the identity: the identity, in the projective form `finish` takes.
fn identity_point() -> (fe::Fe, fe::Fe, fe::Fe) {
    (fe::Fe([0; 5]), fe::Fe([1, 0, 0, 0, 0]), fe::Fe([1, 0, 0, 0, 0]))
}

/// A key as the GPU may have it: it takes only canonical encodings, without a
/// negative zero, so whatever `fe::Ext::decompress` refuses for those reasons
/// becomes the identity, as it does on the CPU. Off the curve, the GPU says so.
fn sendable(key: &[u8; 32]) -> [u8; 32] {
    if fe::canonical(key) { *key } else { CompressedPoint::IDENTITY.to_bytes() }
}

fn varint(mut n: usize, out: &mut Vec<u8>) {
    while n >= 0x80 {
        out.push(n as u8 | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

/// Runs `each` over `items` on `threads` threads, a chunk at a time.
fn in_chunks<T: Sync>(items: &[T], threads: usize, chunk: usize, each: impl Fn(usize, &[T]) + Sync) {
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let at = next.fetch_add(chunk, Ordering::Relaxed);
            if at >= items.len() {
                return;
            }
            each(at, &items[at..(at + chunk).min(items.len())]);
        }
    };
    if threads <= 1 {
        work();
    } else {
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(work);
            }
        });
    }
}

/// What a wallet watches for: its view key and every spend key it could be paid at.
#[derive(Clone)]
pub struct Prefilter {
    digits: fe::Digits,
    spends: HashSet<[u8; 32]>,
}

impl Prefilter {
    /// A filter for `pair`'s primary address. `view` is `pair`'s view key.
    pub fn new(pair: &ViewPair, view: &Zeroizing<[u8; 32]>) -> Prefilter {
        Prefilter { digits: fe::Digits::new(view), spends: HashSet::from([pair.spend().compress().to_bytes()]) }
    }

    /// Watches a subaddress of `pair`, as `Scanner::register_subaddress` does.
    pub fn watch(&mut self, pair: &ViewPair, subaddress: SubaddressIndex) {
        // The spend key is the same on every network.
        self.spends.insert(pair.subaddress(Network::Mainnet, subaddress).spend().compress().to_bytes());
    }

    /// Whether each of `blocks` could hold money for this wallet. Without
    /// `gpu_allowed`, the CPU does the work whatever the mode. In [`Mode::Auto`]
    /// the GPU takes over from `gpu_from` keys: the caller's choice, as where it
    /// starts to pay depends on the device and on what the caller cares about.
    pub fn flag(&self, blocks: &[ScannableBlock], threads: usize, gpu_allowed: bool, gpu_from: usize) -> Vec<bool> {
        let flags: Vec<AtomicBool> = blocks.iter().map(|_| AtomicBool::new(false)).collect();
        let found: std::sync::Mutex<Vec<Vec<Job>>> = std::sync::Mutex::new(vec![]);
        let numbered: Vec<(usize, &ScannableBlock)> = blocks.iter().enumerate().collect();
        in_chunks(&numbered, threads, 16, |_, part| {
            let mut jobs = Vec::with_capacity(part.len() * 64);
            for (i, block) in part {
                jobs_of(*i, block, &flags[*i], &mut jobs);
            }
            found.lock().expect("jobs lock").push(jobs);
        });
        let jobs: Vec<Job> = found.into_inner().expect("jobs lock").into_iter().flatten().collect();
        match engine::route(jobs.len(), gpu_allowed, gpu_from) {
            engine::Route::Cpu => in_chunks(&jobs, threads, CHUNK, |_, chunk| self.check(chunk, &flags)),
            engine::Route::Gpu => self.with_gpu(&jobs, &flags, threads),
        }
        flags.into_iter().map(AtomicBool::into_inner).collect()
    }

    /// The GPU works out every key's 8aR, a batch at a time, and the cores do
    /// what follows. A GPU that fails hands everything left to the cores.
    fn with_gpu(&self, jobs: &[Job], flags: &[AtomicBool], threads: usize) {
        std::thread::scope(|scope| {
            // The GPU goes on to its next batch while the cores finish its last.
            let (done, finishing) = std::sync::mpsc::sync_channel::<(&[Job], gpu::Points)>(1);
            scope.spawn(move || {
                for (part, points) in finishing {
                    in_chunks(part, threads, CHUNK, |at, chunk| {
                        // Off the curve is the identity, as for the CPU.
                        let points: Vec<_> = (at..at + chunk.len()).map(|i| points.get(i).unwrap_or_else(identity_point)).collect();
                        self.finish(&points, chunk, flags);
                    });
                }
            });
            let most = engine::batch();
            for (i, part) in jobs.chunks(most).enumerate() {
                let keys: Vec<[u8; 32]> = part.iter().map(|job| sendable(&job.key)).collect();
                match engine::derive(&self.digits, &keys) {
                    Some(points) => {
                        if let Err(unsent) = done.send((part, points)) {
                            // The finishing thread has gone only by panicking, which the scope passes on.
                            drop(unsent);
                            return;
                        }
                    }
                    // Failed, so this batch and the rest go to the cores.
                    None => {
                        in_chunks(&jobs[i * most..], threads, CHUNK, |_, chunk| self.check(chunk, flags));
                        return;
                    }
                }
            }
        });
    }

    fn check(&self, jobs: &[Job], flags: &[AtomicBool]) {
        // Points the scan's own `Extra::keys` would turn into the identity are
        // the identity here too, as Monero treats them.
        let identity = fe::Ext::decompress(&CompressedPoint::IDENTITY.to_bytes()).expect("the identity decompresses");
        let points: Vec<_> = jobs.iter().map(|job| fe::mul8(&self.digits, &fe::Ext::decompress(&job.key).unwrap_or(identity))).collect();
        self.finish(&points, jobs, flags);
    }

    fn finish(&self, points: &[(fe::Fe, fe::Fe, fe::Fe)], jobs: &[Job], flags: &[AtomicBool]) {
        let mut derivations = Zeroizing::new(Vec::with_capacity(jobs.len()));
        fe::compress_batch(points, &mut derivations);
        for (derivation, job) in derivations.iter().zip(jobs) {
            if !flags[job.block].load(Ordering::Relaxed) && self.pays_us(derivation, &job.outputs) {
                flags[job.block].store(true, Ordering::Relaxed);
            }
        }
    }

    fn pays_us(&self, derivation: &[u8; 32], outputs: &Outputs) -> bool {
        let (prefix, range) = match outputs {
            Outputs::All(prefix) => (*prefix, 0..prefix.outputs.len()),
            Outputs::One(prefix, o) => (*prefix, *o..*o + 1),
        };
        let mut message = Zeroizing::new(Vec::with_capacity(48));
        for o in range {
            let Some(output) = prefix.outputs.get(o) else { continue };
            if output.key == CompressedPoint::IDENTITY {
                continue;
            }
            if let Some(tag) = output.view_tag {
                message.clear();
                message.extend_from_slice(b"view_tag");
                message.extend_from_slice(derivation);
                varint(o, &mut message);
                if Keccak256::digest(&*message)[0] != tag {
                    continue;
                }
            }
            let Some(key) = output.key.decompress() else { continue };
            let key: EdwardsPoint = key.into();
            message.clear();
            message.extend_from_slice(derivation);
            varint(o, &mut message);
            let shared = Zeroizing::new(Scalar::from_bytes_mod_order(Keccak256::digest(&*message).into()));
            let spend = key - (&*shared * ED25519_BASEPOINT_TABLE);
            if self.spends.contains(spend.compress().as_bytes()) {
                return true;
            }
        }
        false
    }
}

/// Adds the work `block` needs to `jobs`, or flags it if it's beyond checking.
fn jobs_of<'a>(index: usize, block: &'a ScannableBlock, flag: &AtomicBool, jobs: &mut Vec<Job<'a>>) {
    if block.block.transactions.len() != block.transactions.len() || block.block.header.hardfork_version > NEWEST_FORK {
        flag.store(true, Ordering::Relaxed);
        return;
    }
    if block.output_index_for_first_ringct_output.is_none() {
        return;
    }
    let miner = block.block.miner_transaction();
    let mut add = |version: u64, prefix: &'a TransactionPrefix| {
        if version != 2 {
            return;
        }
        let Ok(extra) = Extra::read(&mut prefix.extra.as_slice()) else { return };
        let Some((keys, additional)) = encoded_keys(&extra) else { return };
        for key in keys {
            jobs.push(Job { block: index, key, outputs: Outputs::All(prefix) });
        }
        for (o, key) in additional.into_iter().flatten().enumerate() {
            if o < prefix.outputs.len() {
                jobs.push(Job { block: index, key, outputs: Outputs::One(prefix, o) });
            }
        }
    };
    let version = |tx_is_v2: bool| if tx_is_v2 { 2 } else { 1 };
    add(version(matches!(miner, Transaction::V2 { .. })), miner.prefix());
    for tx in &block.transactions {
        add(version(matches!(tx, Transaction::V2 { .. })), tx.prefix());
    }
}

#[cfg(test)]
mod tests;
