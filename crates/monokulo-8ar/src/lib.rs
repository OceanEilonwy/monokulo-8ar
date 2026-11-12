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
//! The arithmetic runs on the GPU, in Vulkan, and only there: a GPU passes test
//! vectors once per process before it's used, and again with each wallet's view
//! key (`vectors.rs`). Without a GPU that passed, or for a batch too small to be
//! worth one, [`Prefilter::flag`] says it didn't filter, and the caller scans
//! every block with the `Scanner`. Which way is a matter of speed and power,
//! never of the answer. [`engine`] says which is in use.
//!
//! A [`Keyring`] does the same for many accounts in one pass: every key with
//! every account's view key, and [`Keyring::flag_many`] says which
//! transactions may pay which account. [`Prefilter`] is a keyring of one. A
//! keyring works in an [`Arena`], its own share of the one GPU, so keyrings in
//! different arenas run side by side.
//!
//! Written for Fob, whose `docs/decisions.md` gives the reasons (decisions 331,
//! 332, 350, 351, 359, 360), and shared with monokulo-lws.

mod engine;
pub mod gpu;
mod vectors;

pub use engine::{Arena, Engine, Reason, check_wallet, engine, gpu_failed, gpu_identity, gpu_ready, rest, set_cache_dir, settle, warm_up};

use std::{
    collections::HashSet,
    ops::Range,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use gpu::{Derived, GpuError};

use curve25519_dalek::{EdwardsPoint, Scalar, constants::ED25519_BASEPOINT_TABLE, edwards::CompressedEdwardsY};
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
/// How many keys a thread takes at once.
const CHUNK: usize = 64;

/// Runs the check a GPU must pass before it's used: for the test tool.
pub fn startup_vectors(gpu: &gpu::Arena) -> Result<bool, gpu::GpuError> {
    vectors::passes(gpu)
}

/// Runs the check a wallet's own view key must pass: for the test tool.
pub fn wallet_vectors(gpu: &gpu::Arena, view: &Zeroizing<[u8; 32]>) -> Result<bool, gpu::GpuError> {
    vectors::passes_for_wallet(gpu, view)
}

/// A transaction key and the outputs to try with it.
struct Job<'a> {
    block: usize,
    /// 0 for the miner transaction, 1 on for the block's others.
    tx: usize,
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

/// Whether `bytes` encodes y below p, and isn't a negative zero: x is zero only
/// for y = 1 and y = -1. These are the encodings monero-oxide takes, before it
/// asks whether the point is on the curve.
fn canonical(bytes: &[u8; 32]) -> bool {
    let sign = bytes[31] >> 7;
    let top = bytes[31] & 0x7f;
    // p is 2^255 - 19: ed ff .. ff 7f, little-endian.
    let at_least_p = top == 0x7f && bytes[1..31].iter().all(|b| *b == 0xff) && bytes[0] >= 0xed;
    if at_least_p {
        return false;
    }
    let one = bytes[0] == 1 && bytes[1..31].iter().all(|b| *b == 0) && top == 0;
    let minus_one = bytes[0] == 0xec && bytes[1..31].iter().all(|b| *b == 0xff) && top == 0x7f;
    !(sign == 1 && (one || minus_one))
}

/// A key as the GPU may have it: it takes only canonical encodings, without a
/// negative zero, so whatever monero-oxide refuses for those reasons becomes the
/// identity, as Monero treats it. Off the curve, the GPU says so.
fn sendable(key: &[u8; 32]) -> [u8; 32] {
    if canonical(key) { *key } else { CompressedPoint::IDENTITY.to_bytes() }
}

/// The longest a usize takes as a varint.
const MOST_VARINT: usize = usize::BITS.div_ceil(7) as usize;

/// Writes `n` as a varint at the start of `out`, which has room for
/// [`MOST_VARINT`] bytes, and says how many bytes it took.
fn varint(mut n: usize, out: &mut [u8]) -> usize {
    let mut at = 0;
    while n >= 0x80 {
        out[at] = n as u8 | 0x80;
        n >>= 7;
        at += 1;
    }
    out[at] = n as u8;
    at + 1
}

/// Runs `each` over `items` on `threads` threads, a chunk at a time.
fn in_chunks<T: Sync>(items: &[T], threads: usize, chunk: usize, each: impl Fn(usize, &[T]) + Sync) {
    in_ranges(items.len(), threads, chunk, |range| each(range.start, &items[range]));
}

/// Runs `each` over `0..count` on `threads` threads, a chunk at a time.
fn in_ranges(count: usize, threads: usize, chunk: usize, each: impl Fn(Range<usize>) + Sync) {
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let at = next.fetch_add(chunk, Ordering::Relaxed);
            if at >= count {
                return;
            }
            each(at..(at + chunk).min(count));
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

/// 8a times the identity, compressed: the identity.
const IDENTITY: [u8; 32] = {
    let mut bytes = [0; 32];
    bytes[0] = 1;
    bytes
};

/// An account in a [`Keyring`]: the order it was added in, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AccountId(u32);

impl AccountId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A transaction that may pay an account: its block's place in the batch, and
/// `tx` 0 for the miner transaction, 1 on for `block.transactions[tx - 1]`.
/// [`Hit::WHOLE_BLOCK`] is a block the prefilter can't vouch for (a newer fork,
/// transactions missing), for the `Scanner` to read whole for the account.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hit {
    pub account: AccountId,
    pub block: usize,
    pub tx: usize,
}

impl Hit {
    /// `tx` for a block to read whole.
    pub const WHOLE_BLOCK: usize = usize::MAX;
}

/// What one account watches for: its view key and every spend key it could be paid at.
#[derive(Clone)]
struct Account {
    digits: gpu::Digits,
    spends: HashSet<[u8; 32]>,
}

impl Account {
    fn pays(&self, derivation: &[u8; 32], outputs: &Outputs) -> bool {
        let (prefix, range) = match outputs {
            Outputs::All(prefix) => (*prefix, 0..prefix.outputs.len()),
            Outputs::One(prefix, o) => (*prefix, *o..*o + 1),
        };
        // "view_tag", the derivation and the output's index: on the stack, as
        // this runs for every item, and wiped after.
        let mut message = Zeroizing::new([0u8; 8 + 32 + MOST_VARINT]);
        message[..8].copy_from_slice(b"view_tag");
        message[8..40].copy_from_slice(derivation);
        for o in range {
            let Some(output) = prefix.outputs.get(o) else { continue };
            if output.key == CompressedPoint::IDENTITY {
                continue;
            }
            let end = 40 + varint(o, &mut message[40..]);
            if let Some(tag) = output.view_tag
                && Keccak256::digest(&message[..end])[0] != tag
            {
                continue;
            }
            let Some(key) = output.key.decompress() else { continue };
            let key: EdwardsPoint = key.into();
            let hash = Zeroizing::new(<[u8; 32]>::from(Keccak256::digest(&message[8..end])));
            let shared = Zeroizing::new(Scalar::from_bytes_mod_order(*hash));
            let spend = key - (&*shared * ED25519_BASEPOINT_TABLE);
            if self.spends.contains(spend.compress().as_bytes()) {
                return true;
            }
        }
        false
    }
}

/// What works out 8aR for a grid of accounts by keys: an arena of the GPU, or
/// curve25519-dalek in the tests.
trait Deriver {
    /// The most items a slab takes.
    fn batch(&self) -> usize;
    /// The most keys and the most accounts a load takes.
    fn most(&self) -> (usize, usize);
    fn load(&mut self, accounts: &[&gpu::Digits], keys: &[[u8; 32]]) -> Result<(), GpuError>;
    fn slab(&mut self, first_item: u64, count: usize) -> Result<gpu::Points, GpuError>;
    fn unload(&mut self);
}

struct OnGpu<'a>(&'a gpu::Arena);

impl Deriver for OnGpu<'_> {
    fn batch(&self) -> usize {
        self.0.tuning().batch
    }

    fn most(&self) -> (usize, usize) {
        (self.0.most_keys(), self.0.most_accounts())
    }

    fn load(&mut self, accounts: &[&gpu::Digits], keys: &[[u8; 32]]) -> Result<(), GpuError> {
        self.0.load(accounts, keys)
    }

    fn slab(&mut self, first_item: u64, count: usize) -> Result<gpu::Points, GpuError> {
        self.0.slab(first_item, count)
    }

    fn unload(&mut self) {
        self.0.unload();
    }
}

/// The most items one load takes: the shader counts them in 32 bits.
const MOST_ITEMS: u64 = (1 << 31) - 1;

/// Many accounts' view keys, and every spend key each could be paid at: one GPU
/// pass works out 8aR for every transaction key with every account's view key.
/// The accounts by keys grid is never written out; the GPU takes a slab of it
/// at a time, and memory doesn't grow with accounts beyond 64 bytes each.
#[derive(Clone)]
pub struct Keyring {
    accounts: Vec<Account>,
    arena: Arena,
}

impl Default for Keyring {
    fn default() -> Keyring {
        Keyring::new()
    }
}

impl Keyring {
    /// An empty keyring on the shared arena, as a phone wants.
    pub fn new() -> Keyring {
        Keyring::with_arena(Arena::shared())
    }

    /// An empty keyring that works in `arena`, beside keyrings in other arenas.
    pub fn with_arena(arena: Arena) -> Keyring {
        Keyring { accounts: vec![], arena }
    }

    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Adds `pair`'s primary address. `view` is `pair`'s view key.
    pub fn add(&mut self, pair: &ViewPair, view: &Zeroizing<[u8; 32]>) -> AccountId {
        let id = AccountId(u32::try_from(self.accounts.len()).expect("fewer than 2^32 accounts"));
        self.accounts.push(Account { digits: gpu::Digits::new(view), spends: HashSet::from([pair.spend().compress().to_bytes()]) });
        id
    }

    /// Watches a subaddress of `pair`, `account`'s, as `Scanner::register_subaddress` does.
    pub fn watch(&mut self, account: AccountId, pair: &ViewPair, subaddress: SubaddressIndex) {
        // The spend key is the same on every network.
        self.accounts[account.index()].spends.insert(pair.subaddress(Network::Mainnet, subaddress).spend().compress().to_bytes());
    }

    pub fn len(&self) -> usize {
        self.accounts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.accounts.is_empty()
    }

    /// The transactions in `blocks` that may pay each account, or `None` if it
    /// didn't filter them: the GPU isn't ready or was retired (or failed now,
    /// which retires it), or the blocks hold fewer than `min_keys` transaction
    /// keys, where a GPU batch costs more than it saves (its cost is the
    /// caller's to weigh). The caller then scans every block for every account
    /// with the `Scanner`. Asking starts the GPU if it hasn't been. Hits come
    /// sorted, by account, block and transaction.
    pub fn flag_many(&self, blocks: &[ScannableBlock], threads: usize, min_keys: usize) -> Option<Vec<Hit>> {
        if !engine::usable() {
            return None;
        }
        self.arena.with(|gpu| self.flag_by(blocks, threads, min_keys, &mut OnGpu(gpu))).flatten()
    }

    /// [`flag_many`](Keyring::flag_many), with `deriver` doing the GPU's part.
    fn flag_by(&self, blocks: &[ScannableBlock], threads: usize, min_keys: usize, deriver: &mut impl Deriver) -> Result<Option<Vec<Hit>>, GpuError> {
        let found: Mutex<(Vec<Job>, Vec<usize>)> = Mutex::new((vec![], vec![]));
        let numbered: Vec<(usize, &ScannableBlock)> = blocks.iter().enumerate().collect();
        in_chunks(&numbered, threads, 16, |_, part| {
            let mut jobs = Vec::with_capacity(part.len() * 64);
            let mut whole = vec![];
            for (i, block) in part {
                if !jobs_of(*i, block, &mut jobs) {
                    whole.push(*i);
                }
            }
            let mut found = found.lock().expect("jobs lock");
            found.0.extend(jobs);
            found.1.extend(whole);
        });
        let (jobs, whole) = found.into_inner().expect("jobs lock");
        if jobs.len() < min_keys {
            return Ok(None);
        }
        let accounts = self.accounts.len();
        let mut hits: Vec<Hit> = whole.iter().flat_map(|&block| (0..accounts).map(move |a| Hit { account: AccountId(a as u32), block, tx: Hit::WHOLE_BLOCK })).collect();
        // As many loads as the GPU needs: each within its buffers' bounds, and
        // the grid's items countable in 32 bits.
        let (most_keys, most_accounts) = deriver.most();
        for first_account in (0..accounts).step_by(most_accounts.max(1)) {
            let these = first_account..accounts.min(first_account + most_accounts.max(1));
            let per_load = (MOST_ITEMS / these.len() as u64).max(1).min(most_keys.max(1) as u64) as usize;
            for part in jobs.chunks(per_load) {
                hits.extend(self.on_grid(part, these.clone(), threads, deriver)?);
            }
        }
        hits.sort_unstable();
        hits.dedup();
        Ok(Some(hits))
    }

    /// Every account with every one of `jobs`' keys: the deriver works out a
    /// slab, then the cores finish it, one after the other and never side by
    /// side, as a scan runs on the GPU or the CPU and never both.
    fn on_grid(&self, jobs: &[Job], accounts: Range<usize>, threads: usize, deriver: &mut impl Deriver) -> Result<Vec<Hit>, GpuError> {
        let keys: Vec<[u8; 32]> = jobs.iter().map(|job| sendable(&job.key)).collect();
        let digits: Vec<&gpu::Digits> = self.accounts[accounts.clone()].iter().map(|a| &a.digits).collect();
        deriver.load(&digits, &keys)?;
        let result = self.slabs(jobs, &keys, accounts, threads, deriver);
        deriver.unload();
        result
    }

    /// The slabs of what's loaded, each finished on the cores before the next.
    /// An answer the GPU can't have got right is an error, which retires it, and
    /// the caller then scans every block with the `Scanner`: a point that fails
    /// the GPU's own check or the CPU's, or a key said to be off the curve that
    /// isn't. Each answer stands alone, so one bad point says nothing of the
    /// others, but nothing from a faulty GPU is trusted.
    fn slabs(&self, jobs: &[Job], keys: &[[u8; 32]], accounts: Range<usize>, threads: usize, deriver: &mut impl Deriver) -> Result<Vec<Hit>, GpuError> {
        let items = (jobs.len() * accounts.len()) as u64;
        let batch = deriver.batch() as u64;
        let found = Mutex::new(vec![]);
        let wrong = AtomicBool::new(false);
        // Whether each key is off the curve, as curve25519-dalek finds when the GPU
        // says one is, which is rare: once a key, not once an account.
        let off_curve: Vec<OnceLock<bool>> = keys.iter().map(|_| OnceLock::new()).collect();
        let mut first = 0;
        while first < items {
            let points = deriver.slab(first, batch.min(items - first) as usize)?;
            in_ranges(points.len(), threads, CHUNK, |range| {
                let mut hits = vec![];
                for i in range {
                    let item = first + i as u64;
                    let (account, key) = (accounts.start + (item / jobs.len() as u64) as usize, (item % jobs.len() as u64) as usize);
                    let derivation = match points.derived(i) {
                        Derived::Point(point) => Zeroizing::new(point),
                        // Off the curve is the identity, as Monero treats it.
                        Derived::OffCurve if *off_curve[key].get_or_init(|| CompressedEdwardsY(keys[key]).decompress().is_none()) => Zeroizing::new(IDENTITY),
                        Derived::OffCurve | Derived::Wrong => {
                            wrong.store(true, Ordering::Relaxed);
                            return;
                        }
                    };
                    let job = &jobs[key];
                    if self.accounts[account].pays(&derivation, &job.outputs) {
                        hits.push(Hit { account: AccountId(account as u32), block: job.block, tx: job.tx });
                    }
                }
                if !hits.is_empty() {
                    found.lock().expect("hits lock").extend(hits);
                }
            });
            if wrong.load(Ordering::Relaxed) {
                return Err(GpuError { what: "a point the GPU got wrong".into(), kind: gpu::GpuErrorKind::Failed });
            }
            first += batch;
        }
        Ok(found.into_inner().expect("hits lock"))
    }
}

/// What a wallet watches for: a keyring of one account, its view key and every
/// spend key it could be paid at.
#[derive(Clone)]
pub struct Prefilter(Keyring);

impl Prefilter {
    /// A filter for `pair`'s primary address. `view` is `pair`'s view key.
    pub fn new(pair: &ViewPair, view: &Zeroizing<[u8; 32]>) -> Prefilter {
        let mut keyring = Keyring::new();
        keyring.add(pair, view);
        Prefilter(keyring)
    }

    /// Watches a subaddress of `pair`, as `Scanner::register_subaddress` does.
    pub fn watch(&mut self, pair: &ViewPair, subaddress: SubaddressIndex) {
        self.0.watch(AccountId(0), pair, subaddress);
    }

    /// Whether each of `blocks` could hold money for this wallet, or `None` if
    /// it didn't filter them, as [`Keyring::flag_many`] says.
    pub fn flag(&self, blocks: &[ScannableBlock], threads: usize, min_keys: usize) -> Option<Vec<bool>> {
        self.0.flag_many(blocks, threads, min_keys).map(|hits| flags(&hits, blocks.len()))
    }
}

/// One bool a block: whether any hit is in it.
fn flags(hits: &[Hit], blocks: usize) -> Vec<bool> {
    let mut flags = vec![false; blocks];
    for hit in hits {
        flags[hit.block] = true;
    }
    flags
}

/// Adds the work `block` needs to `jobs`. False if it's beyond checking, for the
/// `Scanner` to read whole, and then none of its work is added: a newer fork,
/// transactions missing, output indexes past u64::MAX, or a transaction whose
/// keys this can't read as monero-oxide does.
fn jobs_of<'a>(index: usize, block: &'a ScannableBlock, jobs: &mut Vec<Job<'a>>) -> bool {
    let before = jobs.len();
    let judged = block_jobs(index, block, jobs);
    if !judged {
        jobs.truncate(before);
    }
    judged
}

fn block_jobs<'a>(index: usize, block: &'a ScannableBlock, jobs: &mut Vec<Job<'a>>) -> bool {
    if block.block.transactions.len() != block.transactions.len() || block.block.header.hardfork_version > NEWEST_FORK {
        return false;
    }
    let Some(first_ringct_output) = block.output_index_for_first_ringct_output else { return true };
    let miner = block.block.miner_transaction();
    // The Scanner refuses a block whose RingCT outputs' indexes pass u64::MAX.
    let ringct_outputs = std::iter::once(miner.prefix())
        .filter(|_| matches!(miner, Transaction::V2 { .. }))
        .chain(block.transactions.iter().filter(|tx| matches!(tx, Transaction::V2 { .. })).map(Transaction::prefix))
        .try_fold(first_ringct_output, |at, prefix| at.checked_add(prefix.outputs.len() as u64));
    if ringct_outputs.is_none() {
        return false;
    }
    // False for a transaction it can't judge. The Scanner skips a v1
    // transaction and one without a key, as this does.
    let mut add = |tx: usize, version: u64, prefix: &'a TransactionPrefix| -> bool {
        if version != 2 {
            return true;
        }
        // Reading an extra from bytes doesn't fail, but if it ever did, the
        // Scanner would be the one to say what that means.
        let Ok(extra) = Extra::read(&mut prefix.extra.as_slice()) else { return false };
        let Some((keys, additional)) = encoded_keys(&extra) else {
            // No key, unless monero-oxide finds one this didn't.
            return extra.keys().is_none();
        };
        for key in keys {
            jobs.push(Job { block: index, tx, key, outputs: Outputs::All(prefix) });
        }
        for (o, key) in additional.into_iter().flatten().enumerate() {
            if o < prefix.outputs.len() {
                jobs.push(Job { block: index, tx, key, outputs: Outputs::One(prefix, o) });
            }
        }
        true
    };
    let version = |tx_is_v2: bool| if tx_is_v2 { 2 } else { 1 };
    add(0, version(matches!(miner, Transaction::V2 { .. })), miner.prefix())
        && block.transactions.iter().enumerate().all(|(i, tx)| add(i + 1, version(matches!(tx, Transaction::V2 { .. })), tx.prefix()))
}

#[cfg(test)]
mod tests;
