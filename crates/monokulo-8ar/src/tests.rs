use curve25519_dalek::edwards::CompressedEdwardsY;
use monero_wallet::{
    Scanner,
    block::{Block, BlockHeader},
    ed25519::{Point, Scalar as MoneroScalar},
    transaction::{Input, Output, Timelock, Pruned},
};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};

use super::*;

fn random_scalar(rng: &mut StdRng) -> Scalar {
    let mut bytes = [0; 64];
    rng.fill_bytes(&mut bytes);
    Scalar::from_bytes_mod_order_wide(&bytes)
}

fn dalek_derivation(view: &Scalar, r: &EdwardsPoint) -> [u8; 32] {
    (view * r).mul_by_cofactor().compress().to_bytes()
}

/// The eight points of small order, on which curve code tends to go wrong.
fn small_order() -> Vec<[u8; 32]> {
    use curve25519_dalek::constants::EIGHT_TORSION;
    EIGHT_TORSION.iter().map(|p| p.compress().to_bytes()).collect()
}

/// What a sound GPU hands back for `key` with `view`, worked out by curve25519-dalek.
fn dalek_words(view: &Scalar, key: &[u8; 32], words: &mut Vec<u32>) {
    match CompressedEdwardsY(*key).decompress() {
        Some(r) => {
            let bytes = dalek_derivation(view, &r);
            words.extend(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)));
            words.push(0);
        }
        None => {
            words.extend([0; 8]);
            words.push(1);
        }
    }
}

fn dalek_points(view: &Scalar, keys: &[[u8; 32]]) -> gpu::Points {
    let mut words = vec![];
    for key in keys {
        dalek_words(view, key, &mut words);
    }
    gpu::Points::from_words(words)
}

/// curve25519-dalek in the GPU's place, for a keyring whose accounts' view keys
/// are `views`, in slabs of `batch` items.
struct OnDalek {
    views: Vec<Scalar>,
    keys: Vec<[u8; 32]>,
    batch: usize,
    /// An item whose words a faulty GPU gets wrong, and what it hands back.
    fault: Option<(u64, [u32; 9])>,
}

impl OnDalek {
    fn new(views: &[Scalar], batch: usize) -> OnDalek {
        OnDalek { views: views.to_vec(), keys: vec![], batch, fault: None }
    }
}

impl Deriver for OnDalek {
    fn batch(&self) -> usize {
        self.batch
    }

    fn load(&mut self, accounts: &[&gpu::Digits], keys: &[[u8; 32]]) -> Result<(), GpuError> {
        assert_eq!(accounts.len(), self.views.len(), "a view key for each account");
        self.keys = keys.to_vec();
        Ok(())
    }

    fn slab(&mut self, first_item: u64, count: usize) -> Result<gpu::Points, GpuError> {
        assert!(count <= self.batch);
        let mut words = vec![];
        for item in first_item as usize..first_item as usize + count {
            match self.fault {
                Some((at, wrong)) if at == item as u64 => words.extend(wrong),
                _ => dalek_words(&self.views[item / self.keys.len()], &self.keys[item % self.keys.len()], &mut words),
            }
        }
        Ok(gpu::Points::from_words(words))
    }

    fn unload(&mut self) {
        self.keys.clear();
    }
}

/// A keyring's hits with curve25519-dalek doing the GPU's part.
fn dalek_hits(keyring: &Keyring, views: &[Scalar], blocks: &[ScannableBlock], threads: usize, batch: usize) -> Vec<Hit> {
    keyring.flag_by(blocks, threads, 0, &mut OnDalek::new(views, batch)).expect("dalek doesn't fail").expect("it filtered")
}

/// How a test has blocks flagged: the GPU, or curve25519-dalek in its place.
type Flagger = fn(&Prefilter, &Scalar, &[ScannableBlock], usize) -> Vec<bool>;

/// The prefilter with curve25519-dalek doing the GPU's part, in batches of 100
/// keys so that a few blocks take several.
fn by_dalek(filter: &Prefilter, view: &Scalar, blocks: &[ScannableBlock], threads: usize) -> Vec<bool> {
    flags(&dalek_hits(&filter.0, &[*view], blocks, threads, 100), blocks.len())
}

fn by_gpu(filter: &Prefilter, _view: &Scalar, blocks: &[ScannableBlock], threads: usize) -> Vec<bool> {
    filter.flag(blocks, threads, 0).expect("the GPU filtered")
}

/// The first small y that isn't on the curve, canonically encoded.
fn off_curve() -> [u8; 32] {
    (2u8..).map(|y| {
        let mut bytes = [0; 32];
        bytes[0] = y;
        bytes
    }).find(|bytes| CompressedEdwardsY(*bytes).decompress().is_none()).unwrap()
}

#[test]
fn the_dalek_stand_in_gives_what_a_gpu_gives() {
    let view = Scalar::from(3u8);
    let bad = off_curve();
    let points = dalek_points(&view, &[small_order()[1], bad]);
    assert_eq!(points.len(), 2);
    assert_eq!(points.get(0), Some(dalek_derivation(&view, &CompressedEdwardsY(small_order()[1]).decompress().unwrap())));
    assert_eq!(points.get(1), None, "off the curve");
}

/// Exactly the encodings monero-oxide takes go to the GPU as they are: the rest go
/// as the identity, and the GPU refuses whatever's off the curve.
#[test]
fn what_reaches_the_gpu_is_what_monero_oxide_takes() {
    let check = |bytes: [u8; 32]| {
        let theirs = CompressedPoint::from(bytes).decompress().is_some();
        let ours = canonical(&bytes) && CompressedEdwardsY(bytes).decompress().is_some();
        assert_eq!(theirs, ours, "{bytes:?}");
        if !canonical(&bytes) {
            assert_eq!(sendable(&bytes), CompressedPoint::IDENTITY.to_bytes());
        }
    };
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..3000 {
        let mut bytes = [0; 32];
        rng.fill_bytes(&mut bytes);
        check(bytes);
    }
    // y at and past p, negative zeros, and each small-order point with its sign flipped.
    let mut p = [0xff; 32];
    p[0] = 0xed;
    p[31] = 0x7f;
    for low in 0..=40u8 {
        let mut near = p;
        near[0] = 0xed_u8.wrapping_add(low);
        if low >= 19 {
            near[0] = low - 19;
            near[1..31].fill(0);
            near[31] = 0;
        }
        check(near);
        near[31] |= 0x80;
        check(near);
    }
    check([0xff; 32]);
    for point in small_order() {
        let mut flipped = point;
        flipped[31] ^= 0x80;
        check(point);
        check(flipped);
    }
    assert!(CompressedEdwardsY(small_order()[0]).decompress().is_some());
}

// Blocks made to pay a wallet, or nearly: each is held to the Scanner's own answer.

struct Wallet {
    view: Scalar,
    spend: EdwardsPoint,
    pair: ViewPair,
}

const SUBADDRESS: u32 = 5;

fn wallet() -> (Wallet, Prefilter, Scanner) {
    let view_bytes = Zeroizing::new([9u8; 32]);
    let view_monero = MoneroScalar::read(&mut view_bytes.as_slice()).unwrap();
    let view: Scalar = view_monero.into();
    let spend = &Scalar::from(7u64) * ED25519_BASEPOINT_TABLE;
    let pair = ViewPair::new(Point::from(spend), Zeroizing::new(view_monero)).unwrap();
    let mut filter = Prefilter::new(&pair, &view_bytes);
    let mut scanner = Scanner::new(pair.clone());
    let index = SubaddressIndex::new(0, SUBADDRESS).unwrap();
    filter.watch(&pair, index);
    scanner.register_subaddress(index);
    (Wallet { view, spend, pair }, filter, scanner)
}

/// An output of the transaction with key `r`, at `index`, paying `to`.
fn paying(wallet: &Wallet, r: &Scalar, index: usize, to: EdwardsPoint) -> Output {
    let derivation = dalek_derivation(&wallet.view, &(r * ED25519_BASEPOINT_TABLE));
    let mut tagged = b"view_tag".to_vec();
    tagged.extend_from_slice(&derivation);
    varint(index, &mut tagged);
    let mut shared = derivation.to_vec();
    varint(index, &mut shared);
    let shared = Scalar::from_bytes_mod_order(Keccak256::digest(&shared).into());
    let key = &shared * ED25519_BASEPOINT_TABLE + to;
    Output { amount: Some(1_000), key: CompressedPoint::from(key.compress().to_bytes()), view_tag: Some(Keccak256::digest(&tagged)[0]) }
}

fn stranger(rng: &mut StdRng) -> Output {
    let key = &random_scalar(rng) * ED25519_BASEPOINT_TABLE;
    Output { amount: Some(1_000), key: CompressedPoint::from(key.compress().to_bytes()), view_tag: Some(rng.r#gen()) }
}

fn extra_of(keys: &[[u8; 32]], additional: Option<&[[u8; 32]]>) -> Vec<u8> {
    let mut extra = vec![];
    for key in keys {
        extra.push(1);
        extra.extend_from_slice(key);
    }
    if let Some(list) = additional {
        extra.push(4);
        extra.push(list.len() as u8);
        for key in list {
            extra.extend_from_slice(key);
        }
    }
    extra
}

fn point_of(r: &Scalar) -> [u8; 32] {
    (r * ED25519_BASEPOINT_TABLE).compress().to_bytes()
}

fn block_of(outputs: Vec<Output>, extra: Vec<u8>) -> ScannableBlock {
    block_with(outputs, extra, 16, 0)
}

fn block_with(outputs: Vec<Output>, extra: Vec<u8>, fork: u8, version: u8) -> ScannableBlock {
    let prefix = TransactionPrefix { additional_timelock: Timelock::None, inputs: vec![Input::Gen(100)], outputs, extra };
    let miner = if version == 1 { Transaction::V1 { prefix, signatures: vec![] } } else { Transaction::V2 { prefix, proofs: None } };
    let header = BlockHeader { hardfork_version: fork, hardfork_signal: fork, timestamp: 1, previous: [0; 32], nonce: 0 };
    ScannableBlock { block: Block::new(header, miner, vec![]).unwrap(), transactions: vec![], output_index_for_first_ringct_output: Some(0) }
}

fn scanner_finds(scanner: &Scanner, block: ScannableBlock) -> bool {
    match scanner.clone().scan(block) {
        Ok(found) => !found.ignore_additional_timelock().is_empty(),
        Err(_) => true,
    }
}

fn agree(name: &str, flag: Flagger, w: &Wallet, filter: &Prefilter, scanner: &Scanner, block: ScannableBlock, expected: bool) {
    assert_eq!(scanner_finds(scanner, block.clone()), expected, "{name}: the Scanner's answer isn't the one this case is about");
    assert_eq!(flag(filter, &w.view, std::slice::from_ref(&block), 1)[0], expected, "{name}");
}

#[test]
fn every_way_of_being_paid_is_flagged_and_strangers_are_not() {
    every_way_of_being_paid(by_dalek);
}

/// The same on the GPU. Needs a real GPU: it's started once for the process,
/// so the other tests keep to curve25519-dalek.
#[test]
#[ignore = "needs a GPU; run with --ignored"]
fn the_gpu_flags_the_same_blocks() {
    let (w, filter, _) = wallet();
    assert_eq!(filter.flag(&[], 1, 0), None, "not started: asking starts it");
    for _ in 0..1200 {
        if engine::gpu_ready() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(engine::gpu_ready(), "the GPU started");
    assert!(matches!(engine(), Engine::Gpu { .. }));
    every_way_of_being_paid(by_gpu);
    many_blocks(by_gpu);
    let mut rng = StdRng::seed_from_u64(8);
    // More than one of the GPU's batches.
    let mut blocks = vec![];
    let mut expected = vec![];
    for i in 0..(crate::gpu::TUNING.batch + 300) {
        let r = random_scalar(&mut rng);
        let pays = i % 500 == 3;
        let output = if pays { paying(&w, &r, 0, w.spend) } else { stranger(&mut rng) };
        blocks.push(block_of(vec![output], extra_of(&[point_of(&r)], None)));
        expected.push(pays);
    }
    assert_eq!(filter.flag(&blocks, 4, 0), Some(expected.clone()));
    assert_eq!(by_dalek(&filter, &w.view, &blocks, 4), expected);
    // Too few keys to be worth the GPU: the caller scans every block.
    assert_eq!(filter.flag(&blocks, 4, blocks.len() + 1), None);
    assert!(engine::gpu_ready(), "and it hasn't been retired");
}

fn every_way_of_being_paid(flag: Flagger) {
    let (w, filter, scanner) = wallet();
    let mut rng = StdRng::seed_from_u64(5);
    let r = random_scalar(&mut rng);
    let sub = w.pair.subaddress(Network::Mainnet, SubaddressIndex::new(0, SUBADDRESS).unwrap()).spend();
    let sub: EdwardsPoint = sub.into();
    let unwatched: EdwardsPoint = w.pair.subaddress(Network::Mainnet, SubaddressIndex::new(0, SUBADDRESS + 1).unwrap()).spend().into();
    let key = point_of(&r);

    agree("primary, first output", flag, &w, &filter, &scanner, block_of(vec![paying(&w, &r, 0, w.spend)], extra_of(&[key], None)), true);
    agree("primary, later output", flag, &w, &filter, &scanner, block_of(vec![stranger(&mut rng), stranger(&mut rng), paying(&w, &r, 2, w.spend)], extra_of(&[key], None)), true);
    agree("subaddress", flag, &w, &filter, &scanner, block_of(vec![stranger(&mut rng), paying(&w, &r, 1, sub)], extra_of(&[key], None)), true);
    agree("unwatched subaddress", flag, &w, &filter, &scanner, block_of(vec![paying(&w, &r, 0, unwatched)], extra_of(&[key], None)), false);
    agree("a stranger", flag, &w, &filter, &scanner, block_of(vec![stranger(&mut rng)], extra_of(&[key], None)), false);

    // The tag's a filter, and absent on old outputs.
    let mut no_tag = paying(&w, &r, 0, w.spend);
    no_tag.view_tag = None;
    agree("no view tag", flag, &w, &filter, &scanner, block_of(vec![no_tag], extra_of(&[key], None)), true);
    let mut wrong_tag = paying(&w, &r, 0, w.spend);
    wrong_tag.view_tag = Some(wrong_tag.view_tag.unwrap().wrapping_add(1));
    agree("wrong view tag", flag, &w, &filter, &scanner, block_of(vec![wrong_tag], extra_of(&[key], None)), false);

    // Several keys, and additional keys, each with its own output.
    let other = random_scalar(&mut rng);
    agree("second key", flag, &w, &filter, &scanner, block_of(vec![paying(&w, &r, 0, w.spend)], extra_of(&[point_of(&other), key], None)), true);
    let additional = [point_of(&other), key];
    agree("additional key", flag, &w, &filter, &scanner, block_of(vec![stranger(&mut rng), paying(&w, &r, 1, w.spend)], extra_of(&[point_of(&other)], Some(&additional))), true);
    let swapped = [key, point_of(&other)];
    agree("additional key on another output", flag, &w, &filter, &scanner, block_of(vec![stranger(&mut rng), paying(&w, &r, 1, w.spend)], extra_of(&[point_of(&other)], Some(&swapped))), false);
    let mut twice = extra_of(&[point_of(&other)], Some(&[point_of(&other), point_of(&other)]));
    twice.extend_from_slice(&extra_of(&[], Some(&additional))[..]);
    agree("only the first list of additional keys", flag, &w, &filter, &scanner, block_of(vec![stranger(&mut rng), paying(&w, &r, 1, w.spend)], twice), false);
    agree("additional keys and no key", flag, &w, &filter, &scanner, block_of(vec![paying(&w, &r, 0, w.spend)], extra_of(&[], Some(&[key]))), false);

    // Bad keys count as the identity, as they do for Monero: with the identity's
    // derivation, money can be paid to one.
    let mut bad = [0xff; 32];
    bad[31] = 0x7f;
    assert!(CompressedPoint::from(bad).decompress().is_none());
    let zero = Scalar::ZERO;
    let identity_pays = |rng: &mut StdRng, key: [u8; 32]| block_of(vec![stranger(rng), paying(&w, &zero, 1, w.spend)], extra_of(&[key], None));
    agree("a badly encoded key", flag, &w, &filter, &scanner, identity_pays(&mut rng, bad), true);
    // One encoded well, off the curve: the GPU says so, and it's the identity too.
    agree("a key off the curve", flag, &w, &filter, &scanner, identity_pays(&mut rng, off_curve()), true);

    // Output keys that can't be scanned.
    let mut identity_out = stranger(&mut rng);
    identity_out.key = CompressedPoint::IDENTITY;
    agree("identity output", flag, &w, &filter, &scanner, block_of(vec![identity_out], extra_of(&[key], None)), false);
    let mut undecodable = stranger(&mut rng);
    undecodable.key = CompressedPoint::from(bad);
    undecodable.view_tag = None;
    agree("undecodable output", flag, &w, &filter, &scanner, block_of(vec![undecodable], extra_of(&[key], None)), false);

    // Not scanned at all.
    agree("v1 transaction", flag, &w, &filter, &scanner, block_with(vec![paying(&w, &r, 0, w.spend)], extra_of(&[key], None), 16, 1), false);
    agree("unreadable extra", flag, &w, &filter, &scanner, block_of(vec![paying(&w, &r, 0, w.spend)], vec![0x77, 1, 2, 3]), false);
    let mut no_ringct = block_of(vec![paying(&w, &r, 0, w.spend)], extra_of(&[key], None));
    no_ringct.output_index_for_first_ringct_output = None;
    agree("no RingCT outputs", flag, &w, &filter, &scanner, no_ringct, false);

    // The Scanner's refusals stay the Scanner's to make.
    agree("a newer fork", flag, &w, &filter, &scanner, block_with(vec![stranger(&mut rng)], extra_of(&[key], None), NEWEST_FORK + 1, 0), true);
    let mut mismatched = block_of(vec![stranger(&mut rng)], extra_of(&[key], None));
    mismatched.block = Block::new(mismatched.block.header.clone(), mismatched.block.miner_transaction().clone(), vec![[1; 32]]).unwrap();
    agree("transactions missing", flag, &w, &filter, &scanner, mismatched, true);
}

#[test]
fn payments_in_ordinary_transactions_are_flagged_among_many_blocks() {
    many_blocks(by_dalek);
}

fn many_blocks(flag: Flagger) {
    let (w, filter, scanner) = wallet();
    let (blocks, expected) = many_blocks_for(&w);
    assert_eq!(flag(&filter, &w.view, &blocks, 4), expected);
    assert_eq!(flag(&filter, &w.view, &blocks, 1), expected);
    for (block, pays) in blocks.into_iter().zip(expected) {
        assert_eq!(scanner_finds(&scanner, block), pays, "the Scanner has the same answer");
    }
}

/// 300 blocks of three ordinary transactions and a miner's, every 37th paying
/// `w` in its second transaction; and which pay.
fn many_blocks_for(w: &Wallet) -> (Vec<ScannableBlock>, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(6);
    let mut blocks = vec![];
    let mut expected = vec![];
    for i in 0..300u32 {
        let mut block = block_of(vec![stranger(&mut rng)], extra_of(&[point_of(&random_scalar(&mut rng))], None));
        let pays = i % 37 == 0;
        for t in 0..3 {
            let r = random_scalar(&mut rng);
            let outputs = if pays && t == 1 { vec![stranger(&mut rng), paying(w, &r, 1, w.spend)] } else { vec![stranger(&mut rng), stranger(&mut rng)] };
            let prefix = TransactionPrefix { additional_timelock: Timelock::None, inputs: vec![], outputs, extra: extra_of(&[point_of(&r)], None) };
            block.transactions.push(Transaction::<Pruned>::V2 { prefix, proofs: None });
        }
        let hashes = vec![[0; 32]; block.transactions.len()];
        block.block = Block::new(block.block.header.clone(), block.block.miner_transaction().clone(), hashes).unwrap();
        expected.push(pays);
        blocks.push(block);
    }
    (blocks, expected)
}

/// The blocks a keyring is tried on: `many_blocks_for`'s, then one paying the
/// subaddress through an additional key in its miner transaction, and one of a
/// newer fork, which every account must read whole.
fn keyring_blocks(w: &Wallet) -> Vec<ScannableBlock> {
    let (mut blocks, _) = many_blocks_for(w);
    let mut rng = StdRng::seed_from_u64(11);
    let r = random_scalar(&mut rng);
    let sub: EdwardsPoint = w.pair.subaddress(Network::Mainnet, SubaddressIndex::new(0, SUBADDRESS).unwrap()).spend().into();
    let other = point_of(&random_scalar(&mut rng));
    blocks.push(block_of(vec![stranger(&mut rng), paying(w, &r, 1, sub)], extra_of(&[other], Some(&[other, point_of(&r)]))));
    blocks.push(block_with(vec![stranger(&mut rng)], extra_of(&[other], None), NEWEST_FORK + 1, 0));
    blocks
}

/// A keyring of `n` copies of the test wallet.
fn copies(w: &Wallet, n: usize) -> Keyring {
    let mut keyring = Keyring::new();
    for _ in 0..n {
        let id = keyring.add(&w.pair, &Zeroizing::new(w.view.to_bytes()));
        keyring.watch(id, &w.pair, SubaddressIndex::new(0, SUBADDRESS).unwrap());
    }
    keyring
}

/// `hits` that are `account`'s, as account 0's.
fn of_account(hits: &[Hit], account: usize) -> Vec<Hit> {
    hits.iter().filter(|h| h.account.index() == account).map(|h| Hit { account: AccountId(0), ..*h }).collect()
}

fn copies_flag_what_one_flags(hits_of: impl Fn(&Keyring, &[Scalar], &[ScannableBlock]) -> Vec<Hit>) {
    let (w, _, _) = wallet();
    let blocks = keyring_blocks(&w);
    let one = hits_of(&copies(&w, 1), &[w.view], &blocks);
    let (_, mut expected) = many_blocks_for(&w);
    expected.extend([true, true]);
    assert_eq!(flags(&one, blocks.len()), expected, "one key flags the blocks that pay it");
    let last = blocks.len() - 1;
    assert!(one.contains(&Hit { account: AccountId(0), block: last, tx: Hit::WHOLE_BLOCK }), "the newer fork, whole");
    assert!(one.contains(&Hit { account: AccountId(0), block: last - 1, tx: 0 }), "the miner transaction's additional key");
    assert!(one.contains(&Hit { account: AccountId(0), block: 0, tx: 2 }), "an ordinary transaction, by its place");
    for n in [3, 50] {
        let views = vec![w.view; n];
        let many = hits_of(&copies(&w, n), &views, &blocks);
        assert_eq!(many.len(), one.len() * n, "{n} copies");
        for account in 0..n {
            assert_eq!(of_account(&many, account), one, "copy {account} of {n}");
        }
    }
}

#[test]
fn a_keyring_of_copies_flags_what_one_key_flags() {
    copies_flag_what_one_flags(|keyring, views, blocks| dalek_hits(keyring, views, blocks, 4, 1000));
}

/// Accounts that aren't paid see only the block they must read whole, and the
/// one that is paid sees its own hits, whatever its place.
fn each_account_has_its_own_hits(hits_of: impl Fn(&Keyring, &[Scalar], &[ScannableBlock]) -> Vec<Hit>) {
    let (w, _, _) = wallet();
    let blocks = keyring_blocks(&w);
    let one = hits_of(&copies(&w, 1), &[w.view], &blocks);
    let mut rng = StdRng::seed_from_u64(12);
    let mut keyring = Keyring::new();
    let mut views = vec![];
    for i in 0..5 {
        let (pair, view) = if i == 3 {
            (w.pair.clone(), w.view)
        } else {
            let view = random_scalar(&mut rng);
            let spend = &random_scalar(&mut rng) * ED25519_BASEPOINT_TABLE;
            (ViewPair::new(Point::from(spend), Zeroizing::new(MoneroScalar::read(&mut view.to_bytes().as_slice()).unwrap())).unwrap(), view)
        };
        let id = keyring.add(&pair, &Zeroizing::new(view.to_bytes()));
        keyring.watch(id, &pair, SubaddressIndex::new(0, SUBADDRESS).unwrap());
        views.push(view);
    }
    let hits = hits_of(&keyring, &views, &blocks);
    assert_eq!(of_account(&hits, 3), one, "the wallet's own");
    for stranger in [0, 1, 2, 4] {
        assert_eq!(of_account(&hits, stranger), [Hit { account: AccountId(0), block: blocks.len() - 1, tx: Hit::WHOLE_BLOCK }], "account {stranger}");
    }
}

#[test]
fn each_account_in_a_keyring_has_its_own_hits() {
    each_account_has_its_own_hits(|keyring, views, blocks| dalek_hits(keyring, views, blocks, 2, 333));
}

#[test]
fn a_keyring_with_no_accounts_or_too_few_keys() {
    let (w, _, _) = wallet();
    let blocks = keyring_blocks(&w);
    assert_eq!(dalek_hits(&Keyring::new(), &[], &blocks, 1, 100), [], "no accounts, no hits");
    let one = copies(&w, 1);
    assert_eq!(one.flag_by(&blocks, 1, usize::MAX, &mut OnDalek::new(&[w.view], 100)).unwrap(), None, "under min_keys");
}

/// The same on the GPU, after `the_gpu_flags_the_same_blocks`' warm-up.
#[test]
#[ignore = "needs a GPU; run with --ignored"]
fn a_keyring_on_the_gpu_flags_what_dalek_does() {
    warm_up();
    assert!(settle(std::time::Duration::from_secs(120)) && engine::gpu_ready(), "the GPU started");
    let on_gpu = |keyring: &Keyring, _: &[Scalar], blocks: &[ScannableBlock]| keyring.flag_many(blocks, 4, 0).expect("the GPU filtered");
    copies_flag_what_one_flags(on_gpu);
    each_account_has_its_own_hits(on_gpu);
    // Two keyrings with an arena each, at once.
    let (w, _, _) = wallet();
    let blocks = keyring_blocks(&w);
    let mut first = copies(&w, 3);
    first.arena = Arena::new();
    let mut second = copies(&w, 7);
    second.arena = Arena::new();
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| first.flag_many(&blocks, 4, 0));
        let b = scope.spawn(|| second.flag_many(&blocks, 4, 0));
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(a.unwrap(), dalek_hits(&first, &[w.view; 3], &blocks, 4, 1000));
    assert_eq!(b.unwrap(), dalek_hits(&second, &[w.view; 7], &blocks, 4, 1000));
    assert!(engine::gpu_ready(), "and it hasn't been retired");
}

/// Any answer a sound GPU can't give is an error, which retires the GPU, so the
/// caller scans every block: never hits that leave a payment out.
#[test]
fn an_answer_the_gpu_got_wrong_is_never_taken() {
    let (w, _, _) = wallet();
    let blocks = keyring_blocks(&w);
    let keyring = copies(&w, 2);
    let views = [w.view; 2];
    // One thread, so the jobs are in the order `jobs_of` gives them.
    let sound = keyring.flag_by(&blocks, 1, 0, &mut OnDalek::new(&views, 100)).unwrap().unwrap();
    // A point a slab in, on the second account: what a zero Z comes out as, with
    // either sign; the GPU's own check failing; a flag it doesn't know; a y past
    // p; and a key that's on the curve said to be off it.
    let mut past_p = [u32::MAX; 9];
    past_p[7] = 0x7fff_ffff;
    past_p[8] = 0;
    let mut odd_zero = [0; 9];
    odd_zero[7] = 0x8000_0000;
    let mut good = vec![];
    dalek_words(&w.view, &point_of(&Scalar::ONE), &mut good);
    let mut failed_check: [u32; 9] = good.clone().try_into().unwrap();
    failed_check[8] = 2;
    let mut unknown_flag = failed_check;
    unknown_flag[8] = 7;
    let mut refused = [0; 9];
    refused[8] = 1;
    let jobs = blocks.iter().enumerate().fold(vec![], |mut jobs, (i, block)| {
        jobs_of(i, block, &mut jobs);
        jobs
    });
    let item = jobs.len() as u64 + 150;
    for (what, wrong) in [("zero", [0; 9]), ("odd zero", odd_zero), ("failed check", failed_check), ("unknown flag", unknown_flag), ("past p", past_p), ("refused", refused)] {
        let mut faulty = OnDalek::new(&views, 100);
        faulty.fault = Some((item, wrong));
        assert!(keyring.flag_by(&blocks, 1, 0, &mut faulty).is_err(), "{what}");
    }
    // The same item handed back right is taken.
    let mut sound_again = OnDalek::new(&views, 100);
    let mut right = vec![];
    dalek_words(&w.view, &sendable(&jobs[150].key), &mut right);
    sound_again.fault = Some((item, right.try_into().unwrap()));
    assert_eq!(keyring.flag_by(&blocks, 1, 0, &mut sound_again).unwrap().unwrap(), sound);
}

/// Whatever's in a transaction's extra, the keys read are `Extra::keys`' keys.
#[test]
fn encoded_keys_are_the_keys_extra_gives() {
    let mut rng = StdRng::seed_from_u64(7);
    let same = |bytes: &[u8]| {
        let Ok(extra) = Extra::read(&mut &bytes[..]) else { return };
        let identity = CompressedPoint::IDENTITY.to_bytes();
        let cleaned = |key: &[u8; 32]| if CompressedPoint::from(*key).decompress().is_some() { *key } else { identity };
        let ours = encoded_keys(&extra).map(|(keys, more)| (keys.iter().map(cleaned).collect::<Vec<_>>(), more.map(|m| m.iter().map(cleaned).collect::<Vec<_>>())));
        let theirs = extra.keys().map(|(keys, more)| {
            let bytes = |list: Vec<Point>| list.into_iter().map(|k| k.compress().to_bytes()).collect::<Vec<_>>();
            (bytes(keys), more.map(bytes))
        });
        assert_eq!(ours, theirs, "{bytes:?}");
    };
    let key = |rng: &mut StdRng| point_of(&random_scalar(rng));
    for _ in 0..500 {
        // Fields in any order, with nonces and padding about.
        let mut bytes = vec![];
        for _ in 0..rng.gen_range(0..6) {
            match rng.gen_range(0..4) {
                0 => {
                    bytes.push(1);
                    let k = key(&mut rng);
                    bytes.extend_from_slice(&k);
                }
                1 => {
                    let n = rng.gen_range(0..4);
                    bytes.push(4);
                    bytes.push(n);
                    for _ in 0..n {
                        let k = key(&mut rng);
                        bytes.extend_from_slice(&k);
                    }
                }
                2 => {
                    let n = rng.gen_range(0..40);
                    bytes.push(2);
                    bytes.push(n);
                    for _ in 0..n {
                        bytes.push(rng.r#gen());
                    }
                }
                _ => {
                    let mut bad = [0xff; 33];
                    bad[0] = 1;
                    bad[32] = 0x7f;
                    bytes.extend_from_slice(&bad);
                }
            }
        }
        same(&bytes);
    }
    // More keys than a one-byte count holds.
    let mut many = vec![4, 0x90, 0x01];
    for _ in 0..144 {
        let k = key(&mut rng);
        many.extend_from_slice(&k);
    }
    same(&many);
    // Random bytes that happen to parse.
    for _ in 0..3000 {
        let mut bytes = vec![0; rng.gen_range(0..80)];
        rng.fill_bytes(&mut bytes);
        same(&bytes);
    }
}
