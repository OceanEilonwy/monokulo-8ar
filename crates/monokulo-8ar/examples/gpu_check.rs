//! Holds a GPU to curve25519-dalek's answers at length, for development: the
//! crate runs a short form of this (`vectors.rs`) each time the GPU starts.
//!
//!   cargo run --release -p monokulo-8ar --example gpu_check [--any] [--rounds N] [--batch N] [--windows N] [--group N] [--every N]
//!
//! It checks the shader's field arithmetic on the values that break field
//! arithmetic, against plain big integers, then 8aR against curve25519-dalek on random keys, small-order and
//! torsioned points, off-curve keys and awkward view keys, at sizes either side
//! of a workgroup and of a batch. It stops at the first difference.

use curve25519_dalek::{Scalar, constants::ED25519_BASEPOINT_TABLE, constants::EIGHT_TORSION, edwards::{CompressedEdwardsY, EdwardsPoint}};
use crypto_bigint::{Encoding, U256, U512};
use monokulo_8ar::gpu::{Arena, Derived, Device, Digits, TUNING, Tuning};

const P_BIG: U256 = U256::from_be_hex("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed");

/// One field operation, done with big integers: slow and plainly right.
fn field_answer(op: i32, a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
    let big = |limbs: &[u32; 8]| U256::from_le_bytes(to_bytes(*limbs)).wrapping_rem(&P_BIG);
    let (a, b) = (big(a), big(b));
    let times = |x: &U256, y: &U256| {
        let (lo, hi) = x.mul_wide(y);
        hi.concat(&lo).wrapping_rem(&P_BIG.resize::<{ U512::LIMBS }>()).resize::<{ U256::LIMBS }>()
    };
    let answer = match op {
        0 => a.add_mod(&b, &P_BIG),
        1 => a.sub_mod(&b, &P_BIG),
        2 => times(&a, &b),
        3 => times(&a, &a),
        4 => a.neg_mod(&P_BIG),
        _ => a,
    };
    to_limbs(answer.to_le_bytes())
}

/// A small generator with no dependency, so each run's values can be replayed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn limbs(&mut self) -> [u32; 8] {
        std::array::from_fn(|_| self.next() as u32)
    }
    fn scalar(&mut self) -> Scalar {
        let mut bytes = [0; 64];
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        Scalar::from_bytes_mod_order_wide(&bytes)
    }
}

/// Values at and around 0, p, 2p and 2^256, and carry chains across every limb.
fn awkward() -> Vec<[u32; 8]> {
    let p: [u32; 8] = [0xffff_ffed, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0x7fff_ffff];
    let add = |a: [u32; 8], n: i64| {
        let mut out = [0; 8];
        let mut carry = n;
        for i in 0..8 {
            let v = i64::from(a[i]) + carry;
            out[i] = v as u32;
            carry = v >> 32;
        }
        out
    };
    let twice = |a: [u32; 8]| {
        let mut out = [0; 8];
        let mut carry = 0u64;
        for i in 0..8 {
            let v = u64::from(a[i]) * 2 + carry;
            out[i] = v as u32;
            carry = v >> 32;
        }
        out
    };
    let mut values = vec![];
    for n in -40..=40 {
        values.push(add(p, n));
        values.push(add(twice(p), n));
        values.push(add([0; 8], n));
        values.push(add([u32::MAX; 8], n));
        let mut top = [0; 8];
        top[7] = 0x8000_0000;
        values.push(add(top, n));
    }
    for i in 0..8 {
        let mut one = [0; 8];
        one[i] = 1;
        values.push(one);
        one[i] = u32::MAX;
        values.push(one);
        one[i] = 0x8000_0000;
        values.push(one);
        let mut low: [u32; 8] = [u32::MAX; 8];
        low[i] = 0;
        values.push(low);
        // Every limb below i full: a carry that runs.
        let mut run = [0; 8];
        run[..i].fill(u32::MAX);
        values.push(run);
    }
    values
}

fn to_limbs(bytes: [u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
}

fn field(gpu: &Arena, rng: &mut Rng) -> u64 {
    let values = awkward();
    let mut pairs = vec![];
    for a in &values {
        for b in &values {
            pairs.push((*a, *b));
        }
    }
    for _ in 0..100_000 {
        pairs.push((rng.limbs(), rng.limbs()));
    }
    let mut checked = 0;
    for op in 0..6 {
        let mut at = 0;
        for chunk in pairs.chunks(gpu.tuning().batch) {
            let got = gpu.check_field(op as u32, chunk).expect("the GPU ran");
            for (i, (a, b)) in chunk.iter().enumerate() {
                assert_eq!(got[i], field_answer(op, a, b), "field op {op} at {} on {a:x?} {b:x?}", at + i);
                checked += 1;
            }
            at += chunk.len();
        }
    }
    checked
}

enum Want {
    Point([u8; 32]),
    OffCurve,
}

fn compare(gpu: &Arena, view: &Scalar, keys: &[[u8; 32]], wants: &[Want], what: &str) {
    let digits = Digits::new(&view.to_bytes());
    let got = gpu.derive(&[&digits], keys).expect("the GPU ran");
    assert_eq!(got.len(), keys.len());
    for (i, want) in wants.iter().enumerate() {
        match (got.derived(i), want) {
            (Derived::Point(bytes), Want::Point(want)) => assert_eq!(&bytes, want, "{what}: key {i} of {}: {:x?}", keys.len(), keys[i]),
            (Derived::OffCurve, Want::OffCurve) => {}
            (Derived::Point(_), Want::OffCurve) => panic!("{what}: key {i} is off the curve, and the GPU took it"),
            (Derived::OffCurve, Want::Point(_)) => panic!("{what}: key {i} is on the curve, and the GPU refused it"),
            (Derived::Wrong, _) => panic!("{what}: key {i}'s answer failed the checks on it"),
        }
    }
}

fn expected(view: &Scalar, key: &[u8; 32]) -> Want {
    match CompressedEdwardsY(*key).decompress() {
        Some(point) => Want::Point((view * point).mul_by_cofactor().compress().to_bytes()),
        None => Want::OffCurve,
    }
}

fn point(rng: &mut Rng) -> EdwardsPoint {
    &rng.scalar() * ED25519_BASEPOINT_TABLE
}

fn view_keys(rng: &mut Rng) -> Vec<Scalar> {
    let most = Scalar::ZERO - Scalar::ONE;
    // All digits at the extremes of the signed window: 8 is the seam in the recoding.
    let eights = Scalar::from_bytes_mod_order([0x88; 32]);
    let sevens = Scalar::from_bytes_mod_order([0x77; 32]);
    let nines = Scalar::from_bytes_mod_order([0x99; 32]);
    let mut keys = vec![Scalar::ZERO, Scalar::ONE, Scalar::from(8u8), Scalar::from(16u8), most, most - Scalar::ONE, eights, sevens, nines];
    keys.push(Scalar::from_bytes_mod_order([0xff; 32]));
    for _ in 0..4 {
        keys.push(rng.scalar());
    }
    keys
}

fn keys_of_size(rng: &mut Rng, count: usize) -> Vec<[u8; 32]> {
    (0..count)
        .map(|i| match i % 37 {
            // Torsioned, every eighth kind in turn.
            3 => (point(rng) + EIGHT_TORSION[i / 37 % 8]).compress().to_bytes(),
            // Small order alone.
            7 => EIGHT_TORSION[i / 37 % 8].compress().to_bytes(),
            // Not, most likely, on the curve: a random y below p.
            11 => {
                let mut bytes = to_bytes(rng.limbs());
                bytes[31] &= 0x3f;
                bytes
            }
            _ => point(rng).compress().to_bytes(),
        })
        .collect()
}

fn to_bytes(limbs: [u32; 8]) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (i, limb) in limbs.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter().position(|a| a == name).map_or(default, |i| args[i + 1].parse().expect("a number"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tuning = Tuning {
        batch: arg(&args, "--batch", TUNING.batch),
        windows: arg(&args, "--windows", TUNING.windows),
        submit_every: arg(&args, "--every", TUNING.submit_every),
        group: arg(&args, "--group", TUNING.group as usize) as u32,
    };
    let rounds = arg(&args, "--rounds", 3);
    let started = std::time::Instant::now();
    let device = Device::start(None, tuning, args.iter().any(|a| a == "--any")).expect("a GPU");
    let gpu = Arena::new(&device).expect("an arena");
    println!("{}, {tuning:?}, started in {:.1} s", gpu.name(), started.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    println!("the startup vectors: {:?}, {:.3} s", monokulo_8ar::startup_vectors(&gpu), t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let view = zeroize::Zeroizing::new(curve25519_dalek::Scalar::from_bytes_mod_order([0x42; 32]).to_bytes());
    println!("a wallet's vectors: {:?}, {:.3} s", monokulo_8ar::wallet_vectors(&gpu, &view), t.elapsed().as_secs_f64());
    let mut rng = Rng(u64::from(std::process::id()));
    println!("seed {}", rng.0);

    let t = std::time::Instant::now();
    println!("field: {} operations agree in {:.1} s", field(&gpu, &mut rng), t.elapsed().as_secs_f64());

    let group = tuning.group as usize;
    let sizes = [1, 2, 3, group - 1, group, group + 1, 2 * group - 1, 2 * group, 2 * group + 1, 1000, tuning.batch - 1, tuning.batch];
    let mut compared = 0u64;
    for round in 0..rounds {
        let t = std::time::Instant::now();
        for view in view_keys(&mut rng) {
            for &size in sizes.iter().filter(|s| round == 0 || **s < 2000) {
                let keys = keys_of_size(&mut rng, size);
                let wants: Vec<Want> = keys.iter().map(|k| expected(&view, k)).collect();
                compare(&gpu, &view, &keys, &wants, &format!("size {size}"));
                compared += size as u64;
            }
        }
        // A full batch of ordinary keys on a random view key.
        let view = rng.scalar();
        let keys = keys_of_size(&mut rng, tuning.batch);
        let wants: Vec<Want> = keys.iter().map(|k| expected(&view, k)).collect();
        compare(&gpu, &view, &keys, &wants, "full batch");
        compared += tuning.batch as u64;
        println!("round {round}: {compared} keys agree so far, this round {:.1} s", t.elapsed().as_secs_f64());
    }
    println!("all agree: {compared} keys");
}
