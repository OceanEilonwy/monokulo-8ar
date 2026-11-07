//! What a GPU must get right before it's used. [`passes`] runs once per process,
//! before anything else: field arithmetic on the values that break it, against
//! plain big integers, then 8aR on keys chosen to go wrong, against
//! curve25519-dalek's answers. [`passes_for_wallet`] runs the same keys with a
//! wallet's own view key, and that wallet keeps the result.
//! `examples/gpu_check.rs` is the long version, for development.

use crypto_bigint::{Encoding, U256, U512};
use curve25519_dalek::{Scalar, constants::{ED25519_BASEPOINT_TABLE, EIGHT_TORSION}, edwards::CompressedEdwardsY};
use zeroize::{Zeroize, Zeroizing};

use crate::gpu::{Digits, Gpu, GpuError, Points};

const P: [u32; 8] = [0xffff_ffed, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0x7fff_ffff];

fn plus(a: [u32; 8], n: i64) -> [u32; 8] {
    let mut out = [0; 8];
    let mut carry = n;
    for i in 0..8 {
        let v = i64::from(a[i]) + carry;
        out[i] = v as u32;
        carry = v >> 32;
    }
    out
}

/// Values at 0, p, 2^255 and 2^256, and carries that run through every limb.
fn field_values() -> Vec<[u32; 8]> {
    let top = [0, 0, 0, 0, 0, 0, 0, 0x8000_0000];
    let mut values = vec![];
    for n in [-39, -38, -20, -19, -1, 0, 1, 18, 19, 38] {
        values.extend([plus(P, n), plus(top, n), plus([0; 8], n), plus([u32::MAX; 8], n)]);
    }
    for i in 1..8 {
        let mut run = [0; 8];
        run[..i].fill(u32::MAX);
        values.push(run);
    }
    values
}

/// p, for the field answers: worked out with plain big integers, which are slow
/// and obviously right, rather than with curve code of our own.
const P_BIG: U256 = U256::from_be_hex("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed");

fn big(limbs: &[u32; 8]) -> U256 {
    let mut bytes = [0; 32];
    for (limb, out) in limbs.iter().zip(bytes.as_chunks_mut::<4>().0) {
        *out = limb.to_le_bytes();
    }
    U256::from_le_bytes(bytes).wrapping_rem(&P_BIG)
}

fn field_answer(op: i32, a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
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
    let bytes = answer.to_le_bytes();
    std::array::from_fn(|i| u32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]))
}

pub(crate) struct Case {
    view: Scalar,
    keys: Vec<[u8; 32]>,
}

/// A view key and keys for it: ordinary, torsioned, of small order and off the curve.
pub(crate) fn cases(group: usize) -> Vec<Case> {
    let most = Scalar::ZERO - Scalar::ONE;
    let views = [
        Scalar::ZERO,
        Scalar::ONE,
        most,
        // Digits all at the seam of the signed window, and either side of it.
        Scalar::from_bytes_mod_order([0x88; 32]),
        Scalar::from_bytes_mod_order([0x77; 32]),
        Scalar::from_bytes_mod_order([0x99; 32]),
        Scalar::from_bytes_mod_order([0x5a; 32]),
    ];
    // Each dispatch costs tens of milliseconds on a phone, however few keys: every
    // view key gets one good size, and the sizes around a workgroup one view key.
    let sizes = [1, group - 1, group, group + 1];
    let runs = views.iter().map(|view| (*view, 2 * group + 1)).chain(sizes.map(|size| (views[6], size)));
    let mut seed = first_seed();
    runs.map(|(view, size)| Case { view, keys: keys(&mut seed, size) }).collect()
}

fn first_seed() -> Scalar {
    Scalar::from(0x0123_4567_89ab_cdefu64)
}

/// `size` keys, one in eleven of each awkward kind, going on from `seed`.
fn keys(seed: &mut Scalar, size: usize) -> Vec<[u8; 32]> {
    (0..size)
        .map(|i| {
            *seed = *seed * *seed + Scalar::ONE;
            let point = &*seed * ED25519_BASEPOINT_TABLE;
            match i % 11 {
                3 => (point + EIGHT_TORSION[i / 11 % 8]).compress().to_bytes(),
                5 => EIGHT_TORSION[i / 11 % 8].compress().to_bytes(),
                7 => {
                    // A y that's mostly off the curve; dalek says which.
                    let mut y = seed.to_bytes();
                    y[31] &= 0x3f;
                    y
                }
                _ => point.compress().to_bytes(),
            }
        })
        .collect()
}

/// A wallet's view key, with keys that cover a workgroup's edges and its overflow.
fn wallet_case(view: &[u8; 32], group: usize) -> Case {
    Case { view: Scalar::from_bytes_mod_order(*view), keys: keys(&mut first_seed(), 2 * group + 1) }
}

/// Whether `points` are `case`'s answers, as curve25519-dalek gives them.
pub(crate) fn agrees(case: &Case, points: &Points) -> bool {
    if points.len() != case.keys.len() {
        return false;
    }
    case.keys.iter().enumerate().all(|(i, key)| match (CompressedEdwardsY(*key).decompress(), points.get(i)) {
        (Some(point), Some(got)) => Zeroizing::new(got) == Zeroizing::new((case.view * point).mul_by_cofactor().compress().to_bytes()),
        (None, None) => true,
        _ => false,
    })
}

/// Runs the vectors on `gpu`: `Ok(false)` for a wrong answer.
pub(crate) fn passes(gpu: &Gpu) -> Result<bool, GpuError> {
    let values = field_values();
    let pairs: Vec<([u32; 8], [u32; 8])> = values.iter().flat_map(|a| values.iter().map(move |b| (*a, *b))).collect();
    for op in 0..6 {
        let got = gpu.check_field(op, &pairs)?;
        if pairs.iter().zip(&got).any(|((a, b), got)| field_answer(op, a, b) != *got) {
            return Ok(false);
        }
    }
    for case in cases(gpu.tuning().group as usize) {
        let digits = Digits::new(&case.view.to_bytes());
        if !agrees(&case, &gpu.derive(&digits, &case.keys)?) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Runs a wallet's own view key on `gpu`, the same way scanning will: `Ok(false)`
/// for a wrong answer.
pub(crate) fn passes_for_wallet(gpu: &Gpu, view: &Zeroizing<[u8; 32]>) -> Result<bool, GpuError> {
    let mut case = wallet_case(view, gpu.tuning().group as usize);
    let digits = Digits::new(view);
    let result = gpu.derive(&digits, &case.keys).map(|points| agrees(&case, &points));
    case.view.zeroize();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Answers from curve25519-dalek, as a sound GPU would give them.
    fn sound_points(case: &Case) -> Vec<u32> {
        let mut words = vec![];
        for key in &case.keys {
            match CompressedEdwardsY(*key).decompress() {
                Some(r) => {
                    let bytes = (case.view * r).mul_by_cofactor().compress().to_bytes();
                    words.extend((0..8).map(|i| u32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]])));
                    words.push(0);
                }
                None => {
                    words.extend([0; 8]);
                    words.push(1);
                }
            }
        }
        words
    }

    /// The big-integer answers, on values whose answers are known.
    #[test]
    fn the_field_answers_are_right() {
        let limbs = |v: u64| -> [u32; 8] { [v as u32, (v >> 32) as u32, 0, 0, 0, 0, 0, 0] };
        let one = limbs(1);
        let p_minus_one = plus(P, -1);
        assert_eq!(field_answer(2, &p_minus_one, &p_minus_one), one, "(-1)(-1) is 1");
        assert_eq!(field_answer(5, &[0, 0, 0, 0, 0, 0, 0, 0x8000_0000], &one), limbs(19), "2^255 is 19");
        assert_eq!(field_answer(5, &P, &one), limbs(0), "p is 0");
        assert_eq!(field_answer(0, &p_minus_one, &limbs(2)), one);
        assert_eq!(field_answer(1, &limbs(0), &one), p_minus_one);
        assert_eq!(field_answer(4, &one, &one), p_minus_one);
        assert_eq!(field_answer(3, &limbs(1 << 20), &one), limbs(1 << 40));
        assert_eq!(field_answer(2, &limbs(u32::MAX as u64), &limbs(u32::MAX as u64)), limbs((u32::MAX as u64) * (u32::MAX as u64)));
        // 2^128 squared is 2^256, which is 38.
        let mut top = [0; 8];
        top[4] = 1;
        assert_eq!(field_answer(3, &top, &one), limbs(38));
    }

    /// A batch runs in what the last one left: nothing of it may show in the answers.
    #[test]
    #[ignore = "needs a GPU; run with --ignored"]
    fn batches_in_turn_get_their_own_answers() {
        let gpu = Gpu::start_for_device(None, false).expect("a GPU");
        let (batch, group) = (gpu.tuning().batch, gpu.tuning().group as usize);
        let view = Scalar::from_bytes_mod_order([0x5a; 32]);
        let digits = Digits::new(&view.to_bytes());
        let mut seed = first_seed();
        for size in [batch, 1, 3 * group + 1, batch - 1, group - 1, 2, batch / 2 + 1] {
            let case = Case { view, keys: keys(&mut seed, size) };
            assert!(agrees(&case, &gpu.derive(&digits, &case.keys).expect("the GPU ran")), "{size} keys");
        }
        gpu.release();
        let case = Case { view, keys: keys(&mut seed, 2 * group + 1) };
        assert!(agrees(&case, &gpu.derive(&digits, &case.keys).expect("the GPU ran")), "after resting");
    }

    #[test]
    fn the_vectors_cover_what_they_should() {
        let cases = cases(64);
        assert_eq!(cases.len(), 11);
        let keys = cases.iter().flat_map(|c| &c.keys);
        let off_curve = keys.clone().filter(|k| CompressedEdwardsY(**k).decompress().is_none()).count();
        assert!(off_curve > 10, "{off_curve} keys off the curve");
    }

    #[test]
    fn a_wallets_case_has_its_view_key_and_the_awkward_keys() {
        let view = Scalar::from_bytes_mod_order([0x42; 32]);
        let case = wallet_case(&view.to_bytes(), 64);
        assert_eq!(case.view, view);
        assert_eq!(case.keys.len(), 129);
        assert!(case.keys.iter().any(|k| CompressedEdwardsY(*k).decompress().is_none()), "a key off the curve");
    }

    #[test]
    fn sound_answers_pass_and_any_wrong_one_fails() {
        let wallet = Scalar::from_bytes_mod_order([0x42; 32]).to_bytes();
        for case in cases(16).into_iter().chain([wallet_case(&wallet, 16)]) {
            let words = sound_points(&case);
            assert!(agrees(&case, &Points::from_words(words.clone())));
            // One bit wrong anywhere in a key on the curve, the sign included.
            let Some(i) = case.keys.iter().position(|k| CompressedEdwardsY(*k).decompress().is_some()) else { continue };
            for (word, bit) in [(0, 0), (3, 17), (7, 31)] {
                let mut wrong = words.clone();
                wrong[i * 9 + word] ^= 1 << bit;
                assert!(!agrees(&case, &Points::from_words(wrong)), "a changed point");
            }
            // A key's flag the wrong way round.
            let mut wrong = words.clone();
            wrong[i * 9 + 8] = 1;
            assert!(!agrees(&case, &Points::from_words(wrong)), "a refused key");
            // Too few answers.
            assert!(!agrees(&case, &Points::from_words(words[..words.len() - 9].to_vec())));
        }
    }
}
