//! Curve25519 arithmetic written for one job: 8aR for a fixed secret `a` and many
//! public R, ending in a batched inversion. Radix 2^51, as curve25519-dalek's
//! 64-bit backend. The table lookup is constant-time, as `a` is the view key;
//! R is public, so decompressing it needn't be.
//!
//! The tests hold every result to curve25519-dalek's.

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::Zeroize;

const LOW_51: u64 = (1 << 51) - 1;

#[derive(Clone, Copy)]
pub struct Fe(pub [u64; 5]);

const D: Fe = Fe([929955233495203, 466365720129213, 1662059464998953, 2033849074728123, 1442794654840575]);
const D2: Fe = Fe([1859910466990425, 932731440258426, 1072319116312658, 1815898335770999, 633789495995903]);
const SQRT_M1: Fe = Fe([1718705420411056, 234908883556509, 2233514472574048, 2117202627021982, 765476049583133]);
const ZERO: Fe = Fe([0; 5]);
const ONE: Fe = Fe([1, 0, 0, 0, 0]);

#[inline(always)]
fn m(a: u64, b: u64) -> u128 {
    a as u128 * b as u128
}

#[inline(always)]
fn carry(mut c: [u128; 5]) -> Fe {
    c[1] += (c[0] >> 51) as u64 as u128;
    let mut out = [c[0] as u64 & LOW_51, 0, 0, 0, 0];
    c[2] += (c[1] >> 51) as u64 as u128;
    out[1] = c[1] as u64 & LOW_51;
    c[3] += (c[2] >> 51) as u64 as u128;
    out[2] = c[2] as u64 & LOW_51;
    c[4] += (c[3] >> 51) as u64 as u128;
    out[3] = c[3] as u64 & LOW_51;
    let top = (c[4] >> 51) as u64;
    out[4] = c[4] as u64 & LOW_51;
    out[0] += top * 19;
    out[1] += out[0] >> 51;
    out[0] &= LOW_51;
    Fe(out)
}

impl Fe {
    #[inline(always)]
    pub fn add(&self, b: &Fe) -> Fe {
        Fe([self.0[0] + b.0[0], self.0[1] + b.0[1], self.0[2] + b.0[2], self.0[3] + b.0[3], self.0[4] + b.0[4]])
    }

    #[inline(always)]
    pub fn sub(&self, b: &Fe) -> Fe {
        // a + 16p - b, then a light reduction.
        let (a, b) = (&self.0, &b.0);
        let l = [
            a[0] + 36028797018963664 - b[0],
            a[1] + 36028797018963952 - b[1],
            a[2] + 36028797018963952 - b[2],
            a[3] + 36028797018963952 - b[3],
            a[4] + 36028797018963952 - b[4],
        ];
        let c = [l[0] >> 51, l[1] >> 51, l[2] >> 51, l[3] >> 51, l[4] >> 51];
        Fe([(l[0] & LOW_51) + c[4] * 19, (l[1] & LOW_51) + c[0], (l[2] & LOW_51) + c[1], (l[3] & LOW_51) + c[2], (l[4] & LOW_51) + c[3]])
    }

    #[inline(always)]
    pub fn neg(&self) -> Fe {
        ZERO.sub(self)
    }

    #[inline(always)]
    pub fn mul(&self, b: &Fe) -> Fe {
        let (a, b) = (&self.0, &b.0);
        let (b1, b2, b3, b4) = (b[1] * 19, b[2] * 19, b[3] * 19, b[4] * 19);
        carry([
            m(a[0], b[0]) + m(a[4], b1) + m(a[3], b2) + m(a[2], b3) + m(a[1], b4),
            m(a[1], b[0]) + m(a[0], b[1]) + m(a[4], b2) + m(a[3], b3) + m(a[2], b4),
            m(a[2], b[0]) + m(a[1], b[1]) + m(a[0], b[2]) + m(a[4], b3) + m(a[3], b4),
            m(a[3], b[0]) + m(a[2], b[1]) + m(a[1], b[2]) + m(a[0], b[3]) + m(a[4], b4),
            m(a[4], b[0]) + m(a[3], b[1]) + m(a[2], b[2]) + m(a[1], b[3]) + m(a[0], b[4]),
        ])
    }

    #[inline(always)]
    pub fn sq(&self) -> Fe {
        let a = &self.0;
        let (d0, d1, d2) = (a[0] * 2, a[1] * 2, a[2] * 2);
        let (a3_19, a4_19) = (a[3] * 19, a[4] * 19);
        carry([
            m(a[0], a[0]) + m(d1, a4_19) + m(d2, a3_19),
            m(d0, a[1]) + m(d2, a4_19) + m(a[3], a3_19),
            m(d0, a[2]) + m(a[1], a[1]) + m(a[3] * 2, a4_19),
            m(d0, a[3]) + m(d1, a[2]) + m(a[4], a4_19),
            m(d0, a[4]) + m(d1, a[3]) + m(a[2], a[2]),
        ])
    }

    fn sqn(&self, n: u32) -> Fe {
        let mut x = *self;
        for _ in 0..n {
            x = x.sq();
        }
        x
    }

    /// (self^(2^250-1), self^11)
    fn pow250(&self) -> (Fe, Fe) {
        let t0 = self.sq();
        let t1 = t0.sqn(2);
        let t2 = self.mul(&t1);
        let t3 = t0.mul(&t2);
        let t4 = t3.sq();
        let t5 = t2.mul(&t4);
        let t6 = t5.sqn(5);
        let t7 = t6.mul(&t5);
        let t8 = t7.sqn(10);
        let t9 = t8.mul(&t7);
        let t10 = t9.sqn(20);
        let t11 = t10.mul(&t9);
        let t12 = t11.sqn(10);
        let t13 = t12.mul(&t7);
        let t14 = t13.sqn(50);
        let t15 = t14.mul(&t13);
        let t16 = t15.sqn(100);
        let t17 = t16.mul(&t15);
        let t18 = t17.sqn(50);
        (t18.mul(&t13), t3)
    }

    pub fn invert(&self) -> Fe {
        let (t, eleven) = self.pow250();
        t.sqn(5).mul(&eleven)
    }

    /// self^((p-5)/8)
    fn pow_p58(&self) -> Fe {
        let (t, _) = self.pow250();
        t.sqn(2).mul(self)
    }

    pub fn from_bytes(b: &[u8; 32]) -> Fe {
        let w = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        Fe([w(0) & LOW_51, (w(6) >> 3) & LOW_51, (w(12) >> 6) & LOW_51, (w(19) >> 1) & LOW_51, (w(24) >> 12) & LOW_51])
    }

    /// From eight little-endian 32-bit limbs of any value under 2^256.
    pub fn from_limbs(limbs: &[u32]) -> Fe {
        let mut bytes = [0; 32];
        for (i, limb) in limbs.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&limb.to_le_bytes());
        }
        let mut fe = Fe::from_bytes(&bytes);
        // 2^255 is 19.
        fe.0[0] += u64::from(bytes[31] >> 7) * 19;
        fe
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let mut l = self.0;
        // Bring to under 2p, find the quotient, take it off.
        let c = [l[0] >> 51, l[1] >> 51, l[2] >> 51, l[3] >> 51, l[4] >> 51];
        l = [(l[0] & LOW_51) + c[4] * 19, (l[1] & LOW_51) + c[0], (l[2] & LOW_51) + c[1], (l[3] & LOW_51) + c[2], (l[4] & LOW_51) + c[3]];
        let mut q = (l[0] + 19) >> 51;
        q = (l[1] + q) >> 51;
        q = (l[2] + q) >> 51;
        q = (l[3] + q) >> 51;
        q = (l[4] + q) >> 51;
        l[0] += 19 * q;
        l[1] += l[0] >> 51;
        l[0] &= LOW_51;
        l[2] += l[1] >> 51;
        l[1] &= LOW_51;
        l[3] += l[2] >> 51;
        l[2] &= LOW_51;
        l[4] += l[3] >> 51;
        l[3] &= LOW_51;
        l[4] &= LOW_51;
        let mut s = [0u8; 32];
        let v = (l[0] as u128) | ((l[1] as u128) << 51) | ((l[2] as u128) << 102);
        s[0..16].copy_from_slice(&v.to_le_bytes());
        let v = ((l[2] >> 26) as u128) | ((l[3] as u128) << 25) | ((l[4] as u128) << 76);
        s[16..32].copy_from_slice(&v.to_le_bytes());
        s
    }

    fn select(a: &Fe, b: &Fe, choice: Choice) -> Fe {
        Fe(std::array::from_fn(|i| u64::conditional_select(&a.0[i], &b.0[i], choice)))
    }
}

/// Whether `bytes` encodes y below p, and isn't a negative zero: x is zero only
/// for y = 1 and y = -1. Whether it's on the curve is another matter.
pub fn canonical(bytes: &[u8; 32]) -> bool {
    let mut y = *bytes;
    let sign = y[31] >> 7;
    y[31] &= 0x7f;
    let fe = Fe::from_bytes(&y);
    if fe.to_bytes() != y {
        return false;
    }
    let one = ONE.to_bytes();
    let minus_one = ONE.neg().to_bytes();
    !(sign == 1 && (y == one || y == minus_one))
}

#[derive(Clone, Copy)]
pub struct Ext {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

#[derive(Clone, Copy)]
struct Proj {
    x: Fe,
    y: Fe,
    z: Fe,
}

#[derive(Clone, Copy)]
struct Comp {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

#[derive(Clone, Copy)]
struct Niels {
    yp: Fe,
    ym: Fe,
    z: Fe,
    t2d: Fe,
}

impl Comp {
    #[inline(always)]
    fn proj(&self) -> Proj {
        Proj { x: self.x.mul(&self.t), y: self.y.mul(&self.z), z: self.z.mul(&self.t) }
    }
    #[inline(always)]
    fn ext(&self) -> Ext {
        Ext { x: self.x.mul(&self.t), y: self.y.mul(&self.z), z: self.z.mul(&self.t), t: self.x.mul(&self.y) }
    }
}

impl Proj {
    #[inline(always)]
    fn double(&self) -> Comp {
        let xx = self.x.sq();
        let yy = self.y.sq();
        let zz = self.z.sq();
        let zz2 = zz.add(&zz);
        let xy = self.x.add(&self.y).sq();
        let sum = yy.add(&xx);
        let diff = yy.sub(&xx);
        Comp { x: xy.sub(&sum), y: sum, z: diff, t: zz2.sub(&diff) }
    }
}

impl Ext {
    /// R from its compressed form, taking what `CompressedPoint::decompress`
    /// takes: no unreduced field element, and no negative zero.
    pub fn decompress(bytes: &[u8; 32]) -> Option<Ext> {
        let y = Fe::from_bytes(bytes);
        if !canonical(bytes) {
            return None;
        }
        let yy = y.sq();
        let u = yy.sub(&ONE);
        let v = yy.mul(&D).add(&ONE);
        let v3 = v.sq().mul(&v);
        let v7 = v3.sq().mul(&v);
        let mut x = u.mul(&v3).mul(&u.mul(&v7).pow_p58());
        let check = v.mul(&x.sq()).to_bytes();
        if check == u.neg().to_bytes() {
            x = x.mul(&SQRT_M1);
        } else if check != u.to_bytes() {
            return None;
        }
        if (x.to_bytes()[0] & 1) != (bytes[31] >> 7) {
            x = x.neg();
        }
        Some(Ext { x, y, z: ONE, t: x.mul(&y) })
    }

    #[inline(always)]
    fn niels(&self) -> Niels {
        Niels { yp: self.y.add(&self.x), ym: self.y.sub(&self.x), z: self.z, t2d: self.t.mul(&D2) }
    }

    #[inline(always)]
    fn add(&self, n: &Niels) -> Comp {
        let pp = self.y.add(&self.x).mul(&n.yp);
        let mm = self.y.sub(&self.x).mul(&n.ym);
        let tt = self.t.mul(&n.t2d);
        let zz = self.z.mul(&n.z);
        let zz2 = zz.add(&zz);
        Comp { x: pp.sub(&mm), y: pp.add(&mm), z: zz2.add(&tt), t: zz2.sub(&tt) }
    }

    #[inline(always)]
    fn proj(&self) -> Proj {
        Proj { x: self.x, y: self.y, z: self.z }
    }
}

/// A secret scalar as signed base-16 digits, worked out once.
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

#[inline(always)]
fn select(table: &[Niels; 8], digit: i8) -> Niels {
    let sign = (digit as u8) >> 7;
    let abs = ((digit as i16) ^ -(sign as i16)) as i8 as i16 + sign as i16;
    let mut out = Niels { yp: ONE, ym: ONE, z: ONE, t2d: ZERO };
    for (i, entry) in table.iter().enumerate() {
        let hit = (abs as u16).ct_eq(&(i as u16 + 1));
        out.yp = Fe::select(&out.yp, &entry.yp, hit);
        out.ym = Fe::select(&out.ym, &entry.ym, hit);
        out.z = Fe::select(&out.z, &entry.z, hit);
        out.t2d = Fe::select(&out.t2d, &entry.t2d, hit);
    }
    let negative = Choice::from(sign);
    let t2d_neg = out.t2d.neg();
    Niels {
        yp: Fe::select(&out.yp, &out.ym, negative),
        ym: Fe::select(&out.ym, &out.yp, negative),
        z: out.z,
        t2d: Fe::select(&out.t2d, &t2d_neg, negative),
    }
}

/// 8aR in projective form: (X, Y, Z), to compress with a batch's inversion.
#[inline]
pub fn mul8(digits: &Digits, r: &Ext) -> (Fe, Fe, Fe) {
    let base = r.niels();
    let mut table = [base; 8];
    let mut p = *r;
    for entry in table.iter_mut().skip(1) {
        p = p.add(&base).ext();
        *entry = p.niels();
    }
    let identity = Ext { x: ZERO, y: ONE, z: ONE, t: ZERO };
    let mut q = identity.add(&select(&table, digits.0[63])).ext();
    for i in (0..63).rev() {
        let mut s = q.proj();
        for _ in 0..3 {
            s = s.double().proj();
        }
        q = s.double().ext();
        q = q.add(&select(&table, digits.0[i])).ext();
    }
    let mut s = q.proj();
    for _ in 0..3 {
        s = s.double().proj();
    }
    (s.x, s.y, s.z)
}

/// Compresses `points`, sharing one inversion across them.
pub fn compress_batch(points: &[(Fe, Fe, Fe)], out: &mut Vec<[u8; 32]>) {
    let mut running = Vec::with_capacity(points.len());
    let mut acc = ONE;
    for p in points {
        running.push(acc);
        acc = acc.mul(&p.2);
    }
    let mut inv = acc.invert();
    let start = out.len();
    out.resize(start + points.len(), [0; 32]);
    for i in (0..points.len()).rev() {
        let zi = inv.mul(&running[i]);
        inv = inv.mul(&points[i].2);
        let x = points[i].0.mul(&zi);
        let y = points[i].1.mul(&zi);
        let mut s = y.to_bytes();
        s[31] ^= (x.to_bytes()[0] & 1) << 7;
        out[start + i] = s;
    }
}
