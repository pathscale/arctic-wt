# SVE on Apple silicon: a 34x loss, measured

There once was a CPU extension called SVE. It went out to sea on an M4, and it
sank.

This exists so nobody spends another evening on it. The idea is good, it will
occur to everyone who reads `node_47`, and it is wrong on this hardware for a
reason that is not obvious until you measure it.

## The idea

`node_47` holds up to 48 sorted key bytes, and searching it is the hot operation
in the tree: find the index of the first key greater than or equal to a target.
Neon is 128 bits, so that is four 16-byte compares.

This machine says:

```
$ sysctl -a | grep -E 'FEAT_SME|svl'
hw.optional.arm.FEAT_SME: 1
hw.optional.arm.FEAT_SME2: 1
hw.optional.arm.sme_max_svl_b: 64
```

`sme_max_svl_b: 64` is a **512-bit streaming vector length**. Forty-eight bytes
fits in 512 bits with room to spare. Four instructions become one. It is the
most obvious win in the crate.

## The measurement

Two million searches of 48 keys, on an M4 Max, all three implementations
checked against each other on every one of the 256 possible target bytes before
timing:

| implementation | ns per search | against Neon |
|---|---:|---:|
| scalar | 3.64 | |
| **neon, 4x128-bit** | **0.93** | 3.90x faster than scalar |
| sve 512-bit, `smstart` once for the whole run | 31.62 | **0.03x** |
| sve 512-bit, `smstart` per call | 43.44 | 0.02x |

**Read the third row before blaming the mode switch.** That run enters streaming
mode once, for all two million iterations, so `SMSTART` is amortised to nothing.
It is still thirty-four times slower than Neon. Paying the switch on every call
only moves it from 31.6 to 43.4 ns. The instructions themselves are slow here.

## Why

Apple built an SME unit to feed a ZA tile with matrix outer products, for
machine learning. Streaming SVE exists on it because the architecture requires
it of any SME implementation, not because there is a fast general-purpose
512-bit vector engine underneath. Neon is the fast path on this chip; the SME
unit is a coprocessor with entirely different economics.

The chip's other vector-adjacent features point the same way. Beyond Neon it
reports `FEAT_DotProd`, `FEAT_BF16`, `FEAT_I8MM`, `FEAT_SME_F64F64`,
`FEAT_SME_I16I64`. Every one of those is a machine-learning instruction. None of
them helps compare sixteen bytes.

## Three further walls

Any one of these is fatal on its own, so do not go looking for a toolchain fix.

**Rust.** SVE intrinsics are nightly-only, behind `stdarch_aarch64_sve`, and
blocked on an unaccepted RFC for types whose size is not known at compile time,
tracked at [rust-lang/rust#145052](https://github.com/rust-lang/rust/issues/145052).
They merged into stdarch in [April 2026](https://github.com/rust-lang/stdarch/pull/2071)
and are still gated. Verified on three stable releases published since:

```
1.96.0 (2026-05-25)  error[E0554]
1.97.1 (2026-07-14)  error[E0554]
1.98.0 (2026-08-18)  error[E0554]
nightly (2026-08-30) compiles
```

Merging an intrinsic into stdarch is not stabilising it. The code ships in the
stable standard library and the gate refuses to let stable use it: ask for
`svcntb` on 1.98 without the feature line and the error is `E0658: use of
unstable library feature`, not "cannot find function". It is there, switched
off.

**Apple.** There is no non-streaming SVE. `-C target-feature=+sve` compiles
cleanly, emits no SVE instructions into this crate, and produces a binary that
dies with **SIGILL, exit 132**, because the flag licenses LLVM to use SVE
anywhere including in code it inserts itself. Telling the compiler the CPU has a
feature it does not have is not a tuning knob.

**fearless_simd.** Scalable vectors are refused by design, on
[issue #339](https://github.com/linebender/fearless_simd/issues/339), closed the
day it was opened:

> This is a deliberate design choice. Having N as a compile-time constant helps
> with optimization quite a lot. Making it a runtime value would hurt
> performance on common devices.

The whole API rests on `SimdBase` having `const N: usize`. There is no level to
dispatch an SVE path through, and there will not be one.

## What this does not say

This is one operation, one implementation, one chip. A different SVE
formulation might close some of the gap, though not a 34x one. More importantly
**none of it transfers to hardware with real non-streaming SVE**: a Graviton or
a Grace has a vector unit built to be a vector unit, and the answer there could
easily reverse. If Arctic ever runs on one, measure again rather than citing
this.

It also says nothing about SME used as intended. Matrix work on the ZA tile is
what that unit is good at. It is simply not what a radix tree does.

## The harness

Nightly, because of the feature gate. `svcntb()` inside streaming mode returns
64 on this machine, which is how you confirm 512-bit vectors are actually live
before trusting any timing.

```rust
#![feature(stdarch_aarch64_sve)]
use std::arch::aarch64::*;
use std::arch::asm;
use std::hint::black_box;
use std::time::Instant;

const KEYS: usize = 48;

fn scalar(keys: &[u8; KEYS], target: u8, len: usize) -> u32 {
    for (i, k) in keys[..len].iter().enumerate() {
        if *k >= target {
            return i as u32;
        }
    }
    len as u32
}

/// Four 16-byte compares, which is what the Neon path amounts to.
#[target_feature(enable = "neon")]
unsafe fn neon(keys: &[u8; KEYS], target: u8, len: usize) -> u32 { unsafe {
    let t = vdupq_n_u8(target);
    let mut base = 0usize;
    while base < len {
        let v = vld1q_u8(keys.as_ptr().add(base));
        let ge = vcgeq_u8(v, t);
        let narrowed = vshrn_n_u16(vreinterpretq_u16_u8(ge), 4);
        let bits = vget_lane_u64(vreinterpret_u64_u8(narrowed), 0);
        if bits != 0 {
            let lane = (bits.trailing_zeros() >> 2) as usize;
            let idx = base + lane;
            return if idx < len { idx as u32 } else { len as u32 };
        }
        base += 16;
    }
    len as u32
} }

/// One 512-bit compare over all 48 keys. Must run in streaming mode.
#[target_feature(enable = "sve")]
unsafe fn sve(keys: *const u8, target: u8, len: u64) -> u32 { unsafe {
    let pg = svwhilelt_b8_u64(0, len);
    let v = svld1_u8(pg, keys);
    let t = svdup_n_u8(target);
    let ge = svcmpge_u8(pg, v, t);
    // Lanes before the first match, counted: that is the index.
    let before = svbrkb_b_z(pg, ge);
    svcntp_b8(pg, before) as u32
} }
```

Driven by two million lookups over a fixed sorted key array, with the SVE arm
timed twice: once with `asm!("smstart sm")` outside the loop, and once with the
mode entered and left on every call. Correctness is checked across all 256
target bytes before any timing runs.

Two practical notes for anyone re-running it. The streaming region must contain
no Neon or floating-point instructions, because those are illegal in streaming
mode without `FEAT_SME_FA64`, and the compiler does not know it is inside one.
And `rustup component add rust-src` is how you find the real intrinsic names:
they are `svwhilelt_b8_u64`, not `svwhilelt_b8`, and guessing costs a build
cycle each time.
