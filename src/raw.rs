//! Weakly typed implementation of adaptive radix tree.
//!
//! The purpose of this module is to re-use as much code as possible between the
//! sequential ([`crate::sequential::Map`]) and concurrent ([`crate::concurrent::Map`])
//! tree implementations, and between instantiations of these trees with different
//! value types.
//!
//! This module contains:
//! - Structural types ([`crate::raw::edge`], [`crate::raw::node`], [`crate::raw::key`])
//! - Traversal for point operations ([`crate::raw::cursor`])
//! - Iteration for scan operations ([`crate::raw::iter`])
//!
//! This module is "raw" with respect to:
//! - Safe memory reclamation ([`crate::concurrent::smr`])
//! - Mutable vs. immutable access
//! - Value types ([`crate::sequential::Value`], [`crate::concurrent::Value`])

pub(crate) mod cursor;
pub(crate) mod edge;
pub(crate) mod iter;
pub mod key;
pub(crate) mod map;
pub(crate) mod node;
pub(crate) mod set;
pub(crate) mod shard;

pub(crate) use cursor::Cursor;
pub(crate) use edge::Edge;
pub use key::Key;
pub(crate) use map::Map;
pub(crate) use set::Set;
pub(crate) use shard::Shard;

// There once was a CPU extension called SVE. It went out to sea on an M4, and
// it sank.
//
// The idea is good and it will occur to everyone who reads `node_47`. That node
// holds up to 48 sorted keys and searching it is four 16-byte Neon compares.
// This machine reports `hw.optional.arm.sme_max_svl_b: 64`, a 512-bit streaming
// vector length, and 48 bytes fits in 512 bits with room left over. Four
// instructions become one. It is the most obvious win in the crate.
//
// It is a 34x loss. Measured on an M4 Max, two million searches of 48 keys,
// nanoseconds per search, all three implementations agreeing on every one of
// the 256 possible target bytes:
//
//     scalar                       3.64 ns
//     neon, 4x128-bit              0.93 ns    3.90x faster than scalar
//     sve 512-bit, smstart once   31.62 ns    0.03x of neon
//     sve 512-bit, per call       43.44 ns    0.02x of neon
//
// Read the third row before concluding it is the mode switch. That run enters
// streaming mode **once** for all two million iterations, so `SMSTART` is
// amortised to nothing, and it is still thirty-four times slower. Paying the
// switch per call only takes it from 31.6 to 43.4. The instructions themselves
// are slow here.
//
// Which follows from what the hardware is for. Apple built an SME unit to feed
// a ZA tile with matrix outer products. Streaming SVE exists on it because the
// architecture says it must, not because there is a fast general-purpose
// 512-bit vector engine underneath. Neon is the fast path on this chip, and the
// SME unit is a coprocessor with entirely different economics.
//
// Three further walls stand behind that number, any one of which is fatal on
// its own, so do not go looking for a toolchain fix:
//
//   1. SVE intrinsics are nightly-only, gated on `stdarch_aarch64_sve`, and
//      blocked on an unaccepted RFC for types whose size is not known at
//      compile time (rust-lang/rust#145052). Merged into stdarch in April 2026
//      and still gated on 1.96, 1.97 and 1.98.
//   2. Apple has no non-streaming SVE. `-C target-feature=+sve` compiles
//      cleanly, emits no SVE into this crate, and produces a binary that dies
//      with SIGILL.
//   3. `fearless_simd` has refused scalable vectors by design
//      (linebender/fearless_simd#339): its whole API rests on a compile-time
//      lane count, and the maintainer will not trade codegen on common hardware
//      for it.
//
// A machine with real non-streaming SVE, a Graviton or a Grace, would answer
// differently and none of this transfers to it. The harness is in
// `docs/M4-SVE-perf-review.md` if you want to re-run it rather than trust this.
pub(crate) static SIMD_LEVEL: spin::Once<fearless_simd::Level> = spin::Once::new();

/// The SIMD level this machine supports, detected once.
///
/// This was a `std::sync::LazyLock`, which is the only reason the crate needed
/// `std` at all in its core. `spin::Once` is the same thing without one: the
/// detection is idempotent, so a race costs a second detection and nothing else.
pub(crate) fn simd() -> fearless_simd::Level {
    *SIMD_LEVEL.call_once(detect)
}

#[cfg(feature = "std")]
fn detect() -> fearless_simd::Level {
    fearless_simd::Level::new()
}

/// Without `std` there is no runtime feature detection: `is_x86_feature_detected`
/// and its aarch64 twin are `std` macros. So this reports what the compiler was
/// told to target rather than what the chip turns out to have.
///
/// On aarch64 that costs nothing, since Neon is baseline for the architecture.
/// On x86 it means the scalar path unless the build asks for the instructions,
/// with `-C target-feature=+avx2` or a `target-cpu` that implies it.
#[cfg(not(feature = "std"))]
fn detect() -> fearless_simd::Level {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Neon is mandatory on aarch64, so the feature is present on
        // every target this arm compiles for.
        return fearless_simd::Level::Neon(unsafe {
            fearless_simd::aarch64::Neon::new_unchecked()
        });
    }
    #[cfg(not(target_arch = "aarch64"))]
    fearless_simd::Level::fallback()
}

/// Structural modification operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Smo {
    ReplaceNode,
    DeleteNode,
    CompressEdge,
}

impl Smo {
    #[inline]
    pub fn is_allocate(self) -> bool {
        matches!(self, Self::ReplaceNode)
    }
}

fn is_unique(keys: &[u8]) -> bool {
    let mut seen = [0u128; 2];
    for key in keys {
        let row = key / 128;
        let col = key % 128;
        let bit = 1 << col;
        if seen[row as usize] & bit > 0 {
            return false;
        }
        seen[row as usize] |= bit;
    }
    true
}

/// Compute the lowest byte index at which byte 0 appears in `array`.
/// If there is no zero, return 8.
///
/// https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
/// https://richardstartin.github.io/posts/finding-bytes
/// https://orlp.net/blog/extracting-depositing-bits/
/// https://lemire.me/blog/2022/01/21/swar-explained-parsing-eight-digits/
/// https://lamport.azurewebsites.net/pubs/multiple-byte.pdf
#[inline]
fn find_zero(array: u64) -> u8 {
    let high_if_zero_or_ge_0x80 = array.wrapping_sub(0x0101_0101_0101_0101);
    let high_if_lt_0x80 = !array;
    let high_if_zero = high_if_zero_or_ge_0x80 & high_if_lt_0x80 & 0x8080_8080_8080_8080;
    (high_if_zero.trailing_zeros() >> 3) as u8
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "proptest")]
    proptest::proptest! {
        #[test]
        fn find_zero_correct(array: u64) {
            let expected = array
                .to_le_bytes()
                .into_iter()
                .position(|byte| byte == 0)
                .unwrap_or(8)
                as u8;

            let actual = super::find_zero(array);

            assert_eq!(
                actual,
                expected,
                "find_zero mismatch for array = {array:#x}",
            );
        }
    }
}
