# v0.1.10

- Build without `std`. The default feature set gains `std`, so nothing changes for any existing user; with `default-features = false` the crate does not link it. `smr-ps-reclaim` and `stat-garbage` imply `std`, since ps-reclaim is `std` only and the garbage counter is this crate's only `thread_local!`. `NoOp` is what remains available without it.
- Without `std` the SIMD level comes from `cfg!(target_feature)` at compile time rather than runtime detection, which is a `std` facility. On aarch64 that costs nothing, Neon being mandatory in the architecture; on x86 it means the scalar path unless the build asks, with `-C target-feature=+avx2` or a `target-cpu` that implies it.
- `docs/M4-SVE-perf-review.md` records why there is no SVE path and why there will not be one on Apple silicon: measured at 34x slower than Neon for `node_47`'s search, with streaming mode amortised to nothing.
- One new dependency, `spin`, for a `no_std` `Once` holding the SIMD level. It has no dependencies of its own.

# v0.1.6

- Clamp integer-reader `match_prefix` to the reader length, fixing a reachable `unreachable!` panic in concurrent remove racing a neighbor-insert split and reinsert.
- Return an empty iterator for inverted range bounds instead of panicking (debug) or yielding wrapped duplicates (release).
- Require `V: Send` for `ConcurrentMap` to be `Sync`; `remove()` migrates the value drop across threads.
- Wire the path type through sequential `remove_raw` so `remove_non_recursive` is actually non-recursive.

# v0.1.4

- Port SIMD code to `fearless_simd`.
- Fix memory leak during recursive deallocation (https://github.com/nwtnni/arctic/issues/17).
- Add missing `Acquire` fence for a `get` on indirect values (https://github.com/nwtnni/arctic/issues/18).

# v0.1.3

- Fix performance regression for sequential keys.
    - Ensure Node256 is page-aligned, otherwise we encounter kernel-level
      contention when threads page fault on different Node256s that share
      the same page.
- Replace `memcpy` call with inline assembly `mov` for unsized keys.
- Fix edge case when removing unsized, non-null keys.

# v0.1.2

- Fix inverted `opt-no-path` feature flag.

# v0.1.1

- Fix badge links in `README.md`.

# v0.1.0

Initial release.
