use arctic::Order;
use arctic::concurrent;
use arctic::sequential;
use std::sync::Barrier;

#[test]
fn turso_range_24230c111c599daff93a7abc11c5c72b33d0ebfd() {
    // https://github.com/jennyhour/turso-arctic/blob/2c7cbf300adacf8482346d6f927753c9074d8bd7/core/mvcc/database/mod.rs#L88-L111
    const fn turso_row_id(row_id: i64) -> u128 {
        const SIGN: u64 = 1u64.rotate_right(1);

        let table_id = (-1i64) as u64 ^ SIGN;
        ((table_id as u128) << 64) | (((row_id as u64) ^ SIGN) as u128)
    }

    let mut map = sequential::Map::<u128, u64>::default();
    let entries = (0..10u64)
        .map(|index| (turso_row_id(index as i64), index))
        .collect::<Vec<_>>();

    for (row_id, index) in &entries {
        assert!(map.upsert(*row_id, *index).is_err());
    }

    for (row_id, index) in &entries {
        assert_eq!(map.get(row_id), Some(index));
    }

    let prefix = map.range(turso_row_id(5)..=turso_row_id(i64::MAX));
    let values = prefix.values(Order::Ascend).copied().collect::<Vec<_>>();
    assert_eq!(values, (5..10).collect::<Vec<u64>>());
}

#[test]
fn insert_duplicate_82007770fb876db856313cebf12be21b9182f16a() {
    let map = concurrent::Map::<u64, u64>::default();
    map.insert(0u64, 0u64).unwrap();
    map.insert(0u64, 1u64).unwrap_err();
}

#[test]
fn range_common_prefix_72c2fceda258b00fc2e9d4a805b28e9ad8e8107d() {
    let map = crate::concurrent::Map::<u64, u64>::new();
    const NEEDLE: u64 = 0xE642_3BB1_ADBB_F000;
    const LOWER: u64 = 0x39_9100;
    const UPPER: u64 = 0xFF29_D24D_7E9A_920D;
    map.insert(NEEDLE, 0).unwrap();
    map.range(LOWER..=UPPER)
        .entries(Order::Ascend)
        .for_each(|(key, value)| {
            assert_eq!(key, NEEDLE);
            assert_eq!(value, 0);
        })
}

#[test]
fn disjoint_crud_remains_immediately_visible_under_contention() {
    let map = concurrent::Map::<u64, Box<u64>>::default();
    let barrier = Barrier::new(9);

    std::thread::scope(|scope| {
        for worker in 0..8u64 {
            let map = &map;
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                for sequence in 0..10_000u64 {
                    let key = worker * 10_000 + sequence;
                    map.insert(key, Box::new(key + 1))
                        .unwrap_or_else(|_| panic!("disjoint key already existed: {key}"));
                    assert_eq!(map.get(&key).as_deref().copied(), Some(key + 1));
                    assert_eq!(map.remove(&key).as_deref().copied(), Some(key + 1));
                }
            });
        }
        barrier.wait();
    });

    assert!(map.all().entries(Order::Ascend).next().is_none());
}

mod inverted_range {
    use arctic::Order;
    use arctic::concurrent;
    use arctic::key::BoxedStr;
    use arctic::key::NonNull;
    use arctic::key::Str;
    use arctic::sequential;

    /// An inverted range must yield nothing on a Node256-shaped tree.
    /// It used to overflow `u16` in `KeyIter256` in debug builds and yield
    /// every key repeatedly in release builds.
    #[test]
    #[expect(clippy::reversed_empty_ranges, reason = "Inverted on purpose")]
    fn node256_empty() {
        let mut map = sequential::Map::<u64, u64>::new();
        for i in 0..200u64 {
            let _ = map.upsert((i << 56) | 0xAA, i);
        }

        let shard = map.range(((150u64 << 56) | 0xAA)..=((10u64 << 56) | 0xAA));
        assert_eq!(shard.entries(Order::Ascend).count(), 0);
        assert_eq!(shard.entries(Order::Descend).count(), 0);
    }

    /// An inverted range must yield nothing on a Node15-shaped tree.
    /// The Node15/47 SIMD mask used to spuriously include the upper byte.
    #[test]
    #[expect(clippy::reversed_empty_ranges, reason = "Inverted on purpose")]
    fn node15_empty() {
        let mut map = sequential::Map::<u64, u64>::new();
        for i in 0..10u64 {
            let _ = map.upsert((i << 56) | 0xAA, i);
        }

        let shard = map.range(((8u64 << 56) | 0xAA)..=((1u64 << 56) | 0xAA));
        assert_eq!(shard.entries(Order::Ascend).count(), 0);
        assert_eq!(shard.entries(Order::Descend).count(), 0);
    }

    /// Bounds that share a long prefix and only invert at a deeper byte.
    #[test]
    #[expect(clippy::reversed_empty_ranges, reason = "Inverted on purpose")]
    fn deep_divergence_empty() {
        let map = concurrent::Map::<u64, u64>::default();
        for i in 0..10u64 {
            map.insert(0x0101_0101_0101_0100 | i, i).unwrap();
        }

        let shard = map.range(0x0101_0101_0101_0108u64..=0x0101_0101_0101_0101u64);
        assert_eq!(shard.entries(Order::Ascend).count(), 0);
        assert_eq!(shard.entries(Order::Descend).count(), 0);
    }

    /// `x..=x` still yields exactly the single matching entry.
    #[test]
    fn single_key_range() {
        let mut map = sequential::Map::<u64, u64>::new();
        for i in 0..10u64 {
            let _ = map.upsert((i << 56) | 0xAA, i);
        }

        let key = (5u64 << 56) | 0xAA;
        let shard = map.range(key..=key);
        let entries = shard.entries(Order::Ascend).collect::<Vec<_>>();
        assert_eq!(entries, vec![(key, &5)]);
    }

    /// Inverted string ranges, including the proper-prefix case where the
    /// upper bound is a prefix of the lower bound.
    #[test]
    fn boxed_str_empty() {
        let mut map = sequential::Map::<BoxedStr<NonNull>, u64>::new();
        for key in ["a", "ab", "b", "ba"] {
            let _ = map.upsert(Str::new(key).expect("No null byte"), 0);
        }

        let b = Str::<NonNull>::new("b").expect("No null byte");
        let a = Str::<NonNull>::new("a").expect("No null byte");
        let ab = Str::<NonNull>::new("ab").expect("No null byte");

        assert_eq!(map.range(b..=a).entries(Order::Ascend).count(), 0);
        assert_eq!(map.range(ab..=a).entries(Order::Ascend).count(), 0);

        // Non-inverted ranges over the same bounds still work.
        assert_eq!(map.range(a..=b).entries(Order::Ascend).count(), 3);
        assert_eq!(map.range(a..=ab).entries(Order::Ascend).count(), 2);
        assert_eq!(map.range(a..=a).entries(Order::Ascend).count(), 1);
    }
}

#[cfg(feature = "proptest")]
mod inverted_range_proptest {
    use std::collections::BTreeMap;

    use arctic::Order;
    use arctic::sequential;
    use proptest::prelude::*;

    proptest::proptest! {
        /// `range` matches `BTreeMap::range` for ordered bounds and yields
        /// nothing for inverted bounds (where `BTreeMap::range` would panic).
        #[test]
        fn matches_btree_or_empty(
            keys in proptest::collection::vec(any::<u64>(), 0..64),
            lower in any::<u64>(),
            upper in any::<u64>(),
            descend in any::<bool>(),
        ) {
            let mut map = sequential::Map::<u64, u64>::new();
            let mut reference = BTreeMap::new();
            for (index, key) in keys.iter().copied().enumerate() {
                let _ = map.upsert(key, index as u64);
                reference.insert(key, index as u64);
            }

            let order = if descend { Order::Descend } else { Order::Ascend };
            let shard = map.range(lower..=upper);
            let actual = shard.entries(order).collect::<Vec<_>>();

            let expected: Vec<(u64, &u64)> = if lower <= upper {
                let entries = reference.range(lower..=upper).map(|(k, v)| (*k, v));
                if descend {
                    entries.rev().collect()
                } else {
                    entries.collect()
                }
            } else {
                Vec::new()
            };

            prop_assert_eq!(actual, expected);
        }

        /// Same property for string keys, whose readers have terminators.
        #[test]
        fn matches_btree_or_empty_str(
            keys in proptest::collection::vec("[a-c]{0,4}", 0..24),
            lower in "[a-c]{0,4}",
            upper in "[a-c]{0,4}",
        ) {
            use arctic::key::BoxedStr;
            use arctic::key::NonNull;
            use arctic::key::Str;

            let mut map = sequential::Map::<BoxedStr<NonNull>, u64>::new();
            let mut reference = BTreeMap::new();
            for (index, key) in keys.iter().enumerate() {
                let _ = map.upsert(
                    Str::<NonNull>::new(key).expect("No null byte"),
                    index as u64,
                );
                reference.insert(key.clone(), index as u64);
            }

            let shard = map.range(
                Str::<NonNull>::new(&lower).expect("No null byte")
                    ..=Str::<NonNull>::new(&upper).expect("No null byte"),
            );
            let actual = shard
                .entries(Order::Ascend)
                .map(|(key, value)| (key.as_str().to_owned(), *value))
                .collect::<Vec<_>>();

            let expected: Vec<(String, u64)> = if lower <= upper {
                reference
                    .range(lower..=upper)
                    .map(|(k, v)| (k.clone(), *v))
                    .collect()
            } else {
                Vec::new()
            };

            prop_assert_eq!(actual, expected);
        }
    }
}

/// `SequentialMap::remove_non_recursive` used to hardcode the full path type
/// internally, silently behaving as the recursive `remove` and collapsing
/// empty nodes it documents leaving in place.
#[test]
fn sequential_remove_non_recursive_keeps_empty_nodes() {
    use arctic::topology;

    // Two 16-byte keys diverging at the first byte: each branch is a
    // 7-byte edge, a single-child node, and a 7-byte value edge.
    const A: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;
    const B: u128 = 0x0202_0202_0202_0202_0202_0202_0202_0202;

    let build = || {
        let mut map = sequential::Map::<u128, u64>::new();
        assert!(map.insert(A, 1).is_ok());
        assert!(map.insert(B, 2).is_ok());
        map
    };

    // The recursive remove collapses B's emptied single-child node, leaving
    // a valid topology.
    let mut recursive = build();
    assert_eq!(recursive.remove(&B), Some(2));
    assert!(recursive.export_topology(|value| *value).is_ok());

    // The non-recursive remove must leave the emptied node in place, which
    // the topology export detects as a node without live branches.
    let mut non_recursive = build();
    assert_eq!(non_recursive.remove_non_recursive(&B), Some(2));
    assert_eq!(
        non_recursive.export_topology(|value| *value),
        Err(topology::Error::EmptyNode),
    );

    // The map still behaves correctly around the leftover node.
    assert_eq!(non_recursive.get(&A), Some(&1));
    assert_eq!(non_recursive.get(&B), None);
    assert!(non_recursive.insert(B, 3).is_ok());
    assert_eq!(non_recursive.get(&B), Some(&3));
}
