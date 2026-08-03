//! Pointer-free import and export of Arctic's adaptive radix-tree topology.
//!
//! This module deliberately exposes a typed interchange representation rather
//! than a byte codec. Durable framing, checksums, generations, and write-ahead
//! logging remain the caller's responsibility.

use core::fmt;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;

use ribbit::Unpack as _;

use crate::concurrent;
use crate::raw::Edge as RawEdge;
use crate::raw::edge;
use crate::raw::edge::Meta as _;
use crate::raw::node;
use crate::sequential;
use crate::sync::Atomic;

/// Version of the typed topology interchange contract.
pub const VERSION: u16 = 1;

/// An exported, pointer-free Arctic topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology<V> {
    /// Interchange contract version. Must equal [`VERSION`].
    pub version: u16,
    /// Root edge, or `None` for an empty map.
    pub root: Option<Edge<V>>,
}

/// A compressed edge and its child.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge<V> {
    /// Arctic compressed-edge bits with transient flags cleared.
    pub metadata: u64,
    /// Value or adaptive node reached by this edge.
    pub child: Child<V>,
}

/// Child reached by a compressed edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Child<V> {
    /// A caller-encoded value. This is never Arctic's raw in-memory value word.
    Value(V),
    /// An adaptive radix-tree node.
    Node(Node<V>),
}

/// An Arctic adaptive node and its physical branches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node<V> {
    /// Exact adaptive node representation.
    pub kind: NodeKind,
    /// Number of physical slots initialized in the node header.
    ///
    /// This can exceed the number of live branches after removals.
    pub slot_count: u16,
    /// Live branches, including their physical edge slots.
    pub branches: Vec<Branch<V>>,
}

/// A byte branch stored in a physical node edge slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Branch<V> {
    /// Radix byte selecting this branch.
    pub key: u8,
    /// Physical slot in the node's edge array.
    pub slot: u16,
    /// Compressed child edge.
    pub edge: Edge<V>,
}

/// Arctic's adaptive node representations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// Up to three branches in one cache line.
    Node3,
    /// Up to fifteen branches.
    Node15,
    /// Up to forty-seven branches.
    Node47,
    /// Directly addressed 256-way node.
    Node256,
}

impl NodeKind {
    fn capacity(self) -> usize {
        match self {
            Self::Node3 => 3,
            Self::Node15 => 15,
            Self::Node47 => 47,
            Self::Node256 => 256,
        }
    }

    fn from_raw(kind: ribbit::Packed<node::Type>) -> Self {
        match kind.unpack() {
            node::Type::Node3 => Self::Node3,
            node::Type::Node15 => Self::Node15,
            node::Type::Node47 => Self::Node47,
            node::Type::Node256 => Self::Node256,
        }
    }

    fn into_raw(self) -> node::Type {
        match self {
            Self::Node3 => node::Type::Node3,
            Self::Node15 => node::Type::Node15,
            Self::Node47 => node::Type::Node47,
            Self::Node256 => node::Type::Node256,
        }
    }
}

/// A malformed or unsupported topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The interchange version is not supported by this crate.
    UnsupportedVersion {
        /// Version found in the topology.
        found: u16,
    },
    /// Compressed-edge metadata contains invalid or transient bits.
    InvalidMetadata {
        /// Rejected metadata word.
        metadata: u64,
    },
    /// A value does not terminate at the unsigned key's exact byte length.
    InvalidKeyLength {
        /// Number of key bytes represented by the path.
        found: usize,
        /// Required number of key bytes.
        expected: usize,
    },
    /// A node has no live branches.
    EmptyNode,
    /// A node contains more branches than its recorded representation permits.
    NodeCapacity {
        /// Recorded node representation.
        kind: NodeKind,
        /// Number of branches found.
        found: usize,
    },
    /// Two branches in one node use the same radix byte.
    DuplicateKey {
        /// Duplicated radix byte.
        key: u8,
    },
    /// A physical edge slot is invalid or duplicated.
    InvalidSlot {
        /// Recorded node representation.
        kind: NodeKind,
        /// Invalid physical slot.
        slot: u16,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found } => {
                write!(formatter, "unsupported Arctic topology version {found}")
            }
            Self::InvalidMetadata { metadata } => {
                write!(formatter, "invalid Arctic edge metadata {metadata:#018x}")
            }
            Self::InvalidKeyLength { found, expected } => write!(
                formatter,
                "Arctic path has {found} key bytes but key type requires {expected}",
            ),
            Self::EmptyNode => formatter.write_str("Arctic topology contains an empty node"),
            Self::NodeCapacity { kind, found } => {
                write!(formatter, "{kind:?} cannot contain {found} branches")
            }
            Self::DuplicateKey { key } => {
                write!(formatter, "Arctic node contains duplicate byte {key}")
            }
            Self::InvalidSlot { kind, slot } => {
                write!(formatter, "invalid or duplicate {kind:?} slot {slot}")
            }
        }
    }
}

impl std::error::Error for Error {}

mod private {
    use super::*;

    pub trait Sealed: crate::Key {
        const BYTES: usize;

        fn metadata_to_raw(metadata: ribbit::Packed<Self::Edge>) -> u64;

        unsafe fn metadata_from_raw(raw: u64) -> ribbit::Packed<Self::Edge>;

        fn validate_metadata(raw: u64) -> Result<usize, Error>;
    }

    macro_rules! impl_be_key {
        ($($key:ty),+ $(,)?) => {
            $(
                impl Sealed for $key {
                    const BYTES: usize = core::mem::size_of::<Self>();

                    fn metadata_to_raw(metadata: ribbit::Packed<Self::Edge>) -> u64 {
                        metadata.with_frozen(false).with_value(false).into_raw()
                    }

                    unsafe fn metadata_from_raw(raw: u64) -> ribbit::Packed<Self::Edge> {
                        unsafe { ribbit::Packed::<edge::Be>::from_raw_unchecked(raw) }
                    }

                    fn validate_metadata(raw: u64) -> Result<usize, Error> {
                        validate_be_metadata(raw)
                    }
                }
            )+
        };
    }

    impl_be_key!(u16, u32, u128);
    #[cfg(not(feature = "opt-no-int"))]
    impl_be_key!(u64);

    #[cfg(feature = "opt-no-int")]
    impl Sealed for u64 {
        const BYTES: usize = core::mem::size_of::<Self>();

        fn metadata_to_raw(metadata: ribbit::Packed<Self::Edge>) -> u64 {
            metadata.with_frozen(false).with_value(false).into_raw()
        }

        unsafe fn metadata_from_raw(raw: u64) -> ribbit::Packed<Self::Edge> {
            unsafe { ribbit::Packed::<edge::Le>::from_raw_unchecked(raw) }
        }

        fn validate_metadata(raw: u64) -> Result<usize, Error> {
            validate_le_metadata(raw)
        }
    }
}

/// Unsigned key types supported by pointer-free topology snapshots.
///
/// This trait is sealed. Version 1 intentionally matches the unsigned-key
/// restriction in WorkTable's Arctic backend.
pub trait Key: crate::Key + private::Sealed {}

impl Key for u16 {}
impl Key for u32 {}
impl Key for u64 {}
impl Key for u128 {}

impl<V> Topology<V> {
    /// Validate all structural, slot, metadata, and key-length invariants.
    pub fn validate<K: Key>(&self) -> Result<(), Error> {
        if self.version != VERSION {
            return Err(Error::UnsupportedVersion {
                found: self.version,
            });
        }

        if let Some(root) = &self.root {
            validate_edge::<K, V>(root, 0)?;
        }
        Ok(())
    }
}

impl<K, V> sequential::Map<K, V>
where
    K: Key,
    V: sequential::Value,
{
    /// Export the exact quiescent Arctic topology without process pointers.
    ///
    /// `encode` must copy or otherwise encode the logical value; the raw value
    /// word stored in Arctic is deliberately never exposed.
    pub fn export_topology<T>(
        &self,
        mut encode: impl FnMut(&V) -> T,
    ) -> Result<Topology<T>, Error> {
        let root =
            unsafe { export_edge::<K, V, T, _>(NonNull::from(self.raw.root()), &mut encode) };
        let topology = Topology {
            version: VERSION,
            root,
        };
        topology.validate::<K>()?;
        Ok(topology)
    }

    /// Restore a validated topology and reconstruct its exact adaptive node kinds.
    pub fn from_topology<T>(
        topology: Topology<T>,
        mut decode: impl FnMut(T) -> V,
    ) -> Result<Self, Error> {
        topology.validate::<K>()?;

        let mut map = Self::new();
        if let Some(root) = topology.root {
            let root = unsafe { import_edge::<K, V, T, _>(root, &mut decode) };
            map.raw.set_empty_root(root);
        }
        Ok(map)
    }
}

impl<K, V, S> concurrent::Map<K, V, S>
where
    K: Key,
    V: concurrent::Value,
    S: concurrent::Smr<K, V>,
{
    /// Export an exact topology through an exclusive sequential view.
    ///
    /// Requiring `&mut self` prevents concurrent mutation while the snapshot is
    /// captured and adds no synchronization to point operations.
    pub fn export_topology<T>(
        &mut self,
        encode: impl FnMut(&V) -> T,
    ) -> Result<Topology<T>, Error> {
        self.as_sequential().export_topology(encode)
    }
}

impl<K, V, S> concurrent::Map<K, V, S>
where
    K: Key,
    V: concurrent::Value,
    S: concurrent::Smr<K, V> + Default,
{
    /// Restore a concurrent map from a validated pointer-free topology.
    pub fn from_topology<T>(
        topology: Topology<T>,
        decode: impl FnMut(T) -> V,
    ) -> Result<Self, Error> {
        sequential::Map::from_topology(topology, decode).map(Into::into)
    }
}

fn validate_edge<K: Key, V>(edge: &Edge<V>, path_bytes: usize) -> Result<(), Error> {
    let prefix_bytes = K::validate_metadata(edge.metadata)?;
    let path_bytes = path_bytes
        .checked_add(prefix_bytes)
        .ok_or(Error::InvalidKeyLength {
            found: usize::MAX,
            expected: K::BYTES,
        })?;

    match &edge.child {
        Child::Value(_) if path_bytes == K::BYTES => Ok(()),
        Child::Value(_) => Err(Error::InvalidKeyLength {
            found: path_bytes,
            expected: K::BYTES,
        }),
        Child::Node(node) => {
            if path_bytes >= K::BYTES {
                return Err(Error::InvalidKeyLength {
                    found: path_bytes + 1,
                    expected: K::BYTES,
                });
            }
            if node.branches.is_empty() {
                return Err(Error::EmptyNode);
            }
            let slot_count = node.slot_count as usize;
            if node.branches.len() > node.kind.capacity()
                || slot_count > node.kind.capacity()
                || slot_count < node.branches.len()
                || (node.kind == NodeKind::Node256 && slot_count != 256)
            {
                return Err(Error::NodeCapacity {
                    kind: node.kind,
                    found: slot_count.max(node.branches.len()),
                });
            }

            let mut keys = [false; 256];
            let mut slots = [false; 256];
            for branch in &node.branches {
                if core::mem::replace(&mut keys[branch.key as usize], true) {
                    return Err(Error::DuplicateKey { key: branch.key });
                }

                let slot = branch.slot as usize;
                if slot >= slot_count
                    || core::mem::replace(&mut slots[slot], true)
                    || (node.kind == NodeKind::Node256 && slot != branch.key as usize)
                {
                    return Err(Error::InvalidSlot {
                        kind: node.kind,
                        slot: branch.slot,
                    });
                }

                validate_edge::<K, V>(&branch.edge, path_bytes + 1)?;
            }
            Ok(())
        }
    }
}

fn validate_be_metadata(metadata: u64) -> Result<usize, Error> {
    const FLAGS: u64 = 0b111;
    const LENGTH: u64 = 0b11_1000;

    let bits = (metadata & LENGTH) as usize;
    let prefix_mask = if bits == 0 {
        0
    } else {
        u64::MAX << (64 - bits)
    };
    if metadata & FLAGS != 0 || metadata & !(prefix_mask | LENGTH) != 0 {
        return Err(Error::InvalidMetadata { metadata });
    }
    Ok(bits / 8)
}

#[cfg(feature = "opt-no-int")]
fn validate_le_metadata(metadata: u64) -> Result<usize, Error> {
    const FLAGS: u64 = 0b111 << 56;
    const LENGTH: u64 = 0b11_1000 << 56;

    let bits = ((metadata & LENGTH) >> 56) as usize;
    let prefix_mask = if bits == 0 {
        0
    } else {
        u64::MAX >> (64 - bits)
    };
    if metadata & FLAGS != 0 || metadata & !(prefix_mask | LENGTH) != 0 {
        return Err(Error::InvalidMetadata { metadata });
    }
    Ok(bits / 8)
}

unsafe fn export_edge<K, V, T, F>(
    pointer: NonNull<Atomic<RawEdge<K::Edge>>>,
    encode: &mut F,
) -> Option<Edge<T>>
where
    K: Key,
    V: sequential::Value,
    F: FnMut(&V) -> T,
{
    let edge = unsafe { pointer.as_ref() }.load_packed(Ordering::Acquire);
    let child = edge.child()?;
    let metadata = K::metadata_to_raw(edge.meta());

    let child = match child {
        edge::Child::Value(_) => {
            let value = unsafe { RawEdge::as_value_unchecked(pointer).cast::<V>().as_ref() };
            Child::Value(encode(value))
        }
        edge::Child::Node(node) => {
            let kind = NodeKind::from_raw(node.r#type());
            let entries = unsafe { node.topology_entries() };
            let slot_count = entries.len() as u16;
            let mut branches = Vec::new();
            for (key, slot, pointer) in entries {
                if let Some(edge) = unsafe { export_edge::<K, V, T, F>(pointer.cast(), encode) } {
                    branches.push(Branch { key, slot, edge });
                }
            }
            branches.sort_unstable_by_key(|branch| branch.slot);
            Child::Node(Node {
                kind,
                slot_count,
                branches,
            })
        }
    };

    Some(Edge { metadata, child })
}

unsafe fn import_edge<K, V, T, F>(edge: Edge<T>, decode: &mut F) -> ribbit::Packed<RawEdge<K::Edge>>
where
    K: Key,
    V: sequential::Value,
    F: FnMut(T) -> V,
{
    let metadata = unsafe { K::metadata_from_raw(edge.metadata) };
    match edge.child {
        Child::Value(value) => RawEdge::new_value(metadata, decode(value).into_raw()),
        Child::Node(mut node) => {
            node.branches.sort_unstable_by_key(|branch| branch.slot);
            let kind = node.kind;
            let slot_count = node.slot_count as usize;
            let (keys, edges) = if kind == NodeKind::Node256 {
                let mut keys = Vec::with_capacity(node.branches.len());
                let mut edges = Vec::with_capacity(node.branches.len());
                for branch in node.branches {
                    keys.push(branch.key);
                    edges.push(unsafe { import_edge::<K, V, T, F>(branch.edge, decode) }.erase());
                }
                (keys, edges)
            } else {
                let mut used_keys = [false; 256];
                for branch in &node.branches {
                    used_keys[branch.key as usize] = true;
                }

                let mut filler_keys = (u8::MIN..=u8::MAX).filter(|key| !used_keys[*key as usize]);
                let mut keys = Vec::with_capacity(slot_count);
                for _ in 0..slot_count {
                    keys.push(
                        filler_keys
                            .next()
                            .expect("validated node has spare key bytes"),
                    );
                }
                let mut edges = vec![RawEdge::<K::Edge>::NULL.erase(); slot_count];

                for branch in node.branches {
                    let slot = branch.slot as usize;
                    keys[slot] = branch.key;
                    edges[slot] = unsafe { import_edge::<K, V, T, F>(branch.edge, decode) }.erase();
                }
                (keys, edges)
            };
            let pointer = unsafe { node::Ptr::new_exact(kind.into_raw(), &keys, &edges) };
            RawEdge::new_node(metadata, pointer)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_kind<V>(edge: &Edge<V>, expected: NodeKind) -> bool {
        match &edge.child {
            Child::Value(_) => false,
            Child::Node(node) => {
                node.kind == expected
                    || node
                        .branches
                        .iter()
                        .any(|branch| contains_kind(&branch.edge, expected))
            }
        }
    }

    fn assert_round_trip_for_keys(keys: impl IntoIterator<Item = u64>, expected: NodeKind) {
        let mut map = sequential::Map::<u64, u64>::new();
        for key in keys {
            map.insert(key, key.rotate_left(17)).unwrap();
        }

        let before = map.export_topology(|value| *value).unwrap();
        assert!(contains_kind(before.root.as_ref().unwrap(), expected));
        let restored =
            sequential::Map::<u64, u64>::from_topology(before.clone(), |value| value).unwrap();
        let after = restored.export_topology(|value| *value).unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn sequential_round_trip_preserves_topology_and_values() {
        let mut map = sequential::Map::<u64, Box<u64>>::new();
        for key in (0..4_096).map(|key| key * 17) {
            map.insert(key, Box::new(key ^ 0xA5A5)).unwrap();
        }
        for key in (0..4_096).step_by(7).map(|key| key * 17) {
            map.remove(&key).unwrap();
        }

        let before = map.export_topology(|value| **value).unwrap();
        let restored =
            sequential::Map::<u64, Box<u64>>::from_topology(before.clone(), Box::new).unwrap();
        let after = restored.export_topology(|value| **value).unwrap();

        assert_eq!(after, before);
        for key in (0..4_096).map(|key| key * 17) {
            let expected = (key / 17 % 7 != 0).then_some(key ^ 0xA5A5);
            assert_eq!(restored.get(&key).map(|value| **value), expected);
        }
    }

    #[test]
    fn concurrent_round_trip_requires_exclusive_snapshot() {
        let mut map = concurrent::Map::<u64, Box<u64>>::default();
        for key in 0..1_024 {
            map.insert(key, Box::new(key + 1)).unwrap();
        }

        let topology = map.export_topology(|value| **value).unwrap();
        let restored = concurrent::Map::<u64, Box<u64>>::from_topology(topology, Box::new).unwrap();

        for key in 0..1_024 {
            assert_eq!(restored.get(&key).as_deref(), Some(&(key + 1)));
        }
    }

    #[test]
    fn rejects_transient_metadata_flags() {
        let topology = Topology::<u64> {
            version: VERSION,
            root: Some(Edge {
                metadata: 1,
                child: Child::Value(42),
            }),
        };
        assert_eq!(
            topology.validate::<u64>(),
            Err(Error::InvalidMetadata { metadata: 1 }),
        );
    }

    #[test]
    fn preserves_every_adaptive_node_kind() {
        assert_round_trip_for_keys(0..2, NodeKind::Node3);
        assert_round_trip_for_keys(0..10, NodeKind::Node15);
        assert_round_trip_for_keys(0..32, NodeKind::Node47);
        assert_round_trip_for_keys(0..256, NodeKind::Node256);
    }
}
