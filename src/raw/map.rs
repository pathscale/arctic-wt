use core::ops::RangeFull;

use crate::raw;
use crate::raw::Cursor;
use crate::raw::Edge;
use crate::raw::Key;
use crate::raw::cursor;
use crate::raw::edge;
use crate::raw::iter::Order;
use crate::raw::iter::PostorderIter;
use crate::sync::Atomic;

#[repr(transparent)]
pub(crate) struct Map<K: Key>(Atomic<Edge<K::Edge>>);

impl<K: Key> Map<K> {
    /// Constructs a new empty map. Does not allocate.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self(Atomic::from_raw(crate::sync::AtomicU128::new(
            Edge::<K::Edge>::NULL.into_raw(),
        )))
    }

    pub(crate) fn postorder<'g>(&'g mut self, order: Option<Order>) -> PostorderIter<'g, K::Edge> {
        unsafe { PostorderIter::new(self.root(), order) }
    }

    /// Drain the exclusively owned tree without building ordered node
    /// iterators. Destruction does not need key order, so scanning physical
    /// edge slots avoids the per-node iterator allocations used by scans.
    pub(crate) fn drain(&mut self, mut drop_value: impl FnMut(u64)) {
        let root = std::mem::replace(self.0.get_mut_packed(), Edge::<K::Edge>::NULL);
        match root.child() {
            None => {}
            Some(edge::Child::Value(value)) => drop_value(value),
            Some(edge::Child::Node(node)) => unsafe {
                node.deallocate_tree::<K::Edge, _>(drop_value)
            },
        }
    }

    #[inline]
    pub(crate) unsafe fn cursor<'g, 'k, P: cursor::Path<K::Read<'k>>>(
        &'g self,
        key: impl Into<K::Read<'k>>,
    ) -> Cursor<'g, K::Read<'k>, P> {
        unsafe { Cursor::<_, P>::new(self.root(), key.into()) }
    }

    #[inline]
    pub(crate) unsafe fn all(&self) -> raw::Shard<'_, 'static, K, RangeFull> {
        unsafe { raw::Shard::<K>::new_all(self.root()) }
    }

    #[inline]
    pub(crate) unsafe fn prefix<'k>(
        &self,
        prefix: impl Into<K::Read<'k>>,
    ) -> raw::Shard<'_, 'k, K, RangeFull> {
        unsafe { raw::Shard::<K>::new_prefix(self.root(), prefix.into()) }
    }

    #[inline]
    pub(crate) unsafe fn range<'k, R>(
        &self,
        range: R,
        prefix: K::Read<'k>,
    ) -> raw::Shard<'_, 'k, K, R>
    where
        R: raw::iter::Range<K::Read<'k>>,
    {
        unsafe { raw::Shard::new_range(self.root(), range, prefix) }
    }

    #[inline]
    pub(crate) fn root(&self) -> &Atomic<Edge<K::Edge>> {
        &self.0
    }

    /// Bytes occupied by adaptive node allocations reachable from the root.
    ///
    /// The caller must protect the full tree from reclamation for the walk.
    /// Concurrent structural changes can make this a weak snapshot, but every
    /// node counted is protected and its concrete allocation size is exact.
    pub(crate) unsafe fn allocated_node_bytes(&self) -> usize {
        let root = self
            .root()
            .load_packed(core::sync::atomic::Ordering::Acquire);
        let Some(child) = root.child() else { return 0 };
        let edge::Child::Node(root) = child else {
            return 0;
        };

        let mut bytes = 0;
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            bytes += node.allocation_size();
            for (_, _, pointer) in unsafe { node.topology_entries() } {
                let edge = unsafe { Edge::<K::Edge>::from_raw_ref(pointer.as_ref()) }
                    .load_packed(core::sync::atomic::Ordering::Acquire);
                if let Some(edge::Child::Node(child)) = edge.child() {
                    pending.push(child);
                }
            }
        }
        bytes
    }

    /// Replace the root while the map is exclusively owned.
    ///
    /// This is used only by pointer-free topology restoration. The caller must
    /// ensure the current root is empty so no allocation is leaked.
    pub(crate) fn set_empty_root(&mut self, root: ribbit::Packed<Edge<K::Edge>>) {
        validate!(self.0.get_mut_packed().is_null());
        *self.0.get_mut_packed() = root;
    }
}

impl<K> Default for Map<K>
where
    K: Key,
{
    #[inline]
    fn default() -> Self {
        Self(Atomic::new_packed(Edge::NULL))
    }
}
