use core::num::NonZeroU64;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::Key;
use crate::concurrent::Smr;
use crate::concurrent::Value;
use crate::concurrent::smr;
use crate::stat;

const RECLAIM_BATCH: usize = 256;

/// Per-map [`ps_reclaim::Domain`] backend for safe memory reclamation.
///
/// Retirements are advanced in bounded batches. Once the batch threshold is
/// reached, later operations retry the advance before pinning until the
/// backlog falls below it. That allows reclamation to make progress under
/// overlapping readers without paying for a domain scan on every mutation or
/// putting a collector-wide quiescent-state wait on the operation path.
///
/// # Examples
///
/// ```rust
/// use arctic::ConcurrentMap;
/// use arctic::concurrent::smr::PsReclaim;
///
/// let map = ConcurrentMap::<u64, Box<u64>, PsReclaim>::new();
/// map.insert(1, Box::new(2)).unwrap();
/// assert_eq!(map.get(&1).map(|value| *value), Some(2));
/// ```
#[derive(Default)]
pub struct PsReclaim {
    domain: ps_reclaim::Domain,
    pending: AtomicUsize,
}

impl PsReclaim {
    #[inline]
    fn advance_if_needed(&self) {
        if self.pending.load(Ordering::Acquire) < RECLAIM_BATCH {
            return;
        }

        let reclaimed = self.domain.advance();
        if reclaimed != 0 {
            self.pending.fetch_sub(reclaimed, Ordering::AcqRel);
        }
    }

    #[inline]
    fn retire(&self, reclaim: impl FnOnce() + Send + 'static) {
        self.domain.retire(reclaim);
        self.pending.fetch_add(1, Ordering::Release);
    }
}

impl<K: Key, V: Value> Smr<K, V> for PsReclaim {
    type Guard<'g>
        = PsReclaimGuard<'g>
    where
        V: 'g,
        Self: 'g;

    #[inline]
    fn guard<'g>(&'g self, _: K::Read<'_>) -> Self::Guard<'g>
    where
        V: 'g,
    {
        self.advance_if_needed();
        PsReclaimGuard {
            smr: self,
            pin: Some(self.domain.pin()),
            retired: false,
        }
    }

    fn garbage(&self) -> u32 {
        self.pending.load(Ordering::Acquire).min(u32::MAX as usize) as u32
    }
}

/// Arctic guard backed by one pin in the map's reclamation domain.
pub struct PsReclaimGuard<'a> {
    smr: &'a PsReclaim,
    pin: Option<ps_reclaim::Guard<'a>>,
    retired: bool,
}

impl<V: Value> smr::Guard<V> for PsReclaimGuard<'_> {
    unsafe fn retire_node(&mut self, _bits: usize, node: NonZeroU64) {
        stat::increment(stat::Counter::Retire);
        self.smr
            .retire(move || unsafe { smr::deallocate_node(node) });
        self.retired = true;
    }

    unsafe fn retire_value(&mut self, value: u64) {
        stat::increment(stat::Counter::Retire);
        self.smr
            .retire(move || unsafe { smr::deallocate_value::<V>(value) });
        self.retired = true;
    }
}

impl Drop for PsReclaimGuard<'_> {
    fn drop(&mut self) {
        self.pin.take();
        if self.retired {
            self.smr.advance_if_needed();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PsReclaim, RECLAIM_BATCH};
    use crate::ConcurrentMap;
    use crate::concurrent::Smr;

    #[test]
    fn removed_values_are_reclaimed_without_a_quiescent_map() {
        let map = ConcurrentMap::<u64, Box<u64>, PsReclaim>::new();
        for key in 0..RECLAIM_BATCH as u64 * 2 {
            map.insert(key, Box::new(key)).unwrap();
            assert_eq!(map.remove(&key).map(|value| *value), Some(key));
        }

        for key in 0..8 {
            assert!(map.get(&key).is_none());
        }

        assert!(Smr::<u64, Box<u64>>::garbage(map.smr()) < RECLAIM_BATCH as u32);
    }
}
