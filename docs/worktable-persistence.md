# Pointer-free Arctic topology for WorkTable

Status: implementation in progress on `feat/worktable-persistence`.

## Contract

`arctic-wt` exposes a typed topology interchange representation; WorkTable owns
the durable page framing, checksums, generation protocol, and WAL.

The interchange representation records Arctic's own structure:

- compressed-edge metadata;
- `Node3`, `Node15`, `Node47`, or `Node256` for every node;
- the initialized physical slot span, including holes left by removals;
- each live branch byte and physical edge slot; and
- caller-encoded values.

It never records process addresses, allocator state, atomics, locks, frozen
flags, SMR state, or the raw representation of indirect values. A snapshot is
therefore recognizably Arctic on disk without being a memory image.

## Safety and consistency

Version 1 export requires exclusive access to the map. For `ConcurrentMap`, the
API takes `&mut self` and delegates to its sequential view. That makes the first
format implementation exact without adding overhead to Arctic's point-operation
hot path.

WorkTable must coordinate a quiescent checkpoint and retain a logical WAL before
the checkpoint begins. Recovery restores the backend-shaped checkpoint and then
replays later logical mutations. Live non-linearizable Arctic scans are not used
to claim a transactionally consistent checkpoint.

## Initial scope

The first version supports unsigned integer keys (`u16`, `u32`, `u64`, and
`u128`), which covers the key restriction already imposed by WorkTable's Arctic
backend. Import validates the topology before allocating nodes and reconstructs
the recorded node kinds rather than choosing kinds from occupancy.

The public representation intentionally has no serialization dependency. This
keeps disk-version decisions in WorkTable and avoids imposing a codec on Arctic
users.

## Mutation persistence

The topology API solves checkpoints and recovery; it is not structural CDC.
Incremental persistence should use a monotonically sequenced logical WAL with
`Set` and `Remove` records. The checkpoint remains backend-native, while the WAL
captures mutations that happened after its watermark. Adding stable IDs or CDC
fields to Arctic nodes is specifically excluded unless measurements later prove
that a logical WAL cannot meet the recovery target, because node layout is
cache-sensitive.
