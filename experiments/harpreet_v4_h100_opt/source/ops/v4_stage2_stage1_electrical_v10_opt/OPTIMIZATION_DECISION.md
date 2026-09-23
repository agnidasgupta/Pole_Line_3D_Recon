# V10 Stage-2 optimization decision

Decision: freeze `opt1-fix2` as the final experimental implementation. No further
optimization is accepted without a new full 30-session equivalence run.

## Accepted savings

| Block | Before | After | Quality argument |
|---|---:|---:|---|
| Disconnected bridge candidates | `O(F^2)` fragment pairs plus support sampling | `O(1)` empty result | The strict contract forbids every disconnected bridge; candidates never affect geometry. |
| Pole-contact search | repeated scan of pole-support voxels per endpoint | `O(P)` hash build plus bounded neighboring-cell lookup per endpoint | Candidate order is preserved and anchors are unchanged. |
| Full run | 2,065.755 ms/slice weighted Stage-2 compute | 233.851 ms/slice | 3,738 slices passed byte-exact electrical output and bounded production-pole equivalence. |

`F` is the number of disconnected line fragments and `P` the number of inferred
pole-support voxels.

## Remaining blocks

| Weighted mean | Time per slice | Why it is retained |
|---|---:|---|
| Production components | 37.153 ms | Determines pole candidates and component identities. |
| Production refiner/parameterization | 122.485 ms | Determines accepted poles and serialized pole geometry. |
| Line connected components | 3.893 ms | Linear 26-neighbor graph construction is already near-optimal. |
| Line graph tracing | 17.751 ms | Path identity and turn splitting are output-defining. |
| Line geometry/support validation | 48.110 ms | Enforces the central no-jump/no-synthetic-voxel quality invariant. |
| Stage-2 output writes | 30.251 ms | Atomic CSV output and existing directory/schema contracts are required. |

Potential changes such as KD-tree assignment, alternative shortest-path traversal,
dropping production line processing, parallel slice processing, or different output
formats may be faster, but each can alter tie-breaking, floating-point serialization,
memory pressure, component identity, or the required file layout. They remain
unaccepted experiments. A previous pole-only production shortcut already failed the
equivalence gate, confirming that this boundary is material.

The single-slice `EstCharlestontoIslandPondTrans2026/session0` timing is an outlier
and is not enough evidence to change algorithms. Any future work should profile
that slice separately, then repeat all 3,738-slice gates before promotion.
