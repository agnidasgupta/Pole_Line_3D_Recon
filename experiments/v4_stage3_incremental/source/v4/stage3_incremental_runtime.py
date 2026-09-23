#!/usr/bin/env python3
"""Persistent rolling Stage-3 fragment-join state for V4.

This module intentionally optimizes ONLY fragment candidate discovery/scoring.
All downstream Stage-3 logic remains in reconstruct_v4_stage3.py unchanged.

Correctness strategy:
- retain only [S-max_gap, S]
- cache immutable per-fragment geometry and pair scores
- discover only new-new and new-old candidate pairs
- recompute the intervening-fragment guard on every update
- rerun the global two-phase greedy arbitration on every update

The last two bullets deliberately keep window-dependent decisions identical to
batch select_one_to_one_joins().
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

_MODES = ("strict", "detached_bridge")


class Stage3IncrementalJoiner:
    """Rolling fragment candidate graph with batch-equivalent arbitration."""

    def __init__(self, args, stage3_api, max_sequence_gap: Optional[int] = None):
        self.a = args
        self.s3 = stage3_api
        self.max_gap = int(
            getattr(args, "max_span_slices") if max_sequence_gap is None else max_sequence_gap
        )
        self.reset()

    def reset(self):
        self._frag_by_id: Dict[str, dict] = {}
        self._order: List[str] = []
        self._rank: Dict[str, int] = {}
        self._next_rank = 0
        self._slice_of: Dict[str, int] = {}
        self._slice_frags: Dict[int, List[str]] = {}
        self._slice_trees: Dict[int, Tuple[list, cKDTree]] = {}
        # pair -> {mode -> score row without i/j, or None}
        self._scores: Dict[Tuple[str, str], Dict[str, Optional[dict]]] = {}
        self._latest: Optional[int] = None

    @property
    def latest_seq(self) -> Optional[int]:
        return self._latest

    def window_fragments(self) -> List[dict]:
        return [self._frag_by_id[fid] for fid in self._order]

    def window_source_keys(self) -> Tuple[str, ...]:
        return tuple(self._order)

    def add_slice(self, seq: int, frag_rows: Iterable[dict]) -> List[dict]:
        """Add one acquired slice and return accepted joins for the active window."""
        seq = int(seq)
        if self._latest is not None and seq <= self._latest:
            raise ValueError(
                f"Stage3IncrementalJoiner requires increasing arrivals: latest={self._latest}, new={seq}"
            )
        self._evict_before(seq - self.max_gap)
        self._ingest(seq, list(frag_rows))
        self._discover_and_score(seq)
        self._latest = seq
        return self._select_global()

    def bootstrap_window(self, latest_seq: int, frags: Iterable[dict]) -> List[dict]:
        """Rebuild persistent state from a current batch window.

        Used on first invocation, process restart, resume, or any state mismatch.
        This preserves independent Stage-3 replay while enabling incremental speed
        during a persistent in-process realtime session.
        """
        latest_seq = int(latest_seq)
        first_seq = latest_seq - self.max_gap
        rows = [
            dict(f) for f in frags
            if first_seq <= int(f["slice_seq"]) <= latest_seq
        ]
        by_seq: Dict[int, List[dict]] = defaultdict(list)
        for f in rows:
            by_seq[int(f["slice_seq"])].append(f)

        self.reset()
        accepted: List[dict] = []
        for seq in sorted(by_seq):
            accepted = self.add_slice(seq, by_seq[seq])
        # A legal acquired slice may contain zero accepted line fragments.  It
        # still advances the rolling window and must trigger eviction.
        if self._latest is None or self._latest < latest_seq:
            accepted = self.add_slice(latest_seq, [])
        return accepted

    def _evict_before(self, first_seq: int):
        gone = set()
        for seq in sorted([s for s in self._slice_frags if s < int(first_seq)]):
            for fid in self._slice_frags.pop(seq):
                gone.add(fid)
                self._frag_by_id.pop(fid, None)
                self._slice_of.pop(fid, None)
            self._slice_trees.pop(seq, None)

        if not gone:
            return
        self._order = [fid for fid in self._order if fid not in gone]
        # _rank is immutable and intentionally retained; relative order of
        # surviving fragments therefore remains identical after eviction.
        self._scores = {
            pair: scores
            for pair, scores in self._scores.items()
            if pair[0] not in gone and pair[1] not in gone
        }

    def _ingest(self, seq: int, frag_rows: List[dict]):
        rows = [dict(f) for f in frag_rows]
        self.s3._prepare_fragment_runtime_geometry(rows)
        ids = []
        ep_rows = []
        for f in rows:
            fid = str(f["source_key"])
            if fid in self._frag_by_id:
                raise ValueError(f"duplicate Stage3 fragment source_key: {fid}")
            self._frag_by_id[fid] = f
            self._slice_of[fid] = int(seq)
            self._order.append(fid)
            self._rank[fid] = self._next_rank
            self._next_rank += 1
            ids.append(fid)
            ep = f.get("_rt_endpoints")
            if ep is None:
                continue
            ep_rows.append((fid, 0, np.asarray(ep[0], float)))
            ep_rows.append((fid, 1, np.asarray(ep[1], float)))
        self._slice_frags[int(seq)] = ids
        if ep_rows:
            xyz = np.vstack([r[2] for r in ep_rows])
            self._slice_trees[int(seq)] = (ep_rows, cKDTree(xyz))

    def _pair_key(self, ida: str, idb: str) -> Tuple[str, str]:
        return (ida, idb) if self._rank[ida] < self._rank[idb] else (idb, ida)

    def _discover_and_score(self, new_seq: int):
        new = self._slice_trees.get(int(new_seq))
        if new is None:
            return
        new_rows, new_tree = new
        pairs = set()

        # Same-slice pairs.
        radius = self.s3._join_distance_for_mode(self.a, 0, "detached_bridge")
        for pa, pb in new_tree.query_pairs(r=radius):
            ida, idb = new_rows[pa][0], new_rows[pb][0]
            if ida != idb:
                pairs.add(self._pair_key(ida, idb))

        # New slice against each retained old slice.  Detached bridge is the
        # superset distance envelope, exactly as in the batch implementation.
        for old_seq, (old_rows, old_tree) in self._slice_trees.items():
            if int(old_seq) == int(new_seq):
                continue
            gap = abs(int(new_seq) - int(old_seq))
            if gap > self.max_gap:
                continue
            radius = self.s3._join_distance_for_mode(
                self.a, gap, "detached_bridge"
            )
            neighborhoods = old_tree.query_ball_tree(new_tree, r=radius)
            for pa, positions in enumerate(neighborhoods):
                ida = old_rows[pa][0]
                for pb in positions:
                    idb = new_rows[pb][0]
                    if ida != idb:
                        pairs.add(self._pair_key(ida, idb))

        for pair in pairs:
            if pair not in self._scores:
                self._scores[pair] = self._score_pair(pair)

    def _score_pair(self, pair: Tuple[str, str]) -> Dict[str, Optional[dict]]:
        A = self._frag_by_id[pair[0]]
        B = self._frag_by_id[pair[1]]
        out: Dict[str, Optional[dict]] = {}
        for mode in _MODES:
            row = self.s3.best_fragment_join(0, 1, [A, B], self.a, mode=mode)
            out[mode] = None if row is None else {
                k: v for k, v in row.items() if k not in ("i", "j")
            }
        return out

    def _select_global(self) -> List[dict]:
        """Re-run exact current-window arbitration using cached pair scores.

        The intervening-fragment guard is intentionally recomputed every slice.
        It depends on which other fragments are currently in the rolling window,
        so this is the safest minimal optimization and is directly equivalent to
        the production batch implementation.
        """
        frags = self.window_fragments()
        index = {fid: i for i, fid in enumerate(self._order)}
        midpoint_index = self.s3._fragment_midpoint_index(frags)
        endpoint_used = set()
        uf = self.s3.UF(len(frags))
        accepted: List[dict] = []

        for mode in _MODES:
            candidates = []
            for (ida, idb), by_mode in self._scores.items():
                row = by_mode.get(mode)
                if row is None or ida not in index or idb not in index:
                    continue
                i, j = index[ida], index[idb]
                if uf.find(i) == uf.find(j):
                    continue
                c = {"i": i, "j": j, **row}
                if self.s3.candidate_skips_intervening_fragment(
                    c, frags, midpoint_index=midpoint_index
                ):
                    continue
                candidates.append(c)

            # batch select_one_to_one_joins() sorts stably by cost over pair_ids
            # sorted by (i,j); this explicit tie break is equivalent.
            candidates.sort(key=lambda r: (r["cost"], r["i"], r["j"]))
            for c in candidates:
                ka = (c["i"], c["i_end"])
                kb = (c["j"], c["j_end"])
                if ka in endpoint_used or kb in endpoint_used:
                    continue
                if uf.find(c["i"]) == uf.find(c["j"]):
                    continue
                endpoint_used.add(ka)
                endpoint_used.add(kb)
                uf.union(c["i"], c["j"])
                accepted.append(c)
        return accepted


@dataclass
class Stage3Session:
    """Persistent per-session Stage-3 state for the minimal experiment.

    At this stage only fragment joining is incremental.  Pole merging and every
    downstream reconstruction function remain batch/global in the existing
    reconstruct_v4_stage3.py main path.
    """

    args: Any
    group_id: str
    stage3_api: Any

    def __post_init__(self):
        self.group_id = str(self.group_id)
        self.joiner = Stage3IncrementalJoiner(
            self.args,
            self.stage3_api,
            max_sequence_gap=int(self.args.max_span_slices),
        )

    def add_slice(self, seq: int, frags: Iterable[dict], poles=None) -> List[dict]:
        """Add only the current slice's Stage-2 objects.

        ``poles`` is accepted now to freeze the intended Stage3Session API, but
        pole merging remains deliberately unchanged/global in experiment 1.
        """
        return self.joiner.add_slice(int(seq), frags)

    def select_for_window(self, latest_seq: int, frags: Iterable[dict], poles=None) -> List[dict]:
        """Advance from a full current window, rebuilding only when necessary."""
        latest_seq = int(latest_seq)
        frags = list(frags)
        expected = tuple(str(f["source_key"]) for f in frags)

        if self.joiner.latest_seq is None:
            return self.joiner.bootstrap_window(latest_seq, frags)

        # Normal realtime path: ingest only the newest slice. Missing sequence
        # numbers are legal, so latest_seq may jump by more than one.
        if latest_seq > int(self.joiner.latest_seq):
            newest = [f for f in frags if int(f["slice_seq"]) == latest_seq]
            accepted = self.add_slice(latest_seq, newest, poles=poles)
            # If the persistent state missed an earlier call/resume event, rebuild
            # from the authoritative full rolling window rather than guessing.
            if self.joiner.window_source_keys() != expected:
                return self.joiner.bootstrap_window(latest_seq, frags)
            return accepted

        # Repeated/out-of-order main() invocation (resume, verification, direct
        # CLI call): rebuild exact state from the supplied authoritative window.
        return self.joiner.bootstrap_window(latest_seq, frags)
