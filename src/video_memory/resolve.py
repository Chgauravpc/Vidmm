"""
Interval conflict resolution — the crown-jewel port.

This is the direct generalization of `_resolve_conflicts` in the text project
(Context-Engineering Using TKG/context_engineer.py:126-178). That function:

  * grouped facts by a cardinality key
        single-valued -> (subject, relation)
        multi-valued  -> (subject, relation, object)
  * sorted each group by time descending
  * kept the newest, marked the rest superseded

Text facts are points, so "conflict" only ever means "a later fact replaced an
earlier one" — a total order, trivially resolved by taking the max.

Video facts are intervals, so two facts about the same single-valued slot can
OVERLAP: the person cannot be HOLDING both a cup and a phone in the same hand at
the same instant, yet both detectors may fire over overlapping windows. The
generalization is a sweep line: at every instant, the winning fact for a
single-valued slot is the one with the highest confidence among the facts
covering that instant. The losers are trimmed (their intervals shrink or split)
rather than deleted, and the boundary they lose becomes an INFERRED_END.

The tie-break, when confidences are equal, is start time (later wins) — which is
exactly "newest wins". So the text behaviour falls out as the special case where
every interval is open-ended and every confidence is equal.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .types import OPEN_END, Assertion, Closure, Interval, Seconds


def resolve(
    assertions: Iterable[Assertion],
    exclusive_relations: set[str],
    gap: Seconds = 0.0,
    min_duration: Seconds = 0.0,
) -> list[Assertion]:
    """Resolve a bag of raw assertions into a temporally-consistent set.

    Args:
        assertions: raw, possibly-overlapping, possibly-conflicting facts.
        exclusive_relations: relation names that are single-valued (a subject
            can have at most one object at a time). Everything else is
            multi-valued and coexists.
        gap: occlusion tolerance tau (seconds). Same-fact intervals separated by
            a gap <= tau are merged into one continuous interval.
        min_duration: drop resolved fragments shorter than this (denoising).

    Returns:
        A new list of assertions, sorted by (subject, relation, start), in which
        no two single-valued assertions for the same slot overlap.
    """
    groups: dict[tuple, list[Assertion]] = defaultdict(list)
    for a in assertions:
        groups[a.group_key(exclusive_relations)].append(a)

    out: list[Assertion] = []
    for key, group in groups.items():
        single_valued = len(key) == 2  # (subject, relation) vs (subject, relation, object)
        if single_valued:
            out.extend(_resolve_single_valued(group, gap, min_duration))
        else:
            out.extend(_merge_multi_valued(group, gap, min_duration))

    out.sort(key=lambda a: (a.subject, a.relation, a.interval.start))
    return out


def _resolve_single_valued(
    group: list[Assertion], gap: Seconds, min_duration: Seconds
) -> list[Assertion]:
    """Sweep line over one single-valued slot.

    Only one object may hold at a time. Partition the timeline at every interval
    boundary; on each elementary segment, the winner is the covering assertion
    with the highest confidence (ties broken by later start = "newest wins").
    Adjacent segments with the same winning object are coalesced.
    """
    if len(group) == 1:
        return [group[0]]

    # Boundary points of the partition.
    bounds: set[Seconds] = set()
    for a in group:
        bounds.add(a.interval.start)
        bounds.add(a.interval.end)
    ordered = sorted(bounds)

    # For each elementary segment [lo, hi), pick the winning assertion.
    segments: list[tuple[Seconds, Seconds, Assertion]] = []
    for lo, hi in zip(ordered, ordered[1:]):
        if hi <= lo:
            continue
        # Representative point strictly inside [lo, hi). For an open-ended final
        # segment (hi == +inf) any finite point > lo works; the midpoint would
        # be +inf and spuriously fall outside every interval.
        mid = (lo + 1.0) if hi == OPEN_END else lo + (hi - lo) / 2.0
        covering = [a for a in group if a.interval.start <= mid < a.interval.end]
        if not covering:
            continue  # a genuine gap where nothing holds
        winner = max(
            covering,
            key=lambda a: (a.confidence, a.interval.start),
        )
        segments.append((lo, hi, winner))

    # Coalesce adjacent segments won by the same object, honouring the gap
    # tolerance. When a winner is cut short by a rival, its end is INFERRED_END;
    # its natural (observed) end is preserved only if it wins right up to it.
    return _coalesce(segments, gap, min_duration)


def _coalesce(
    segments: list[tuple[Seconds, Seconds, Assertion]],
    gap: Seconds,
    min_duration: Seconds,
) -> list[Assertion]:
    if not segments:
        return []

    resolved: list[Assertion] = []
    cur_lo, cur_hi, cur = segments[0]
    for lo, hi, a in segments[1:]:
        same_object = a.object == cur.object
        contiguous = lo <= cur_hi + gap
        if same_object and contiguous:
            cur_hi = max(cur_hi, hi)
            # keep the higher-confidence evidence for the merged span
            if a.confidence > cur.confidence:
                cur = a
        else:
            resolved.append(_finalize(cur, cur_lo, cur_hi))
            cur_lo, cur_hi, cur = lo, hi, a
    resolved.append(_finalize(cur, cur_lo, cur_hi))

    return [r for r in resolved if r.interval.duration >= min_duration]


def _finalize(a: Assertion, lo: Seconds, hi: Seconds) -> Assertion:
    """Rebuild an assertion over the resolved span [lo, hi), setting closure.

    If the resolved end matches the assertion's original observed end, keep its
    closure (it ended on its own terms). Otherwise it was trimmed by a rival, so
    the end is INFERRED.
    """
    orig = a.interval
    if hi >= orig.end:
        closure = orig.closure
    elif orig.is_open:
        closure = Closure.INFERRED_END
    else:
        closure = Closure.INFERRED_END
    return a.with_interval(Interval(start=lo, end=hi, closure=closure))


def _merge_multi_valued(
    group: list[Assertion], gap: Seconds, min_duration: Seconds
) -> list[Assertion]:
    """Multi-valued slot: the same (subject, relation, object) can recur.

    Different objects coexist (no conflict). We only merge repeated observations
    of the SAME fact whose intervals overlap or are within the gap tolerance,
    into one continuous interval — the union — carrying the max confidence.
    """
    group = sorted(group, key=lambda a: a.interval.start)
    merged: list[Assertion] = []
    cur = group[0]
    for a in group[1:]:
        if a.interval.start <= cur.interval.end + gap:
            new_end = max(cur.interval.end, a.interval.end)
            # OBSERVED_END wins over OPEN/INFERRED when extending
            closure = a.interval.closure if new_end == a.interval.end else cur.interval.closure
            cur = cur.with_interval(
                Interval(start=cur.interval.start, end=new_end, closure=closure)
            )
            if a.confidence > cur.confidence:
                cur = a.with_interval(cur.interval)
        else:
            merged.append(cur)
            cur = a
    merged.append(cur)

    return [m for m in merged if m.interval.duration >= min_duration]
