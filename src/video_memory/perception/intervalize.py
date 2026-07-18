"""
Per-frame observations -> intervals.

This is the step where video stops looking like text. Stage 2 gives an
independent reading at every sampled frame; what memory needs is "the person
held the cup from 4:20 to 4:55". We build runs of consecutive same-value
observations, tolerating gaps of up to tau seconds (the occlusion tolerance),
then hand the resulting intervals to the sweep-line resolver.

The gap rule is deliberately used INSTEAD of an object tracker. A tracker would
be more accurate and would also be a second research project; tau is one
scalar, it is tunable, and its failure modes are easy to characterize.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..types import Assertion, Closure, Evidence, Interval
from .prompt_bank import DEFAULT_BANK, Relation, exclusive_relations
from .stage2_siglip import Observation

# Default occlusion tolerance in seconds.
DEFAULT_TAU = 2.0
# Default subject for egocentric footage: everything is about the camera wearer.
DEFAULT_SUBJECT = "camera_wearer"


def build_intervals(
    observations: Sequence[Observation],
    frame_period: float,
    tau: float = DEFAULT_TAU,
    subject: str = DEFAULT_SUBJECT,
    last_timestamp: float | None = None,
    min_confidence: float = 0.0,
) -> list[Assertion]:
    """Group observations into runs and emit one Assertion per run.

    Args:
        frame_period: seconds between sampled frames (1 / sample_fps). Used to
            give the final frame of a run its full duration, so an interval
            covers the time the frame represents rather than ending on its
            instant.
        tau: merge runs separated by a gap no larger than this.
        last_timestamp: media time of the final processed frame. A run reaching
            it is OPEN (still holding when we ran out of video) rather than
            OBSERVED_END.
    """
    kept = [o for o in observations if o.confidence >= min_confidence]
    runs: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for o in kept:
        runs[(o.relation, o.object)].append(o)

    assertions: list[Assertion] = []
    for (relation, obj), obs in runs.items():
        obs = sorted(obs, key=lambda o: o.timestamp)
        current: list[Observation] = [obs[0]]

        for prev, nxt in zip(obs, obs[1:]):
            if nxt.timestamp - prev.timestamp <= tau + frame_period:
                current.append(nxt)
            else:
                assertions.append(
                    _emit(current, relation, obj, subject, frame_period, last_timestamp)
                )
                current = [nxt]
        assertions.append(
            _emit(current, relation, obj, subject, frame_period, last_timestamp)
        )

    return assertions


def _emit(
    run: list[Observation],
    relation: str,
    obj: str,
    subject: str,
    frame_period: float,
    last_timestamp: float | None,
) -> Assertion:
    start = run[0].timestamp
    end = run[-1].timestamp + frame_period

    # If the run runs to the end of the footage, we never saw it stop.
    if last_timestamp is not None and run[-1].timestamp >= last_timestamp - 1e-9:
        closure = Closure.OPEN
    else:
        closure = Closure.OBSERVED_END

    # Confidence of an interval is the mean of its evidence, not the max: a run
    # that was strong once and weak for a minute should not inherit the peak.
    confidence = sum(o.confidence for o in run) / len(run)
    sources = {o.source for o in run}

    return Assertion(
        subject=subject,
        relation=relation,
        object=obj,
        interval=Interval(start=start, end=end, closure=closure),
        confidence=confidence,
        evidence=Evidence(
            frame_ids=tuple(o.frame_id for o in run),
            source="+".join(sorted(sources)),
            raw_score=max(o.confidence for o in run),
        ),
    )


def observations_to_memory(
    observations: Sequence[Observation],
    frame_period: float,
    tau: float = DEFAULT_TAU,
    bank: tuple[Relation, ...] = DEFAULT_BANK,
    subject: str = DEFAULT_SUBJECT,
    last_timestamp: float | None = None,
    min_duration: float = 0.0,
) -> list[Assertion]:
    """Full path: observations -> intervals -> temporally-consistent memory."""
    from ..resolve import resolve  # local import keeps this module import-light

    raw = build_intervals(
        observations,
        frame_period=frame_period,
        tau=tau,
        subject=subject,
        last_timestamp=last_timestamp,
    )
    return resolve(
        raw,
        exclusive_relations=exclusive_relations(bank),
        gap=tau,
        min_duration=min_duration,
    )
