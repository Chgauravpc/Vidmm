"""
Core data model for interval-based video memory.

The central idea of this project in one type.

TEXT memory (the system this is ported from) stamps every fact at a single
POINT in time and assumes it holds forever after ("newest wins"):

    ("user", "WORKING_ON", "GraphQL")  @ 2026-07-01T10:00

VIDEO memory needs facts that occupy an INTERVAL with a real beginning and a
real end, and conflict is decided by OVERLAP, not by succession:

    ("person_1", "HOLDS", "cup")  during [00:04:20, 00:04:55]   conf=0.82

Text is the degenerate special case: a point-stamped fact is an interval
[t, +inf) whose end is merely INFERRED, and "newest wins" is the special case
of "highest-confidence-wins-on-overlap" where all confidences are equal and the
tie-break is start time. `resolve.py` implements the general rule; feeding it
open-ended, equal-confidence intervals reproduces the text behaviour exactly
(see tests/test_resolve.py::test_text_is_degenerate_case).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional

# A time is seconds from the start of the media (float). This is MEDIA time
# (a.k.a. valid time), NOT wall-clock ingest time. The bug we deliberately do
# NOT inherit from the text system is that it stamped facts with
# datetime.now() (ingest time) and used that as if it were valid time.
Seconds = float

# Sentinel for an interval whose end is not yet known / never ends.
OPEN_END: Seconds = float("inf")


class Closure(str, Enum):
    """How the END of an assertion's interval was determined.

    This distinction does not exist in the text system (text only ever has
    INFERRED_END). It is the honest bookkeeping that lets a reviewer see
    *why* an interval ends where it does.
    """

    OBSERVED_END = "observed_end"   # we saw evidence the fact stopped holding (video-only)
    INFERRED_END = "inferred_end"   # end assumed because a conflicting fact took over
    OPEN = "open"                   # still holding at the last frame we processed


@dataclass(frozen=True)
class Interval:
    """A half-open media-time interval [start, end)."""

    start: Seconds
    end: Seconds = OPEN_END
    closure: Closure = Closure.OPEN

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval end {self.end} < start {self.start}")

    @property
    def is_open(self) -> bool:
        return self.end == OPEN_END

    @property
    def duration(self) -> Seconds:
        return self.end - self.start

    def overlaps(self, other: "Interval", gap: Seconds = 0.0) -> bool:
        """True if the two intervals overlap, tolerating a gap of `gap` seconds
        (the occlusion tolerance tau: a fact that briefly disappears and returns
        is treated as one continuous fact rather than two)."""
        return self.start <= other.end + gap and other.start <= self.end + gap


@dataclass(frozen=True)
class Evidence:
    """A citation: which frames/segment made us assert this fact.

    In the text system the `properties` edge field was dead weight. Here it
    becomes load-bearing: every answer must be able to point back to the
    evidence that produced it (the "how do you know?" requirement)."""

    frame_ids: tuple[int, ...] = ()
    source: str = ""            # e.g. "siglip:prompt-bank" or "qwen2vl"
    raw_score: Optional[float] = None


@dataclass(frozen=True)
class Assertion:
    """One fact that held over one interval.

    This is the video generalization of a single temporal edge in the text
    TKG. The (subject, relation, object) triple and the cardinality handling
    are inherited unchanged; the interval + closure + calibrated confidence are
    the new, video-specific machinery.
    """

    subject: str
    relation: str
    object: str
    interval: Interval
    confidence: float = 1.0
    evidence: Evidence = field(default_factory=Evidence)
    # transaction/ingest time — the bi-temporal second axis. Kept separate from
    # media time on purpose; None until persisted.
    ingest_time: Optional[str] = None

    # ---- keys used by the resolver (ported from context_engineer.py) ----
    def group_key(self, exclusive_relations: set[str]) -> tuple:
        """Same cardinality trick as the text resolver:

        single-valued relations conflict on (subject, relation)   -> newest/best wins
        multi-valued relations coexist, keyed on (subject, relation, object)
        """
        if self.relation in exclusive_relations:
            return (self.subject, self.relation)
        return (self.subject, self.relation, self.object)

    def with_interval(self, interval: Interval) -> "Assertion":
        return replace(self, interval=interval)

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "start": self.interval.start,
            "end": self.interval.end,
            "closure": self.interval.closure.value,
            "confidence": self.confidence,
            "evidence": {
                "frame_ids": list(self.evidence.frame_ids),
                "source": self.evidence.source,
                "raw_score": self.evidence.raw_score,
            },
            "ingest_time": self.ingest_time,
        }
