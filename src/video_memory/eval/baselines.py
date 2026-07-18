"""
Competing memory designs, all answering the same question: what held at time t?

The point of this file is to make H1 falsifiable. It would be easy to beat a
weak baseline and declare the interval machinery vindicated, so the lineup
includes two baselines that are deliberately hard to beat:

  PointMemory            the text system's degenerate case: newest wins.
  WindowVote             majority vote in a sliding window - tests whether the
                         gain is just temporal smoothing.
  ConfidenceWindowVote   confidence-weighted vote in a sliding window - uses
                         BOTH signals interval resolution uses (confidence and
                         temporal locality) but has no interval structure.
  IntervalMemory         this project: intervals + sweep-line resolution.

If ConfidenceWindowVote matches IntervalMemory, then the interval machinery adds
nothing on this task and H1 is false as stated. That outcome is reportable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..perception.intervalize import build_intervals
from ..perception.stage2_siglip import Observation
from ..resolve import resolve


class Memory(Protocol):
    name: str

    def state_at(self, t: float) -> str | None: ...
    def segments(self) -> list[tuple[float, float, str]]: ...


def _segments_from_grid(
    state_fn, grid: Sequence[float]
) -> list[tuple[float, float, str]]:
    """Collapse a sampled state function into contiguous segments."""
    out: list[tuple[float, float, str]] = []
    cur_obj, cur_start = None, None
    for i, t in enumerate(grid):
        obj = state_fn(t)
        if obj != cur_obj:
            if cur_obj is not None and cur_start is not None:
                out.append((cur_start, t, cur_obj))
            cur_obj, cur_start = obj, t
    if cur_obj is not None and cur_start is not None:
        out.append((cur_start, grid[-1], cur_obj))
    return out


# --------------------------------------------------------------------------
@dataclass
class PointMemory:
    """The text system, unchanged: facts are points, newest wins.

    Each observation supersedes the previous one and is assumed to hold until
    the next arrives. Confidence is ignored, exactly as the text system ignored
    it (it hardcoded 0.9 on every edge).
    """

    observations: Sequence[Observation]
    name: str = "PointMemory (text degenerate)"

    def __post_init__(self) -> None:
        self._sorted = sorted(self.observations, key=lambda o: o.timestamp)

    def state_at(self, t: float) -> str | None:
        latest = None
        for o in self._sorted:
            if o.timestamp <= t:
                latest = o
            else:
                break
        return latest.object if latest else None

    def segments(self) -> list[tuple[float, float, str]]:
        out: list[tuple[float, float, str]] = []
        for i, o in enumerate(self._sorted):
            end = self._sorted[i + 1].timestamp if i + 1 < len(self._sorted) else o.timestamp
            if end > o.timestamp:
                out.append((o.timestamp, end, o.object))
        return out


@dataclass
class WindowVote:
    """Majority vote over a sliding window. Tests the 'it's just smoothing' null."""

    observations: Sequence[Observation]
    window: float = 5.0
    name: str = "WindowVote"

    def __post_init__(self) -> None:
        self._sorted = sorted(self.observations, key=lambda o: o.timestamp)

    def _vote(self, t: float, weighted: bool) -> str | None:
        lo, hi = t - self.window / 2, t + self.window / 2
        tally: dict[str, float] = defaultdict(float)
        for o in self._sorted:
            if lo <= o.timestamp <= hi:
                tally[o.object] += o.confidence if weighted else 1.0
        if not tally:
            return None
        return max(tally.items(), key=lambda kv: kv[1])[0]

    def state_at(self, t: float) -> str | None:
        return self._vote(t, weighted=False)

    def segments(self) -> list[tuple[float, float, str]]:
        raise NotImplementedError("use segments_on_grid")


@dataclass
class ConfidenceWindowVote(WindowVote):
    """The hard baseline: uses confidence AND locality, but no intervals."""

    name: str = "ConfidenceWindowVote"

    def state_at(self, t: float) -> str | None:
        return self._vote(t, weighted=True)


@dataclass
class IntervalMemory:
    """This project: build intervals with the gap rule, resolve by confidence."""

    observations: Sequence[Observation]
    relation: str
    frame_period: float = 1.0
    tau: float = 2.0
    min_duration: float = 0.0
    name: str = "IntervalMemory (ours)"

    def __post_init__(self) -> None:
        raw = build_intervals(
            self.observations,
            frame_period=self.frame_period,
            tau=self.tau,
            last_timestamp=None,
        )
        self._resolved = resolve(
            raw,
            exclusive_relations={self.relation},
            gap=self.tau,
            min_duration=self.min_duration,
        )

    @property
    def assertions(self):
        return self._resolved

    def state_at(self, t: float) -> str | None:
        best = None
        for a in self._resolved:
            if a.interval.start <= t < a.interval.end:
                if best is None or a.confidence > best.confidence:
                    best = a
        return best.object if best else None

    def segments(self) -> list[tuple[float, float, str]]:
        return [
            (a.interval.start, a.interval.end, a.object)
            for a in sorted(self._resolved, key=lambda x: x.interval.start)
        ]


def segments_on_grid(mem: Memory, grid: Sequence[float]) -> list[tuple[float, float, str]]:
    """Uniform way to get segments out of any memory, including the vote ones."""
    try:
        return mem.segments()
    except NotImplementedError:
        return _segments_from_grid(mem.state_at, grid)
