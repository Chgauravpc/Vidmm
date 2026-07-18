"""
Synthetic ground truth + a simulated detector.

Why synthetic. The thesis claim ("interval memory beats point memory") needs a
setting where the true state at every instant is known exactly. Real video gives
you neither exact ground truth nor a knob to turn on detector noise. So the
thesis is tested here, in a controlled setting, and the real-video runs measure
something different: whether the pipeline works at all on real footage.

Nothing in this module is a claim about real video. It is a claim about the
resolution algorithm, under a stated noise model.

The critical parameter is `confidence_gap`. Interval resolution wins overlaps by
confidence, so it can only beat recency if confidence carries signal about
correctness. Setting confidence_gap=0 makes confidence pure noise and should
erase the advantage. An experiment that cannot fail is not an experiment, so
that regime is swept explicitly rather than avoided.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..perception.stage2_siglip import Observation


@dataclass
class GroundTruth:
    """A single-valued relation's true timeline: contiguous, non-overlapping."""

    relation: str
    segments: list[tuple[float, float, str]] = field(default_factory=list)

    def at(self, t: float) -> str | None:
        for start, end, obj in self.segments:
            if start <= t < end:
                return obj
        return None

    @property
    def duration(self) -> float:
        return self.segments[-1][1] if self.segments else 0.0

    def objects(self) -> list[str]:
        return sorted({o for _, _, o in self.segments})


def make_ground_truth(
    duration: float = 600.0,
    relation: str = "HOLDS",
    objects: tuple[str, ...] = ("a cup", "a phone", "a book", "a bottle", "nothing"),
    mean_segment: float = 25.0,
    min_segment: float = 5.0,
    seed: int = 0,
) -> GroundTruth:
    """A piecewise-constant timeline: the state changes, then holds for a while.

    Consecutive segments never repeat the same object - a repeat would be one
    longer segment, and would quietly inflate every system's score.
    """
    rng = random.Random(seed)
    segments: list[tuple[float, float, str]] = []
    t = 0.0
    previous: str | None = None

    while t < duration:
        choices = [o for o in objects if o != previous]
        obj = rng.choice(choices)
        length = max(min_segment, rng.expovariate(1.0 / mean_segment))
        end = min(duration, t + length)
        segments.append((t, end, obj))
        previous = obj
        t = end

    return GroundTruth(relation=relation, segments=segments)


def simulate_detector(
    gt: GroundTruth,
    sample_fps: float = 1.0,
    accuracy: float = 0.75,
    dropout: float = 0.05,
    confidence_gap: float = 0.25,
    base_confidence: float = 0.55,
    confidence_noise: float = 0.08,
    seed: int = 0,
) -> list[Observation]:
    """Emit noisy per-frame observations from the true timeline.

    Args:
        accuracy: probability a frame reports the true object.
        dropout: probability a frame reports nothing at all (occlusion).
        confidence_gap: how much higher confidence is when the detector is
            RIGHT than when it is wrong. This is the signal interval resolution
            exploits. Zero means confidence is uninformative.
        base_confidence: mean confidence of a WRONG detection.

    The model is deliberately simple and stated, so the result is interpretable:
    errors are independent across frames. Real detectors make correlated errors
    (they stay wrong for a while), which is harder for every method here - noted
    as a threat to validity rather than modelled.
    """
    rng = random.Random(seed)
    period = 1.0 / sample_fps
    all_objects = gt.objects()

    observations: list[Observation] = []
    frame_id = 0
    t = 0.0
    while t < gt.duration:
        truth = gt.at(t)
        if truth is not None and rng.random() >= dropout:
            correct = rng.random() < accuracy
            if correct:
                obj = truth
                mean_conf = base_confidence + confidence_gap
            else:
                wrong = [o for o in all_objects if o != truth]
                obj = rng.choice(wrong) if wrong else truth
                mean_conf = base_confidence

            conf = min(0.99, max(0.01, rng.gauss(mean_conf, confidence_noise)))
            observations.append(
                Observation(
                    frame_id=frame_id,
                    timestamp=t,
                    relation=gt.relation,
                    object=obj,
                    confidence=conf,
                )
            )
        frame_id += 1
        t += period

    return observations
