"""
Metrics for interval memory.

Three things are measured, because each one alone is gameable:

  state_accuracy       fraction of sampled instants where the reported state is
                       correct. Gameable by a system that fragments wildly but
                       happens to be right pointwise.
  mean_temporal_iou    per-object overlap between predicted and true intervals.
                       Gameable by a system that reports one huge interval.
  fragmentation        predicted segments per true segment. A system scoring
                       well on the first two while emitting 10x the segments is
                       not building memory, it is smoothing noise.

A method has to do well on all three to be doing what this project claims.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .synthetic import GroundTruth


def make_grid(duration: float, step: float = 0.5) -> list[float]:
    n = int(duration / step)
    return [i * step for i in range(n)]


def state_accuracy(gt: GroundTruth, state_fn, grid: Sequence[float]) -> float:
    if not grid:
        return 0.0
    hits = 0
    for t in grid:
        truth = gt.at(t)
        if truth is not None and state_fn(t) == truth:
            hits += 1
    return hits / len(grid)


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def mean_temporal_iou(
    gt: GroundTruth, predicted: Sequence[tuple[float, float, str]]
) -> float:
    """Per-object temporal IoU, averaged over objects present in ground truth.

    Union of the object's true spans vs union of its predicted spans, measured
    by total overlap / (total_true + total_pred - overlap).
    """
    true_by_obj: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for start, end, obj in gt.segments:
        true_by_obj[obj].append((start, end))

    pred_by_obj: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for start, end, obj in predicted:
        pred_by_obj[obj].append((start, end))

    ious: list[float] = []
    for obj, true_spans in true_by_obj.items():
        pred_spans = pred_by_obj.get(obj, [])
        inter = sum(_overlap(t, p) for t in true_spans for p in pred_spans)
        total_true = sum(e - s for s, e in true_spans)
        total_pred = sum(e - s for s, e in pred_spans)
        union = total_true + total_pred - inter
        ious.append(inter / union if union > 0 else 0.0)

    return sum(ious) / len(ious) if ious else 0.0


def fragmentation(
    gt: GroundTruth, predicted: Sequence[tuple[float, float, str]]
) -> float:
    """Predicted segments per true segment. 1.0 is ideal; higher is noisier."""
    if not gt.segments:
        return 0.0
    return len(predicted) / len(gt.segments)


def evaluate(
    gt: GroundTruth,
    state_fn,
    predicted_segments: Sequence[tuple[float, float, str]],
    grid: Sequence[float],
) -> dict[str, float]:
    return {
        "accuracy": state_accuracy(gt, state_fn, grid),
        "temporal_iou": mean_temporal_iou(gt, predicted_segments),
        "fragmentation": fragmentation(gt, predicted_segments),
        "n_segments": float(len(predicted_segments)),
    }
