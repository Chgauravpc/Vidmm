"""Evaluation harness: synthetic ground truth, baselines, metrics, ablations."""

from .baselines import (
    ConfidenceWindowVote,
    IntervalMemory,
    PointMemory,
    WindowVote,
    segments_on_grid,
)
from .metrics import evaluate, make_grid, mean_temporal_iou, state_accuracy
from .synthetic import GroundTruth, make_ground_truth, simulate_detector

__all__ = [
    "ConfidenceWindowVote",
    "GroundTruth",
    "IntervalMemory",
    "PointMemory",
    "WindowVote",
    "evaluate",
    "make_grid",
    "make_ground_truth",
    "mean_temporal_iou",
    "segments_on_grid",
    "simulate_detector",
    "state_accuracy",
]
