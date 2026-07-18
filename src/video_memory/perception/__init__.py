"""Perception: video frames -> calibrated observations -> intervals.

Everything in this package is GPU-side / ingest-only. The query side of the
system never imports it — it reads feature packs instead.
"""

from .cascade import (
    STAGE_3_CONFIDENCE_THRESHOLD,
    CascadeStats,
    Stage3VLM,
    merge_stage_results,
    motion_scores,
    should_escalate,
)
from .frames import Frame, sample_frames
from .intervalize import build_intervals, observations_to_memory
from .prompt_bank import DEFAULT_BANK, Relation, exclusive_relations, flatten
from .stage2_siglip import Observation, SigLIPTagger

__all__ = [
    "DEFAULT_BANK",
    "STAGE_3_CONFIDENCE_THRESHOLD",
    "CascadeStats",
    "Frame",
    "Observation",
    "Relation",
    "SigLIPTagger",
    "Stage3VLM",
    "build_intervals",
    "exclusive_relations",
    "flatten",
    "merge_stage_results",
    "motion_scores",
    "observations_to_memory",
    "sample_frames",
    "should_escalate",
]
