"""
Tests for the cascade logic and intervalization that need NO GPU and no models.

These cover everything between "SigLIP produced numbers" and "the memory is
built", so the expensive Kaggle run only has to prove the model wiring works,
not the logic.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from video_memory.pack import read_pack, write_pack  # noqa: E402
from video_memory.perception.cascade import (  # noqa: E402
    STAGE_3_CONFIDENCE_THRESHOLD,
    CascadeStats,
    merge_stage_results,
    should_escalate,
)
from video_memory.perception.intervalize import (  # noqa: E402
    build_intervals,
    observations_to_memory,
)
from video_memory.perception.prompt_bank import (  # noqa: E402
    DEFAULT_BANK,
    exclusive_relations,
    flatten,
)
from video_memory.perception.stage2_siglip import Observation  # noqa: E402
from video_memory.types import Closure  # noqa: E402


def O(fid, ts, rel, obj, conf, source="siglip"):
    return Observation(frame_id=fid, timestamp=ts, relation=rel, object=obj,
                       confidence=conf, source=source)


# --------------------------------------------------------------------------
# Prompt bank
# --------------------------------------------------------------------------
def test_bank_flattens_consistently():
    captions, index = flatten(DEFAULT_BANK)
    assert len(captions) == len(index)
    assert len(captions) == sum(len(r.objects) for r in DEFAULT_BANK)
    assert all("{object}" not in c for c in captions)


def test_exclusive_relations_drive_the_resolver():
    ex = exclusive_relations(DEFAULT_BANK)
    assert "HOLDS" in ex and "LOCATED_IN" in ex
    assert "NEAR" not in ex  # multi-valued: several things can be near you


# --------------------------------------------------------------------------
# Escalation predicate — the structural port from the text system
# --------------------------------------------------------------------------
def test_no_stage2_results_escalates_only_if_important():
    hi, _ = should_escalate([], heuristic_score=0.9)
    lo, _ = should_escalate([], heuristic_score=0.1)
    assert hi is True
    assert lo is False


def test_low_confidence_escalates_with_hint():
    obs = [O(0, 0.0, "HOLDS", "a cup", 0.55), O(0, 0.0, "LOCATED_IN", "a kitchen", 0.60)]
    esc, hint = should_escalate(obs, heuristic_score=0.0)
    assert esc is True
    assert hint == "HOLDS"  # the weakest slot is the one worth asking about


def test_high_confidence_does_not_escalate():
    obs = [O(0, 0.0, "HOLDS", "a cup", 0.95)]
    esc, hint = should_escalate(obs, heuristic_score=1.0)
    assert esc is False and hint is None


def test_threshold_matches_text_system():
    assert STAGE_3_CONFIDENCE_THRESHOLD == 0.7


def test_stage3_disabled_never_escalates():
    obs = [O(0, 0.0, "HOLDS", "a cup", 0.01)]
    esc, _ = should_escalate(obs, heuristic_score=1.0, stage3_enabled=False)
    assert esc is False


# --------------------------------------------------------------------------
# Merge policy
# --------------------------------------------------------------------------
def test_stage3_overrides_its_own_slot_only():
    s2 = [O(0, 0.0, "HOLDS", "a cup", 0.4), O(0, 0.0, "LOCATED_IN", "a kitchen", 0.9)]
    s3 = [O(0, 0.0, "HOLDS", "a phone", 0.85, source="qwen2vl")]
    merged = merge_stage_results(s2, s3)
    holds = [m for m in merged if m.relation == "HOLDS"]
    located = [m for m in merged if m.relation == "LOCATED_IN"]
    assert len(holds) == 1 and holds[0].object == "a phone"
    assert len(located) == 1 and located[0].object == "a kitchen"  # untouched


# --------------------------------------------------------------------------
# Intervalization
# --------------------------------------------------------------------------
def test_consecutive_frames_become_one_interval():
    obs = [O(i, float(i), "HOLDS", "a cup", 0.9) for i in range(5)]
    ivs = build_intervals(obs, frame_period=1.0, tau=2.0, last_timestamp=4.0)
    assert len(ivs) == 1
    a = ivs[0]
    assert a.interval.start == 0.0
    assert a.interval.end == 5.0          # last frame gets its full period
    assert a.evidence.frame_ids == (0, 1, 2, 3, 4)


def test_gap_within_tau_merges_gap_beyond_tau_splits():
    obs = [O(0, 0.0, "HOLDS", "a cup", 0.9), O(2, 2.0, "HOLDS", "a cup", 0.9)]
    assert len(build_intervals(obs, frame_period=1.0, tau=2.0)) == 1

    far = [O(0, 0.0, "HOLDS", "a cup", 0.9), O(20, 20.0, "HOLDS", "a cup", 0.9)]
    assert len(build_intervals(far, frame_period=1.0, tau=2.0)) == 2


def test_run_reaching_end_of_footage_is_open():
    obs = [O(i, float(i), "HOLDS", "a cup", 0.9) for i in range(3)]
    ivs = build_intervals(obs, frame_period=1.0, tau=2.0, last_timestamp=2.0)
    assert ivs[0].interval.closure == Closure.OPEN


def test_run_ending_early_is_observed_end():
    obs = [O(i, float(i), "HOLDS", "a cup", 0.9) for i in range(3)]
    ivs = build_intervals(obs, frame_period=1.0, tau=2.0, last_timestamp=100.0)
    assert ivs[0].interval.closure == Closure.OBSERVED_END


def test_interval_confidence_is_mean_not_peak():
    obs = [O(0, 0.0, "HOLDS", "a cup", 1.0), O(1, 1.0, "HOLDS", "a cup", 0.0)]
    ivs = build_intervals(obs, frame_period=1.0, tau=2.0)
    assert abs(ivs[0].confidence - 0.5) < 1e-9
    assert ivs[0].evidence.raw_score == 1.0  # peak retained as evidence


# --------------------------------------------------------------------------
# End-to-end: observations -> resolved memory
# --------------------------------------------------------------------------
def test_end_to_end_exclusive_conflict_resolved():
    """Cup is confidently held early; phone weakly overlaps. The resolver must
    hand the contested span to the cup."""
    obs = (
        [O(i, float(i), "HOLDS", "a cup", 0.9) for i in range(0, 6)]
        + [O(i, float(i), "HOLDS", "a phone", 0.4) for i in range(3, 9)]
    )
    mem = observations_to_memory(obs, frame_period=1.0, tau=2.0, last_timestamp=8.0)
    holds = sorted([m for m in mem if m.relation == "HOLDS"], key=lambda a: a.interval.start)
    assert [h.object for h in holds] == ["a cup", "a phone"], [h.object for h in holds]
    # No overlap survives on a single-valued slot.
    assert holds[0].interval.end <= holds[1].interval.start


def test_end_to_end_multivalued_coexists():
    obs = (
        [O(i, float(i), "NEAR", "a table", 0.8) for i in range(0, 5)]
        + [O(i, float(i), "NEAR", "a chair", 0.8) for i in range(0, 5)]
    )
    mem = observations_to_memory(obs, frame_period=1.0, tau=2.0, last_timestamp=4.0)
    near = sorted(m.object for m in mem if m.relation == "NEAR")
    assert near == ["a chair", "a table"], near


# --------------------------------------------------------------------------
# Feature pack round-trip (the GPU/CPU boundary)
# --------------------------------------------------------------------------
def test_pack_roundtrip(tmp_dir="_tmp_pack_test"):
    obs = [O(i, float(i), "HOLDS", "a cup", 0.9) for i in range(4)]
    mem = observations_to_memory(obs, frame_period=1.0, tau=2.0, last_timestamp=3.0)
    emb = np.random.rand(4, 8).astype(np.float32)

    write_pack(
        tmp_dir,
        assertions=mem,
        embeddings=emb,
        frames=[(i, float(i)) for i in range(4)],
        observations=obs,
        meta={"video_path": "fake.mp4"},
    )
    loaded = read_pack(tmp_dir)

    assert loaded["meta"]["video_path"] == "fake.mp4"
    assert len(loaded["assertions"]) == len(mem)
    a0, b0 = mem[0], loaded["assertions"][0]
    assert (a0.subject, a0.relation, a0.object) == (b0.subject, b0.relation, b0.object)
    assert a0.interval.start == b0.interval.start
    assert a0.interval.closure == b0.interval.closure
    assert a0.evidence.frame_ids == b0.evidence.frame_ids
    assert np.allclose(loaded["embeddings"], emb)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cascade_stats_escalation_rate():
    s = CascadeStats(n_frames=100, n_gated_out=40, n_stage2=60, n_stage3=9)
    assert abs(s.escalation_rate - 0.15) < 1e-9
    assert s.as_dict()["escalation_rate"] == 0.15


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [OK ] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [XX ] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n  {passed}/{len(tests)} checks behaved as characterized")
    sys.exit(0 if passed == len(tests) else 1)
