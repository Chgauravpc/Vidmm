"""
Tests for carrying state across gated-out frames.

The regression these exist for is real and came from the first Kaggle run: the
Stage-1 motion gate dropped 52 of 60 frames, the 8 survivors sat 3-12s apart,
tau=2.0 could not bridge them, and every fact became its own 1-second island.
The interval memory had degenerated into the point memory it exists to beat.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video_memory.perception.intervalize import (  # noqa: E402
    carry_forward_gated,
    observations_to_memory,
)
from video_memory.perception.stage2_siglip import Observation  # noqa: E402

# The exact frames the gate kept on the first real run.
KAGGLE_KEPT_TIMESTAMPS = [0.0, 12.0, 15.0, 23.0, 30.0, 34.0, 45.0, 56.0]


def O(fid, ts, rel, obj, conf=0.85, source="siglip"):
    return Observation(frame_id=fid, timestamp=ts, relation=rel, object=obj,
                       confidence=conf, source=source)


def frames_for(timestamps):
    return [(int(t), float(t)) for t in timestamps]


def test_carried_observations_fill_the_gaps():
    kept = frames_for([0.0, 5.0])
    gated = frames_for([1.0, 2.0, 3.0, 4.0])
    obs = [O(0, 0.0, "HOLDS", "a cup"), O(5, 5.0, "HOLDS", "a cup")]

    out = carry_forward_gated(obs, kept=kept, gated=gated)
    times = sorted(o.timestamp for o in out if o.object == "a cup")
    assert times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], times


def test_carried_observations_are_marked_as_carried():
    kept = frames_for([0.0])
    gated = frames_for([1.0, 2.0])
    out = carry_forward_gated([O(0, 0.0, "HOLDS", "a cup")], kept=kept, gated=gated)

    carried = [o for o in out if o.timestamp > 0.0]
    assert carried, "expected carried observations"
    assert all(o.source.endswith(":carried") for o in carried)
    # The originally-seen frame keeps its honest source.
    original = [o for o in out if o.timestamp == 0.0][0]
    assert original.source == "siglip"


def test_gated_frames_before_any_kept_frame_inherit_nothing():
    """Nothing has been observed yet, so there is no state to carry."""
    kept = frames_for([10.0])
    gated = frames_for([0.0, 5.0])
    out = carry_forward_gated([O(10, 10.0, "HOLDS", "a cup")], kept=kept, gated=gated)
    assert all(o.timestamp >= 10.0 for o in out), [o.timestamp for o in out]


def test_state_changes_are_not_carried_backwards():
    """A gated frame inherits the most recent PRIOR kept frame, never a later
    one - otherwise a future observation would rewrite the past."""
    kept = frames_for([0.0, 10.0])
    gated = frames_for([5.0])
    obs = [O(0, 0.0, "HOLDS", "a cup"), O(10, 10.0, "HOLDS", "a phone")]

    out = carry_forward_gated(obs, kept=kept, gated=gated)
    at_five = [o for o in out if o.timestamp == 5.0]
    assert len(at_five) == 1
    assert at_five[0].object == "a cup", at_five[0].object


# --------------------------------------------------------------------------
# The regression itself
# --------------------------------------------------------------------------
def _kaggle_scenario(carry: bool):
    """Reproduce the first real run: 60 frames, only 8 survive the gate."""
    all_ts = [float(i) for i in range(60)]
    kept_ts = KAGGLE_KEPT_TIMESTAMPS
    gated_ts = [t for t in all_ts if t not in kept_ts]

    # The camera wearer holds a cup the whole time; the gate keeps 8 frames.
    obs = [O(int(t), t, "HOLDS", "a cup") for t in kept_ts]

    if carry:
        obs = carry_forward_gated(obs, kept=frames_for(kept_ts), gated=frames_for(gated_ts))

    return observations_to_memory(
        obs, frame_period=1.0, tau=2.0, last_timestamp=59.0 if carry else 56.0
    )


def test_without_carry_forward_memory_shatters_into_points():
    """Characterizes the bug: this is what the first Kaggle pack looked like.

    7 islands, not 8, from 8 kept frames: the 12s and 15s frames are exactly
    3s apart and tau + frame_period = 3.0, so that one pair merges. Every other
    gap (7-12s) is far too wide. A single fact that held for the whole minute is
    represented as 7 disconnected fragments totalling ~9s of the 60s timeline.
    """
    mem = _kaggle_scenario(carry=False)
    holds = [a for a in mem if a.relation == "HOLDS"]
    assert len(holds) == 7, f"expected 7 islands, got {len(holds)}"

    covered = sum(a.interval.duration for a in holds)
    assert covered < 12.0, f"fragments cover {covered}s of 60s"


def test_with_carry_forward_one_continuous_interval():
    """The fix: a fact that held throughout is represented as ONE interval."""
    mem = _kaggle_scenario(carry=True)
    holds = [a for a in mem if a.relation == "HOLDS"]
    assert len(holds) == 1, f"expected 1 interval, got {len(holds)}: " \
                            f"{[(a.interval.start, a.interval.end) for a in holds]}"
    a = holds[0]
    assert a.interval.start == 0.0
    assert a.interval.end >= 59.0, a.interval.end


def test_carry_forward_preserves_real_state_changes():
    """Carrying must not smear over genuine transitions: cup until 30, then
    phone. Two intervals, with the boundary near 30."""
    kept_ts = [0.0, 12.0, 30.0, 45.0]
    all_ts = [float(i) for i in range(60)]
    gated_ts = [t for t in all_ts if t not in kept_ts]

    obs = [
        O(0, 0.0, "HOLDS", "a cup"),
        O(12, 12.0, "HOLDS", "a cup"),
        O(30, 30.0, "HOLDS", "a phone"),
        O(45, 45.0, "HOLDS", "a phone"),
    ]
    obs = carry_forward_gated(obs, kept=frames_for(kept_ts), gated=frames_for(gated_ts))
    mem = observations_to_memory(obs, frame_period=1.0, tau=2.0, last_timestamp=59.0)

    holds = sorted([a for a in mem if a.relation == "HOLDS"],
                   key=lambda x: x.interval.start)
    assert [a.object for a in holds] == ["a cup", "a phone"], [a.object for a in holds]
    assert abs(holds[0].interval.end - 30.0) < 1e-6, holds[0].interval.end


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
