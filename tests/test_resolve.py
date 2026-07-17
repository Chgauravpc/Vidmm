"""
Test battery for interval resolution.

Written in the same spirit as the text project's test_battery.py: it
characterizes both the intended behaviour AND the known limitations, so a
reviewer can see exactly what the resolver does and does not promise.

Run:  python -m pytest tests/ -v      (or)  python tests/test_resolve.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video_memory import Assertion, Closure, Interval, resolve  # noqa: E402
from video_memory.types import OPEN_END  # noqa: E402

# In the text system these were the single-valued ("exclusive") relations.
EXCLUSIVE = {"WORKING_ON", "HOLDS", "LOCATED_IN"}


def A(subj, rel, obj, start, end=OPEN_END, conf=1.0, closure=Closure.OBSERVED_END):
    return Assertion(
        subject=subj,
        relation=rel,
        object=obj,
        interval=Interval(start=start, end=end, closure=closure),
        confidence=conf,
    )


# --------------------------------------------------------------------------
# 1. THE THESIS: text "newest wins" is the degenerate case of interval resolve
# --------------------------------------------------------------------------
def test_text_is_degenerate_case():
    """Point-stamped facts = open-ended intervals with equal confidence.
    With equal confidence the tie-break is start time, i.e. 'newest wins'.
    A single-valued relation should collapse to ONLY the latest object —
    exactly the text TKG's temporal-supersession behaviour (its test 3)."""
    facts = [
        A("user", "WORKING_ON", "REST",    start=0.0, end=OPEN_END, conf=0.9),
        A("user", "WORKING_ON", "API",     start=10.0, end=OPEN_END, conf=0.9),
        A("user", "WORKING_ON", "GraphQL", start=20.0, end=OPEN_END, conf=0.9),
    ]
    out = resolve(facts, EXCLUSIVE)
    working_on = [a.object for a in out if a.relation == "WORKING_ON"]
    # GraphQL took over at t=20 and never ends; the earlier two were trimmed.
    assert working_on[-1] == "GraphQL", working_on
    # The final open segment is GraphQL alone.
    final = [a for a in out if a.interval.is_open]
    assert len(final) == 1 and final[0].object == "GraphQL", final


# --------------------------------------------------------------------------
# 2. THE VIDEO CASE: overlap resolved by confidence, not recency
# --------------------------------------------------------------------------
def test_confidence_wins_on_overlap():
    """Two single-valued facts overlap. The lower-confidence one is trimmed;
    the higher-confidence one keeps the contested span. This is the behaviour
    text CANNOT express (text would just take the newer one)."""
    facts = [
        A("p1", "HOLDS", "cup",   start=0.0, end=10.0, conf=0.9),
        A("p1", "HOLDS", "phone", start=5.0, end=15.0, conf=0.5),  # newer but weaker
    ]
    out = resolve(facts, EXCLUSIVE)
    # In [5,10] both cover it; cup (0.9) beats phone (0.5) despite phone starting later.
    cup = next(a for a in out if a.object == "cup")
    phone = next(a for a in out if a.object == "phone")
    assert cup.interval.start == 0.0 and cup.interval.end == 10.0, cup.interval
    assert phone.interval.start == 10.0 and phone.interval.end == 15.0, phone.interval


def test_trimmed_end_is_inferred():
    """When a fact is cut short by a rival, its new end is INFERRED, not
    OBSERVED — we did not see it stop, we deduced it stopped."""
    facts = [
        A("p1", "HOLDS", "cup",   start=0.0, end=10.0, conf=0.5),
        A("p1", "HOLDS", "phone", start=5.0, end=15.0, conf=0.9),
    ]
    out = resolve(facts, EXCLUSIVE)
    cup = next(a for a in out if a.object == "cup")
    assert cup.interval.end == 5.0, cup.interval
    assert cup.interval.closure == Closure.INFERRED_END, cup.interval.closure


# --------------------------------------------------------------------------
# 3. MULTI-VALUED coexistence (inherited from text test 4)
# --------------------------------------------------------------------------
def test_multivalued_coexist():
    """Non-exclusive relations do not conflict; distinct objects all survive."""
    facts = [
        A("p1", "NEAR", "table", start=0.0, end=10.0),
        A("p1", "NEAR", "chair", start=0.0, end=10.0),
    ]
    out = resolve(facts, EXCLUSIVE)  # NEAR is not exclusive
    objs = sorted(a.object for a in out)
    assert objs == ["chair", "table"], objs


def test_multivalued_merge_with_gap():
    """The same fact seen twice with a short occlusion gap merges into one
    continuous interval when the gap <= tau."""
    facts = [
        A("p1", "NEAR", "table", start=0.0, end=4.0),
        A("p1", "NEAR", "table", start=5.0, end=10.0),  # 1s gap
    ]
    out = resolve(facts, EXCLUSIVE, gap=2.0)
    tables = [a for a in out if a.object == "table"]
    assert len(tables) == 1, tables
    assert tables[0].interval.start == 0.0 and tables[0].interval.end == 10.0, tables[0].interval


def test_gap_beyond_tau_stays_split():
    """A gap larger than tau is a real absence: two separate intervals."""
    facts = [
        A("p1", "NEAR", "table", start=0.0, end=4.0),
        A("p1", "NEAR", "table", start=9.0, end=12.0),  # 5s gap
    ]
    out = resolve(facts, EXCLUSIVE, gap=2.0)
    tables = [a for a in out if a.object == "table"]
    assert len(tables) == 2, tables


# --------------------------------------------------------------------------
# 4. min_duration denoising
# --------------------------------------------------------------------------
def test_min_duration_drops_fragments():
    facts = [
        A("p1", "HOLDS", "cup",   start=0.0, end=10.0, conf=0.9),
        A("p1", "HOLDS", "phone", start=4.9, end=5.0, conf=0.99),  # tiny high-conf blip
    ]
    out = resolve(facts, EXCLUSIVE, min_duration=0.5)
    objs = [a.object for a in out]
    assert "phone" not in objs, objs


# --------------------------------------------------------------------------
# 5. LIMITATION PROBES (documenting real behaviour, not bugs)
# --------------------------------------------------------------------------
def test_limitation_cannot_distinguish_absence_from_nondetection():
    """A genuine gap where nothing holds looks identical to 'the detector
    missed it'. The resolver emits no assertion for the gap; it does NOT know
    whether the fact was truly absent or merely unseen. Named, not fixed."""
    facts = [
        A("p1", "HOLDS", "cup", start=0.0, end=4.0, conf=0.9),
        A("p1", "HOLDS", "cup", start=8.0, end=12.0, conf=0.9),
    ]
    out = resolve(facts, EXCLUSIVE, gap=1.0)  # gap 4s > tau 1s
    holds = [a for a in out if a.relation == "HOLDS"]
    # Two intervals, nothing asserted in [4,8] — could be empty hand OR occlusion.
    assert len(holds) == 2, holds
    assert all(not (h.interval.start < 6.0 < h.interval.end) for h in holds)


def test_limitation_equal_confidence_ties_break_by_recency_only():
    """When confidences are exactly equal (the text case), the ONLY signal is
    recency. There is no principled winner between two equally-confident
    simultaneous facts — later start wins by convention, same as text."""
    facts = [
        A("p1", "HOLDS", "cup",   start=0.0, end=10.0, conf=0.8),
        A("p1", "HOLDS", "phone", start=0.0, end=10.0, conf=0.8),
    ]
    out = resolve(facts, EXCLUSIVE)
    # identical intervals + equal confidence -> exactly one winner over [0,10]
    holds = [a for a in out if a.relation == "HOLDS"]
    assert len(holds) == 1, holds


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
