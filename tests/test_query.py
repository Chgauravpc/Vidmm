"""
Tests for the CPU query side: store -> retriever -> cited answer.

No GPU, no torch, no models. This is the half of the system a reviewer runs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video_memory.answer import answer, classify_query, fmt_time  # noqa: E402
from video_memory.retriever import (  # noqa: E402
    Retriever,
    expand_relations,
    parse_focus_time,
)
from video_memory.store import IntervalStore  # noqa: E402
from video_memory.types import Assertion, Closure, Evidence, Interval  # noqa: E402


def A(rel, obj, start, end, conf=0.9, closure=Closure.OBSERVED_END, frames=(1, 2, 3)):
    return Assertion(
        subject="camera_wearer",
        relation=rel,
        object=obj,
        interval=Interval(start=start, end=end, closure=closure),
        confidence=conf,
        evidence=Evidence(frame_ids=frames, source="siglip"),
    )


def fixture_store():
    store = IntervalStore()
    store.add_all([
        A("HOLDS", "a cup", 10.0, 40.0, conf=0.85),
        A("HOLDS", "a phone", 60.0, 95.0, conf=0.72),
        A("LOCATED_IN", "a kitchen", 0.0, 50.0, conf=0.91),
        A("LOCATED_IN", "an office", 50.0, 120.0, conf=0.88),
        A("NEAR", "a table", 5.0, 45.0, conf=0.65),
        A("NEAR", "a chair", 5.0, 45.0, conf=0.60),
        A("ACTIVITY", "typing", 55.0, 118.0, conf=0.79),
    ])
    return store


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
def test_at_returns_only_covering_intervals():
    s = fixture_store()
    at20 = {(a.relation, a.object) for a in s.at(20.0)}
    assert ("HOLDS", "a cup") in at20
    assert ("LOCATED_IN", "a kitchen") in at20
    assert ("HOLDS", "a phone") not in at20      # starts at 60
    assert ("LOCATED_IN", "an office") not in at20


def test_at_is_half_open():
    """[start, end): an interval ending at 40 does NOT cover t=40."""
    s = fixture_store()
    assert any(a.object == "a cup" for a in s.at(39.999))
    assert not any(a.object == "a cup" for a in s.at(40.0))


def test_during_window_overlap():
    s = fixture_store()
    got = {(a.relation, a.object) for a in s.during(45.0, 65.0)}
    assert ("LOCATED_IN", "an office") in got
    assert ("HOLDS", "a phone") in got
    assert ("HOLDS", "a cup") not in got          # ended at 40


def test_open_interval_stored_and_queried():
    s = IntervalStore()
    s.add_all([A("HOLDS", "a bag", 10.0, float("inf"), closure=Closure.OPEN)])
    assert len(s.at(10_000.0)) == 1               # +inf compares correctly in SQL
    assert len(s.at(5.0)) == 0


def test_timeline_ordered():
    s = fixture_store()
    tl = s.timeline("LOCATED_IN")
    assert [a.object for a in tl] == ["a kitchen", "an office"]


def test_find_by_object_substring():
    s = fixture_store()
    assert [a.object for a in s.find("cup")] == ["a cup"]


def test_roundtrip_preserves_evidence_and_closure():
    s = fixture_store()
    a = s.at(20.0)[0]
    assert a.evidence.frame_ids == (1, 2, 3)
    assert a.interval.closure == Closure.OBSERVED_END
    assert a.subject == "camera_wearer"


# --------------------------------------------------------------------------
# Query parsing
# --------------------------------------------------------------------------
def test_parse_focus_time_formats():
    assert parse_focus_time("what was I holding at 1:30") == 90.0
    assert parse_focus_time("at 45 seconds") == 45.0
    assert parse_focus_time("at 2 minutes") == 120.0
    assert parse_focus_time("what was I holding") is None


def test_concept_expansion():
    assert "HOLDS" in expand_relations("what was I holding")
    assert "LOCATED_IN" in expand_relations("where was I")
    assert "ACTIVITY" in expand_relations("what was I doing")
    assert expand_relations("xyzzy") == set()      # OOV -> nothing, honestly


def test_query_classification():
    assert classify_query("how long was I in the kitchen") == "duration"
    assert classify_query("when did I hold the cup") == "when"
    assert classify_query("what was I holding at 0:20") == "at_time"
    assert classify_query("tell me about the video") == "general"


# --------------------------------------------------------------------------
# Retrieval + ranking
# --------------------------------------------------------------------------
def test_retrieval_finds_the_right_fact_at_a_time():
    r = Retriever(fixture_store())
    top = r.retrieve("what was I holding at 0:20", top_k=3)
    assert top[0].assertion.relation == "HOLDS"
    assert top[0].assertion.object == "a cup"


def test_signals_are_inspectable():
    r = Retriever(fixture_store())
    top = r.retrieve("what was I holding at 0:20", top_k=1)
    sig = top[0].signals
    assert set(sig) == {"lexical", "temporal", "confidence", "duration", "closure"}
    assert sig["temporal"] == 1.0        # the interval covers the focus time
    assert 0.0 <= top[0].score <= 1.0


def test_temporal_signal_decays_with_distance():
    r = Retriever(fixture_store(), temporal_sigma=30.0)
    near = r.retrieve("what was I holding at 0:45", top_k=10)
    cup = next(s for s in near if s.assertion.object == "a cup")
    phone = next(s for s in near if s.assertion.object == "a phone")
    # t=45 is 5s after cup ended, 15s before phone started -> cup is closer
    assert cup.signals["temporal"] > phone.signals["temporal"]


def test_closure_signal_prefers_observed_ends():
    s = IntervalStore()
    s.add_all([
        A("HOLDS", "a cup", 0.0, 10.0, closure=Closure.OBSERVED_END),
        A("HOLDS", "a pen", 20.0, 30.0, closure=Closure.INFERRED_END),
    ])
    r = Retriever(s)
    got = {x.assertion.object: x.signals["closure"] for x in r.retrieve("holding", top_k=5)}
    assert got["a cup"] > got["a pen"]


def test_at_query_returns_all_covering():
    r = Retriever(fixture_store())
    got = {x.assertion.relation for x in r.at(20.0)}
    assert {"HOLDS", "LOCATED_IN", "NEAR"} <= got


# --------------------------------------------------------------------------
# Answers + citations
# --------------------------------------------------------------------------
def test_answer_at_time_is_cited():
    r = Retriever(fixture_store())
    ans = answer(r, "what was I holding at 0:20")
    assert "a cup" in ans.text
    assert ans.citations, "every answer must carry evidence"
    c = ans.citations[0]
    assert c.frame_ids == (1, 2, 3)
    assert c.start == 10.0 and c.end == 40.0


def test_answer_when_lists_intervals():
    r = Retriever(fixture_store())
    ans = answer(r, "when was I holding a cup")
    assert ans.query_type == "when"
    assert "00:10" in ans.text and "00:40" in ans.text


def test_answer_duration_reports_total():
    r = Retriever(fixture_store())
    ans = answer(r, "how long was I holding a cup")
    assert ans.query_type == "duration"
    assert "30.0s" in ans.text          # 10 -> 40


def test_answer_admits_gaps_rather_than_guessing():
    """t=52 has no HOLDS assertion. The answer must say so and must NOT
    silently extend the cup or phone interval to cover it."""
    r = Retriever(fixture_store())
    ans = answer(r, "what was I holding at 0:52")
    covering = [c for c in ans.citations if c.start <= 52.0 < c.end and c.relation == "HOLDS"]
    assert not covering
    assert "cannot distinguish" in ans.text or "Nothing is asserted" in ans.text


def test_empty_store_answers_honestly():
    r = Retriever(IntervalStore())
    ans = answer(r, "what was I holding at 0:20")
    assert "no assertion" in ans.text.lower()
    assert ans.citations == []


def test_open_interval_duration_is_flagged_as_lower_bound():
    s = IntervalStore()
    s.add_all([A("HOLDS", "a bag", 10.0, float("inf"), closure=Closure.OPEN)])
    ans = answer(Retriever(s), "how long was I holding a bag")
    assert "lower bound" in ans.text


def test_open_closure_with_finite_end_is_still_a_lower_bound():
    """The subtle case: an interval that runs to the last frame has a FINITE
    end (so Interval.is_open is False) but was never observed to stop. Its
    duration is still a lower bound, and the caveat must key off closure."""
    s = IntervalStore()
    s.add_all([A("HOLDS", "a pen", 120.0, 180.0, closure=Closure.OPEN)])
    ans = answer(Retriever(s), "how long was I holding a pen")
    assert "lower bound" in ans.text
    assert "60.0s" in ans.text          # the observed portion is still reported


def test_retrieval_does_not_double_count_multi_branch_hits():
    """A fact reachable via BOTH the relation branch and the term branch must
    appear once. Store calls rebuild Assertion objects per row, so identity-
    based dedup silently double-counts and inflates durations."""
    r = Retriever(fixture_store())
    got = r.retrieve("holding a cup", top_k=10)
    cups = [x for x in got if x.assertion.object == "a cup"]
    assert len(cups) == 1, f"duplicated: {len(cups)}"


def test_explain_exposes_ranking():
    r = Retriever(fixture_store())
    ans = answer(r, "what was I holding at 0:20")
    text = ans.explain()
    assert "lexical=" in text and "temporal=" in text


def test_fmt_time_handles_open():
    assert fmt_time(float("inf")) == "end of video"
    assert fmt_time(90.0) == "01:30.00"


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
