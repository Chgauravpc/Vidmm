"""
Retrieval: a dual-branch retriever over interval memory.

Ported from the text system's retriever (dual-branch + 5-signal ranker). The
structure survives the port; only what the signals *mean* changes.

    branch A  structured  text: keyword + concept expansion over the graph
                          video: same, over relations/objects in the prompt bank
    branch B  temporal    text: recency window
                          video: interval overlap with the query's focus time

    ranker    5 signals   text: semantic, recency, importance, access, decay
                          video: lexical, temporal, confidence, duration, closure

Two of the text signals are deliberately dropped rather than faked: access-count
and decay were engagement proxies for a chat assistant, and the text system's
own decay was inert anyway (MEMORY_DECAY_ENABLED=False). Inventing video
analogues for them would be cargo-culting.

Every score is returned with its per-signal breakdown. A ranker you cannot
inspect is a ranker you cannot debug, and the citation story requires being able
to say *why* a fact was retrieved, not just that it was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .store import IntervalStore
from .types import Assertion, Closure, Seconds

# ---- signal weights. Tuned by hand, exposed so ablations can zero them out. ----
DEFAULT_WEIGHTS = {
    "lexical": 0.40,
    "temporal": 0.25,
    "confidence": 0.20,
    "duration": 0.10,
    "closure": 0.05,
}

# Concept expansion: query language -> relation. The video analogue of the text
# system's "database -> PostgreSQL" expansion, and it inherits the same hard
# limit: anything outside this map is invisible to branch A.
CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "HOLDS": ("hold", "holding", "held", "carry", "carrying", "hand", "hands",
              "pick", "picked", "grab", "grabbing", "using"),
    "LOCATED_IN": ("where", "room", "place", "location", "located", "inside",
                   "kitchen", "bedroom", "office", "bathroom", "store", "street"),
    "ACTIVITY": ("doing", "did", "activity", "action", "cooking", "walking",
                 "typing", "reading", "eating", "cleaning", "driving", "talking"),
    "NEAR": ("near", "next", "beside", "around", "close", "nearby", "by"),
}

STOPWORDS = {
    "the", "a", "an", "was", "were", "is", "are", "i", "at", "in", "on", "of",
    "to", "what", "when", "did", "do", "does", "am", "and", "or", "my", "me",
    "it", "that", "this", "there", "how", "long", "much",
}


@dataclass
class Scored:
    """An assertion plus a fully inspectable score breakdown."""

    assertion: Assertion
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.signals.items()))
        return f"score={self.score:.3f}  [{parts}]"


# --------------------------------------------------------------------------
# Query parsing
# --------------------------------------------------------------------------
_TIME_PATTERNS = (
    # "at 4:20", "at 04:20"
    (re.compile(r"\b(\d{1,2}):(\d{2})\b"), lambda m: int(m.group(1)) * 60 + int(m.group(2))),
    # "at 90 seconds", "90s"
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b"), lambda m: float(m.group(1))),
    # "at 3 minutes", "3 min"
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b"), lambda m: float(m.group(1)) * 60),
)


def parse_focus_time(query: str) -> Seconds | None:
    """Pull an explicit media time out of the query, if there is one."""
    q = query.lower()
    for pattern, convert in _TIME_PATTERNS:
        m = pattern.search(q)
        if m:
            return float(convert(m))
    return None


def expand_relations(query: str) -> set[str]:
    """Branch A concept expansion: which relations is this query about?"""
    q = query.lower()
    tokens = set(re.findall(r"[a-z]+", q))
    hits = {rel for rel, words in CONCEPT_MAP.items() if tokens & set(words)}
    return hits


def query_terms(query: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", query.lower()) if t not in STOPWORDS and len(t) > 2}


# --------------------------------------------------------------------------
# The ranker
# --------------------------------------------------------------------------
class Retriever:
    def __init__(
        self,
        store: IntervalStore,
        weights: dict[str, float] | None = None,
        temporal_sigma: float = 30.0,
    ) -> None:
        self.store = store
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.temporal_sigma = temporal_sigma
        self._max_duration = self._compute_max_duration()

    def _compute_max_duration(self) -> float:
        durations = [
            a.interval.duration for a in self.store.all() if not a.interval.is_open
        ]
        return max(durations) if durations else 1.0

    # ---- individual signals ----
    def _lexical(self, a: Assertion, terms: set[str], relations: set[str]) -> float:
        score = 0.0
        if a.relation in relations:
            score += 0.6
        obj_tokens = set(re.findall(r"[a-z]+", a.object.lower()))
        if terms & obj_tokens:
            score += 0.4
        return min(1.0, score)

    def _temporal(self, a: Assertion, focus: Seconds | None) -> float:
        """1.0 if the interval contains the focus time, decaying with distance."""
        if focus is None:
            return 0.0
        if a.interval.start <= focus < a.interval.end:
            return 1.0
        distance = (
            a.interval.start - focus if focus < a.interval.start
            else focus - a.interval.end
        )
        return float(max(0.0, 1.0 - distance / self.temporal_sigma))

    def _duration(self, a: Assertion) -> float:
        # An interval whose end was never observed has an unknown true duration,
        # so it gets a neutral score rather than being rewarded or punished for
        # however much of it happened to fall inside the footage.
        if a.interval.is_open or a.interval.closure == Closure.OPEN or self._max_duration <= 0:
            return 0.5
        return min(1.0, a.interval.duration / self._max_duration)

    @staticmethod
    def _closure(a: Assertion) -> float:
        """Evidence quality: we actually saw it end > we deduced it ended."""
        return {
            Closure.OBSERVED_END: 1.0,
            Closure.OPEN: 0.6,
            Closure.INFERRED_END: 0.3,
        }[a.interval.closure]

    # ---- retrieval ----
    def retrieve(self, query: str, top_k: int = 5, focus_time: Seconds | None = None) -> list[Scored]:
        focus = focus_time if focus_time is not None else parse_focus_time(query)
        relations = expand_relations(query)
        terms = query_terms(query)

        # Branch A (structured) + branch B (temporal). Union, then rank — the
        # same shape as the text retriever's dual-branch merge.
        #
        # Dedup by VALUE, not by object identity: every store call rebuilds
        # fresh Assertion objects out of SQL rows, so the same underlying fact
        # arriving via two branches would otherwise be counted twice (and, e.g.,
        # have its duration double-counted in an answer).
        candidates: dict[tuple, Assertion] = {}

        def offer(a: Assertion) -> None:
            candidates[
                (a.subject, a.relation, a.object, a.interval.start, a.interval.end)
            ] = a

        for rel in relations or self.store.relations():
            for a in self.store.timeline(rel):
                offer(a)
        if focus is not None:
            for a in self.store.at(focus):
                offer(a)
        for term in terms:
            for a in self.store.find(term):
                offer(a)

        scored: list[Scored] = []
        for a in candidates.values():
            signals = {
                "lexical": self._lexical(a, terms, relations),
                "temporal": self._temporal(a, focus),
                "confidence": float(a.confidence),
                "duration": self._duration(a),
                "closure": self._closure(a),
            }
            total = sum(self.weights.get(k, 0.0) * v for k, v in signals.items())
            scored.append(Scored(assertion=a, score=total, signals=signals))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def at(self, t: Seconds, top_k: int = 10) -> list[Scored]:
        """Direct temporal query: what was true at t, ranked by confidence."""
        return [
            Scored(
                assertion=a,
                score=a.confidence,
                signals={"confidence": a.confidence},
            )
            for a in self.store.at(t)[:top_k]
        ]
