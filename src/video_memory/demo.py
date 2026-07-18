"""
The demo: query a feature pack from the command line. CPU-only.

    python -m video_memory.demo --pack packs/clip
    python -m video_memory.demo --pack packs/clip --query "what was I holding at 4:20"
    python -m video_memory.demo --pack packs/clip --timeline
    python -m video_memory.demo --demo          # synthetic pack, no files needed

`--explain` shows the ranker's per-signal breakdown for each retrieved fact.
"""

from __future__ import annotations

import argparse
import sys

from .answer import answer, fmt_time
from .retriever import Retriever
from .store import IntervalStore, store_from_pack
from .types import Assertion, Closure, Evidence, Interval

SAMPLE_QUERIES = [
    "what was I holding at 0:20",
    "where was I at 1:10",
    "when was I holding a cup",
    "how long was I in the kitchen",
    "what was I doing at 1:30",
]


def synthetic_store() -> IntervalStore:
    """A hand-built memory, so the demo runs with no pack and no GPU."""
    def A(rel, obj, s, e, c, closure=Closure.OBSERVED_END, frames=()):
        return Assertion(
            subject="camera_wearer", relation=rel, object=obj,
            interval=Interval(start=s, end=e, closure=closure), confidence=c,
            evidence=Evidence(frame_ids=frames or tuple(range(int(s), int(e), 5)),
                              source="siglip"),
        )

    store = IntervalStore()
    store.add_all([
        A("LOCATED_IN", "a kitchen", 0.0, 62.0, 0.91),
        A("LOCATED_IN", "an office", 62.0, 180.0, 0.88),
        A("HOLDS", "a cup", 12.0, 44.0, 0.85),
        A("HOLDS", "nothing", 44.0, 70.0, 0.61),
        A("HOLDS", "a phone", 70.0, 96.0, 0.72),
        A("HOLDS", "a pen", 120.0, 180.0, 0.66, closure=Closure.OPEN),
        A("ACTIVITY", "cooking", 5.0, 58.0, 0.80),
        A("ACTIVITY", "typing", 74.0, 178.0, 0.83),
        A("NEAR", "a table", 3.0, 60.0, 0.70),
        A("NEAR", "a sink", 8.0, 40.0, 0.64),
        A("NEAR", "a chair", 63.0, 180.0, 0.69),
    ])
    return store


def print_timeline(store: IntervalStore) -> None:
    print(f"\n{len(store)} assertions\n")
    for rel in store.relations():
        print(f"{rel}")
        for a in store.timeline(rel):
            bar_start = int(a.interval.start / 6)
            span = max(1, int((min(a.interval.end, 180.0) - a.interval.start) / 6))
            bar = " " * bar_start + "#" * span
            print(f"  {fmt_time(a.interval.start)}-{fmt_time(a.interval.end):>12} "
                  f"{a.object:<16} {a.confidence:.2f} |{bar:<30}|")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="Query interval video memory (CPU-only).")
    p.add_argument("--pack", help="path to a feature pack directory")
    p.add_argument("--demo", action="store_true", help="use the built-in synthetic memory")
    p.add_argument("--query", "-q", help="a single question to ask")
    p.add_argument("--timeline", action="store_true", help="print the whole memory")
    p.add_argument("--explain", action="store_true", help="show ranker signal breakdown")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    if args.pack:
        store = store_from_pack(args.pack)
        print(f"loaded pack: {args.pack}")
    elif args.demo:
        store = synthetic_store()
        print("using synthetic demo memory (no pack, no GPU)")
    else:
        p.error("pass --pack PATH or --demo")
        return

    if args.timeline:
        print_timeline(store)
        return

    retriever = Retriever(store)
    queries = [args.query] if args.query else SAMPLE_QUERIES

    for q in queries:
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        print("-" * 72)
        ans = answer(retriever, q, top_k=args.top_k)
        print(ans)
        if args.explain:
            print("\n" + ans.explain())


if __name__ == "__main__":
    sys.exit(main())
