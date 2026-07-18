"""
Ablation A1 - the thesis check.

H1: representing facts as INTERVALS and resolving overlap by CONFIDENCE recovers
    the true state of a noisy stream better than the text system's point-stamped
    "newest wins".

The experiment sweeps two axes and reports all four methods on each cell:

  detector accuracy  how often the per-frame detector is right
  confidence_gap     how much more confident it is when right than when wrong

The second axis is the falsifier. Interval resolution wins overlaps by
confidence, so at confidence_gap=0 - where confidence is pure noise - it should
lose its advantage over the confidence-weighted baseline. If it still "wins"
there, the harness is measuring something other than what it claims.

Run:
    python -m video_memory.eval.ablation_a1
    python -m video_memory.eval.ablation_a1 --seeds 20 --duration 900
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

from .baselines import (
    ConfidenceWindowVote,
    IntervalMemory,
    PointMemory,
    WindowVote,
    segments_on_grid,
)
from .metrics import evaluate, make_grid
from .synthetic import make_ground_truth, simulate_detector

RELATION = "HOLDS"


@dataclass
class Cell:
    accuracy: float
    confidence_gap: float
    results: dict[str, dict[str, float]]


def run_cell(
    detector_accuracy: float,
    confidence_gap: float,
    seeds: int = 10,
    duration: float = 600.0,
    sample_fps: float = 1.0,
    dropout: float = 0.05,
    tau: float = 2.0,
    window: float = 5.0,
) -> dict[str, dict[str, float]]:
    """Average every method over `seeds` independent timelines."""
    per_method: dict[str, list[dict[str, float]]] = {}

    for seed in range(seeds):
        gt = make_ground_truth(duration=duration, relation=RELATION, seed=seed)
        obs = simulate_detector(
            gt,
            sample_fps=sample_fps,
            accuracy=detector_accuracy,
            dropout=dropout,
            confidence_gap=confidence_gap,
            seed=seed,
        )
        grid = make_grid(gt.duration, step=0.5)

        methods = [
            PointMemory(obs),
            WindowVote(obs, window=window),
            ConfidenceWindowVote(obs, window=window),
            IntervalMemory(obs, relation=RELATION,
                           frame_period=1.0 / sample_fps, tau=tau),
        ]

        for m in methods:
            segs = segments_on_grid(m, grid)
            scores = evaluate(gt, m.state_at, segs, grid)
            per_method.setdefault(m.name, []).append(scores)

    summary: dict[str, dict[str, float]] = {}
    for name, runs in per_method.items():
        summary[name] = {
            key: statistics.mean(r[key] for r in runs) for key in runs[0]
        }
        summary[name]["accuracy_stdev"] = (
            statistics.stdev(r["accuracy"] for r in runs) if len(runs) > 1 else 0.0
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation A1: intervals vs points.")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--window", type=float, default=5.0)
    p.add_argument("--json", help="write full results to this path")
    args = p.parse_args()

    accuracies = [0.5, 0.65, 0.8, 0.95]
    gaps = [0.0, 0.15, 0.30]

    print("=" * 92)
    print("ABLATION A1 - does interval memory beat point memory?")
    print(f"  {args.seeds} seeds x {args.duration:.0f}s per cell, "
          f"sampled at {args.fps} fps, tau={args.tau}s, vote window={args.window}s")
    print("=" * 92)

    cells: list[Cell] = []
    for gap in gaps:
        print(f"\n### confidence_gap = {gap:.2f}"
              + ("   <- confidence carries NO signal (the falsifier)" if gap == 0 else ""))
        print(f"{'detector':>9} | {'method':<32} | {'acc':>6} {'sd':>5} | {'tIoU':>6} | {'frag':>6}")
        print("-" * 92)

        for acc in accuracies:
            summary = run_cell(
                detector_accuracy=acc,
                confidence_gap=gap,
                seeds=args.seeds,
                duration=args.duration,
                sample_fps=args.fps,
                tau=args.tau,
                window=args.window,
            )
            cells.append(Cell(accuracy=acc, confidence_gap=gap, results=summary))

            best = max(summary.items(), key=lambda kv: kv[1]["accuracy"])[0]
            for i, (name, s) in enumerate(summary.items()):
                marker = " *" if name == best else "  "
                label = f"{acc:.2f}" if i == 0 else ""
                print(f"{label:>9} | {name:<32} | {s['accuracy']:.3f} "
                      f"{s['accuracy_stdev']:.3f} | {s['temporal_iou']:.3f} | "
                      f"{s['fragmentation']:5.2f}{marker}")
            print("-" * 92)

    _verdict(cells)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                [{"accuracy": c.accuracy, "confidence_gap": c.confidence_gap,
                  "results": c.results} for c in cells],
                f, indent=2,
            )
        print(f"\nwrote {args.json}")


def _verdict(cells: list[Cell]) -> None:
    """State plainly whether H1 survived, including where it did not."""
    ours = "IntervalMemory (ours)"
    point = "PointMemory (text degenerate)"
    hard = "ConfidenceWindowVote"

    beat_point = sum(
        1 for c in cells if c.results[ours]["accuracy"] > c.results[point]["accuracy"]
    )
    beat_hard = sum(
        1 for c in cells if c.results[ours]["accuracy"] > c.results[hard]["accuracy"]
    )
    zero_gap = [c for c in cells if c.confidence_gap == 0.0]
    beat_hard_zero = sum(
        1 for c in zero_gap if c.results[ours]["accuracy"] > c.results[hard]["accuracy"]
    )

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  vs PointMemory (the text degenerate case): won {beat_point}/{len(cells)} cells")
    print(f"  vs ConfidenceWindowVote (the hard baseline): won {beat_hard}/{len(cells)} cells")
    print(f"  at confidence_gap=0 (confidence uninformative): "
          f"won {beat_hard_zero}/{len(zero_gap)} cells")
    print()
    if beat_point == len(cells):
        print("  H1 vs point memory: SUPPORTED in every cell.")
    elif beat_point > len(cells) / 2:
        print(f"  H1 vs point memory: PARTIAL - lost {len(cells) - beat_point} cells.")
    else:
        print("  H1 vs point memory: NOT SUPPORTED.")

    if beat_hard <= len(cells) / 2:
        print("  vs the hard baseline: NOT clearly better. The interval machinery")
        print("  is not buying much over confidence-weighted smoothing on this task.")
    else:
        print(f"  vs the hard baseline: better in {beat_hard}/{len(cells)} cells.")

    if beat_hard_zero > len(zero_gap) / 2:
        print("  WARNING: still winning where confidence is pure noise. Interval")
        print("  resolution has no signal to exploit there, so an advantage in that")
        print("  regime points at a harness artifact, not a real effect.")


if __name__ == "__main__":
    main()
