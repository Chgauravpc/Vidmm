"""
Ingest driver: video file -> feature pack.

This is the only entry point that needs a GPU. Run it once on Kaggle/Colab per
video; commit the resulting pack; everything downstream is CPU-only.

    python -m video_memory.ingest --video clip.mp4 --out packs/clip --fps 1.0

Design note: Stage 3 is loaded LAZILY. If no frame escalates, the VLM is never
downloaded or instantiated — which is the cascade's efficiency claim made
literal rather than merely measured.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .pack import write_pack
from .perception.cascade import (
    MOTION_GATE,
    STAGE_3_CONFIDENCE_THRESHOLD,
    CascadeStats,
    Stage3VLM,
    merge_stage_results,
    motion_scores,
    should_escalate,
)
from .perception.frames import sample_frames
from .perception.intervalize import DEFAULT_TAU, observations_to_memory
from .perception.prompt_bank import DEFAULT_BANK
from .perception.stage2_siglip import DEFAULT_MODEL, SigLIPTagger


def ingest_video(
    video_path: str,
    out_path: str,
    sample_fps: float = 1.0,
    tau: float = DEFAULT_TAU,
    max_frames: int | None = None,
    stage3_enabled: bool = True,
    motion_gate: float = MOTION_GATE,
    siglip_model: str = DEFAULT_MODEL,
    vlm_model: str = Stage3VLM.DEFAULT_MODEL,
    min_duration: float = 0.0,
    device: str | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    stats = CascadeStats()
    frame_period = 1.0 / sample_fps

    # ---- decode ----
    frames = list(sample_frames(video_path, sample_fps=sample_fps, max_frames=max_frames))
    stats.n_frames = len(frames)
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")

    # ---- Stage 1: cheap motion gate ----
    scores = motion_scores(frames)
    keep = [i for i, s in enumerate(scores) if s >= motion_gate]
    if not keep:
        keep = [0]
    stats.n_gated_out = len(frames) - len(keep)
    kept_frames = [frames[i] for i in keep]
    kept_scores = [scores[i] for i in keep]
    stats.n_stage2 = len(kept_frames)

    # ---- Stage 2: SigLIP ----
    tagger = SigLIPTagger(model_name=siglip_model, bank=DEFAULT_BANK, device=device)
    observations, embeddings, _ = tagger.tag(kept_frames)

    per_frame: dict[int, list] = {f.frame_id: [] for f in kept_frames}
    for o in observations:
        per_frame.setdefault(o.frame_id, []).append(o)

    # ---- Stage 3: escalate only where needed ----
    vlm: Stage3VLM | None = None
    stage3_all: list = []
    for frame, heuristic in zip(kept_frames, kept_scores):
        obs = per_frame.get(frame.frame_id, [])
        escalate, hint = should_escalate(
            obs, heuristic_score=heuristic, stage3_enabled=stage3_enabled,
            threshold=STAGE_3_CONFIDENCE_THRESHOLD,
        )
        if not escalate:
            continue
        if vlm is None:
            vlm = Stage3VLM(model_name=vlm_model, bank=DEFAULT_BANK, device=device)
        got = vlm.extract(frame, hint=hint)
        if got:
            stage3_all.extend(got)
            stats.n_stage3 += 1
        else:
            stats.stage3_failures += 1

    merged = merge_stage_results(observations, stage3_all)

    # ---- intervalize + resolve ----
    last_ts = kept_frames[-1].timestamp
    assertions = observations_to_memory(
        merged,
        frame_period=frame_period,
        tau=tau,
        bank=DEFAULT_BANK,
        last_timestamp=last_ts,
        min_duration=min_duration,
    )

    elapsed = time.time() - t0
    duration = frames[-1].timestamp + frame_period
    meta = {
        "video_path": video_path,
        "sample_fps": sample_fps,
        "tau": tau,
        "min_duration": min_duration,
        "siglip_model": siglip_model,
        "vlm_model": vlm_model if stage3_enabled else None,
        "stage3_enabled": stage3_enabled,
        "motion_gate": motion_gate,
        "video_duration_s": duration,
        "ingest_seconds": round(elapsed, 2),
        "x_realtime": round(duration / elapsed, 2) if elapsed else None,
        "cascade": stats.as_dict(),
    }

    write_pack(
        out_path,
        assertions=assertions,
        embeddings=embeddings,
        frames=[(f.frame_id, f.timestamp) for f in kept_frames],
        observations=merged,
        meta=meta,
    )
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a video into a feature pack.")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=1.0, help="sampling rate")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU, help="occlusion tolerance (s)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-stage3", action="store_true", help="ablation: SigLIP only")
    p.add_argument("--min-duration", type=float, default=0.0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    meta = ingest_video(
        video_path=args.video,
        out_path=args.out,
        sample_fps=args.fps,
        tau=args.tau,
        max_frames=args.max_frames,
        stage3_enabled=not args.no_stage3,
        min_duration=args.min_duration,
        device=args.device,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
