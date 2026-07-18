"""
Feature packs — the GPU/CPU boundary.

Ingest is the only part of this system that needs a GPU, and it is a pure
function of (video, prompt bank, model versions). So we run it once, offline, on
Kaggle/Colab, and freeze the result into a *feature pack*: a small directory of
plain files that the query side reads with numpy and the standard library alone.

    pack/
      meta.json           what produced this, and with what settings
      assertions.jsonl    the resolved interval memory (the actual output)
      observations.jsonl  raw per-frame readings, kept for ablations
      embeddings.npy      (n_frames, d) SigLIP frame embeddings, for retrieval
      frames.jsonl        frame_id -> timestamp

Packs are small enough to commit, which is the point: a reviewer clones the repo
and runs the demo with no GPU, no dataset download, and no model weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from .types import Assertion, Closure, Evidence, Interval

PACK_VERSION = 1


def write_pack(
    path: str,
    assertions: Sequence[Assertion],
    embeddings: "np.ndarray",
    frames: Sequence[tuple[int, float]],
    observations: Sequence[Any] = (),
    meta: dict | None = None,
) -> str:
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "assertions.jsonl"), "w", encoding="utf-8") as f:
        for a in assertions:
            f.write(json.dumps(a.as_dict()) + "\n")

    with open(os.path.join(path, "observations.jsonl"), "w", encoding="utf-8") as f:
        for o in observations:
            f.write(json.dumps(asdict(o) if hasattr(o, "__dataclass_fields__") else o) + "\n")

    with open(os.path.join(path, "frames.jsonl"), "w", encoding="utf-8") as f:
        for frame_id, ts in frames:
            f.write(json.dumps({"frame_id": frame_id, "timestamp": ts}) + "\n")

    np.save(os.path.join(path, "embeddings.npy"), embeddings.astype(np.float32))

    full_meta = {"pack_version": PACK_VERSION, "n_assertions": len(assertions)}
    full_meta.update(meta or {})
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)

    return path


def read_pack(path: str) -> dict[str, Any]:
    """Load a pack. numpy + stdlib only — no torch, no transformers, no GPU."""
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    assertions: list[Assertion] = []
    with open(os.path.join(path, "assertions.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                assertions.append(_assertion_from_dict(json.loads(line)))

    frames: list[tuple[int, float]] = []
    frames_path = os.path.join(path, "frames.jsonl")
    if os.path.exists(frames_path):
        with open(frames_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    frames.append((d["frame_id"], d["timestamp"]))

    emb_path = os.path.join(path, "embeddings.npy")
    embeddings = np.load(emb_path) if os.path.exists(emb_path) else np.zeros((0, 0), dtype=np.float32)

    return {
        "meta": meta,
        "assertions": assertions,
        "frames": frames,
        "embeddings": embeddings,
    }


def _assertion_from_dict(d: dict) -> Assertion:
    ev = d.get("evidence") or {}
    return Assertion(
        subject=d["subject"],
        relation=d["relation"],
        object=d["object"],
        interval=Interval(
            start=float(d["start"]),
            end=float(d["end"]),
            closure=Closure(d.get("closure", Closure.OPEN.value)),
        ),
        confidence=float(d.get("confidence", 1.0)),
        evidence=Evidence(
            frame_ids=tuple(ev.get("frame_ids", ())),
            source=ev.get("source", ""),
            raw_score=ev.get("raw_score"),
        ),
        ingest_time=d.get("ingest_time"),
    )
