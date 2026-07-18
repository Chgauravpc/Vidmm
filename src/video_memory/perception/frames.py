"""
Frame sampling.

Deliberately dumb: uniform sampling at a fixed fps. No shot detection, no
keyframe selection. Those are optimizations; the interval semantics are the
contribution, and a uniform grid makes media time trivially exact
(frame_id -> timestamp is just frame_id / sample_fps).

Uses OpenCV if available, else PyAV. Both are optional — only the ingest side
needs them, and ingest runs offline on a GPU box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Frame:
    frame_id: int          # index into the SAMPLED sequence, not the source video
    timestamp: float       # media time in seconds
    image: np.ndarray      # HxWx3, RGB, uint8


def sample_frames(
    video_path: str,
    sample_fps: float = 1.0,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    """Yield frames at `sample_fps`, carrying exact media timestamps."""
    try:
        return _sample_opencv(video_path, sample_fps, max_frames)
    except ImportError:
        return _sample_pyav(video_path, sample_fps, max_frames)


def _sample_opencv(video_path: str, sample_fps: float, max_frames: int | None) -> Iterator[Frame]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / sample_fps)))

    try:
        src_idx, out_idx = 0, 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if src_idx % stride == 0:
                yield Frame(
                    frame_id=out_idx,
                    timestamp=src_idx / src_fps,
                    image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                )
                out_idx += 1
                if max_frames is not None and out_idx >= max_frames:
                    break
            src_idx += 1
    finally:
        cap.release()


def _sample_pyav(video_path: str, sample_fps: float, max_frames: int | None) -> Iterator[Frame]:
    import av  # noqa: PLC0415

    container = av.open(video_path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate or 30.0)
    stride = max(1, int(round(src_fps / sample_fps)))

    try:
        src_idx, out_idx = 0, 0
        for packet_frame in container.decode(video=0):
            if src_idx % stride == 0:
                yield Frame(
                    frame_id=out_idx,
                    timestamp=src_idx / src_fps,
                    image=packet_frame.to_ndarray(format="rgb24"),
                )
                out_idx += 1
                if max_frames is not None and out_idx >= max_frames:
                    break
            src_idx += 1
    finally:
        container.close()
