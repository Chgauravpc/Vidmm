"""
The three-stage extraction cascade.

Ported from the text system (Memora/src/extractor.py:364-412). The escalation
predicate is kept deliberately, visibly parallel to the original, because that
parallelism IS a result: the routing logic that decided "is this text turn worth
an LLM call?" is modality-agnostic scalar logic that decides "is this frame
worth a VLM call?" unchanged.

  Stage 1  cheap gate        text: keyword/importance heuristic
                             video: motion energy vs the previous kept frame
  Stage 2  cheap tagger      text: pattern classifier
                             video: SigLIP prompt bank (calibrated sigmoid)
  Stage 3  expensive model   text: LLM extraction
                             video: Qwen2-VL on the escalated frame only

The headline efficiency metric is the ESCALATION RATE: what fraction of frames
needed Stage 3, and what accuracy was bought by paying for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .frames import Frame
from .prompt_bank import DEFAULT_BANK, Relation, by_name
from .stage2_siglip import Observation

# Same constant, same meaning, same value as the text system's config.py:147.
STAGE_3_CONFIDENCE_THRESHOLD = 0.7
# Stage 1: below this much visual change, the frame is redundant — skip it.
MOTION_GATE = 0.02
# Stage 1 "looks important" bar, the analogue of text's heuristic_score > 0.5.
HEURISTIC_ESCALATE_FLOOR = 0.5


# --------------------------------------------------------------------------
# Stage 1 — cheap gate
# --------------------------------------------------------------------------
def motion_scores(frames: Sequence[Frame], thumb: int = 32) -> list[float]:
    """Mean absolute difference against the previous frame, on a tiny thumbnail.

    Cheap enough to run on every frame on CPU. Doubles as the video analogue of
    the text heuristic score: a frame with a lot of change is 'important'.
    """
    scores: list[float] = []
    prev: np.ndarray | None = None
    for f in frames:
        img = f.image
        h, w = img.shape[0], img.shape[1]
        sy, sx = max(1, h // thumb), max(1, w // thumb)
        small = img[::sy, ::sx].astype(np.float32) / 255.0
        if prev is None:
            scores.append(1.0)  # first frame is always novel
        else:
            n = min(small.shape[0], prev.shape[0])
            m = min(small.shape[1], prev.shape[1])
            scores.append(float(np.abs(small[:n, :m] - prev[:n, :m]).mean()))
        prev = small
    return scores


# --------------------------------------------------------------------------
# Stage 2 -> Stage 3 escalation predicate  (the verbatim structural port)
# --------------------------------------------------------------------------
def should_escalate(
    stage2_observations: Sequence[Observation],
    heuristic_score: float,
    stage3_enabled: bool = True,
    threshold: float = STAGE_3_CONFIDENCE_THRESHOLD,
) -> tuple[bool, str | None]:
    """Decide whether this frame needs the expensive model.

    Mirrors Memora/src/extractor.py:374-388 line for line:

        if not stage2_memories:
            should_use_stage3 = heuristic_score > 0.5
        elif stage2_memories:
            max_confidence = max(m['confidence'] for m in stage2_memories)
            if max_confidence < STAGE_3_CONFIDENCE_THRESHOLD:
                should_use_stage3 = True
                stage2_hint = stage2_memories[0]['type']

    Returns (should_escalate, stage2_hint). The hint is the relation Stage 2 was
    least sure about, handed to the VLM so it asks a targeted question rather
    than an open-ended one.
    """
    if not stage3_enabled:
        return False, None

    if not stage2_observations:
        # Nothing recognized at all. Escalate only if the frame looks important.
        return heuristic_score > HEURISTIC_ESCALATE_FLOOR, None

    max_confidence = max(o.confidence for o in stage2_observations)
    if max_confidence < threshold:
        weakest = min(stage2_observations, key=lambda o: o.confidence)
        return True, weakest.relation

    return False, None


# --------------------------------------------------------------------------
# Stage 3 — VLM
# --------------------------------------------------------------------------
@dataclass
class CascadeStats:
    """What the cascade actually cost. This is the headline efficiency result."""

    n_frames: int = 0
    n_gated_out: int = 0        # Stage 1 rejected (redundant)
    n_stage2: int = 0           # reached SigLIP
    n_stage3: int = 0           # escalated to the VLM
    stage3_failures: int = 0

    @property
    def escalation_rate(self) -> float:
        return self.n_stage3 / self.n_stage2 if self.n_stage2 else 0.0

    def as_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "n_gated_out": self.n_gated_out,
            "n_stage2": self.n_stage2,
            "n_stage3": self.n_stage3,
            "stage3_failures": self.stage3_failures,
            "escalation_rate": round(self.escalation_rate, 4),
        }


class Stage3VLM:
    """Qwen2-VL asked a targeted, closed-vocabulary question about one frame.

    The vocabulary is kept closed even here so Stage 3 output is directly
    comparable with Stage 2 output and lands in the same schema. An open-ended
    caption would be richer but would not resolve into the same relation slots.
    """

    DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        bank: tuple[Relation, ...] = DEFAULT_BANK,
        device: str | None = None,
        max_new_tokens: int = 24,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
        )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.relations = by_name(bank)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _ask(self, image: np.ndarray, question: str) -> str:
        from PIL import Image  # noqa: PLC0415

        torch = self.torch
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=[Image.fromarray(image)], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1] :]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def extract(
        self,
        frame: Frame,
        hint: str | None = None,
        confidence: float = 0.85,
    ) -> list[Observation]:
        """Ask about the hinted relation (or all single-valued ones if no hint).

        Answers are snapped back onto the closed vocabulary; anything the model
        says that is not in the bank is DROPPED, not invented into the graph.
        This is the video analogue of `_parse_and_validate` in the text system.
        """
        targets = [hint] if hint and hint in self.relations else [
            r for r, rel in self.relations.items() if rel.cardinality == "single"
        ]

        results: list[Observation] = []
        for rel_name in targets:
            rel = self.relations[rel_name]
            options = ", ".join(o.lstrip("a ").lstrip("an ") for o in rel.objects)
            question = (
                f"Look at this image. Choose exactly one option from this list "
                f"that best matches: {options}. Answer with the option only."
            )
            try:
                answer = self._ask(frame.image, question).lower()
            except Exception:  # noqa: BLE001 — a VLM failure must not kill ingest
                continue

            matched = self._snap(answer, rel)
            if matched is not None:
                results.append(
                    Observation(
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        relation=rel_name,
                        object=matched,
                        confidence=confidence,
                        source="qwen2vl",
                    )
                )
        return results

    @staticmethod
    def _snap(answer: str, rel: Relation) -> str | None:
        """Map free text back onto the closed vocabulary, or give up."""
        answer = answer.strip().strip(".").lower()
        for obj in rel.objects:
            core = obj.removeprefix("an ").removeprefix("a ").strip()
            if core and (core in answer or answer in core):
                return obj
        return None


def merge_stage_results(
    stage2: Sequence[Observation], stage3: Sequence[Observation]
) -> list[Observation]:
    """Prefer Stage 3 where it spoke, keep Stage 2 elsewhere.

    Same policy as the text system's `_merge_stage_results`: the expensive model
    wins its own slots, the cheap model retains the rest. Note the text version
    had a bug where merged Stage-3 ids could collide; here the key is
    (frame_id, relation), which is unique by construction.
    """
    merged: dict[tuple[int, str, str], Observation] = {}
    for o in stage2:
        merged[(o.frame_id, o.relation, o.object)] = o

    overridden_slots = {(o.frame_id, o.relation) for o in stage3}
    for key in list(merged):
        if (key[0], key[1]) in overridden_slots:
            # Only single-valued slots get overridden wholesale.
            del merged[key]

    for o in stage3:
        merged[(o.frame_id, o.relation, o.object)] = o

    return sorted(merged.values(), key=lambda o: (o.frame_id, o.relation, o.object))
