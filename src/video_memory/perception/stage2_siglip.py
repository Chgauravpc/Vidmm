"""
Stage 2: cheap, calibrated, closed-vocabulary tagging with SigLIP.

Why SigLIP and not CLIP. CLIP is trained with a softmax contrastive loss, so its
per-prompt scores are only meaningful *relative to the other prompts in the
batch* — they sum to 1 and shift as you add or remove candidates. SigLIP is
trained with a pairwise sigmoid loss, so each (image, caption) pair gets an
INDEPENDENT probability. That matters here for one specific reason: the whole
cascade hinges on the predicate `max_confidence < 0.7 -> escalate to the VLM`.
That comparison is only sound if the confidence is an absolute, per-prompt
calibrated number. With CLIP softmax scores it would silently depend on how many
objects happen to be in the prompt bank.

Output is a flat list of per-frame Observations, plus the frame embedding matrix
that gets written into the feature pack so the query side needs no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .frames import Frame
from .prompt_bank import DEFAULT_BANK, Relation, by_name, flatten

DEFAULT_MODEL = "google/siglip-base-patch16-224"

# Multi-valued relations emit every object above this bar (several can hold at
# once); single-valued relations emit only their argmax.
MULTI_VALUED_FLOOR = 0.30


@dataclass(frozen=True)
class Observation:
    """One (relation, object) reading at one frame, with calibrated confidence."""

    frame_id: int
    timestamp: float
    relation: str
    object: str
    confidence: float
    source: str = "siglip"


class SigLIPTagger:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        bank: tuple[Relation, ...] = DEFAULT_BANK,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.batch_size = batch_size
        self.bank = bank
        self.relations = by_name(bank)

        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.captions, self.index = flatten(bank)
        self.text_embeds = self._encode_text(self.captions)

    # ---------------- encoding ----------------
    @staticmethod
    def _as_tensor(out):
        """Normalize what get_*_features returns across transformers versions.

        Some versions hand back a bare tensor; others return a
        BaseModelOutputWithPooling. Reach for the pooled representation, which
        is the one SigLIP's contrastive head is trained on - falling back to the
        CLS position only if there is no pooler.
        """
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _encode_text(self, captions: Sequence[str]) -> "np.ndarray":
        torch = self.torch
        inputs = self.processor(
            text=list(captions), padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            feats = self._as_tensor(self.model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_images(self, images: Sequence["np.ndarray"]) -> "np.ndarray":
        torch = self.torch
        out: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            chunk = list(images[i : i + self.batch_size])
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device, self.dtype)
            with torch.no_grad():
                feats = self._as_tensor(self.model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.text_embeds.shape[1]))

    # ---------------- calibration ----------------
    def _sigmoid_probs(self, image_embeds: "np.ndarray") -> "np.ndarray":
        """(n_frames, n_prompts) independent probabilities.

        Reproduces SigLIP's own head: sigmoid(logit_scale * cos_sim + logit_bias).
        Using the model's learned scale/bias is what makes these numbers
        comparable across frames and across prompt-bank sizes.
        """
        # SigLIP exposes these as learned parameters; fall back to the published
        # initialization if a future version relocates them, so a rename
        # degrades the calibration instead of crashing mid-ingest.
        scale_param = getattr(self.model, "logit_scale", None)
        bias_param = getattr(self.model, "logit_bias", None)
        logit_scale = float(scale_param.exp().detach().cpu()) if scale_param is not None else 100.0
        logit_bias = float(bias_param.detach().cpu()) if bias_param is not None else -12.92
        sims = image_embeds @ self.text_embeds.T
        logits = sims * logit_scale + logit_bias
        return 1.0 / (1.0 + np.exp(-logits))

    # ---------------- tagging ----------------
    def tag(self, frames: Iterable[Frame]) -> tuple[list[Observation], "np.ndarray", list[Frame]]:
        """Tag frames. Returns (observations, frame_embeddings, frames_as_list)."""
        frame_list = list(frames)
        if not frame_list:
            return [], np.zeros((0, self.text_embeds.shape[1])), []

        embeds = self.encode_images([f.image for f in frame_list])
        probs = self._sigmoid_probs(embeds)

        # Column groups per relation, so argmax is taken within a relation.
        cols: dict[str, list[int]] = {}
        for col, (rel_name, _obj) in enumerate(self.index):
            cols.setdefault(rel_name, []).append(col)

        observations: list[Observation] = []
        for row, frame in enumerate(frame_list):
            for rel_name, col_ids in cols.items():
                rel = self.relations[rel_name]
                if rel.cardinality == "single":
                    best = max(col_ids, key=lambda c: probs[row, c])
                    observations.append(
                        Observation(
                            frame_id=frame.frame_id,
                            timestamp=frame.timestamp,
                            relation=rel_name,
                            object=self.index[best][1],
                            confidence=float(probs[row, best]),
                        )
                    )
                else:
                    for c in col_ids:
                        p = float(probs[row, c])
                        if p >= MULTI_VALUED_FLOOR:
                            observations.append(
                                Observation(
                                    frame_id=frame.frame_id,
                                    timestamp=frame.timestamp,
                                    relation=rel_name,
                                    object=self.index[c][1],
                                    confidence=p,
                                )
                            )
        return observations, embeds, frame_list
