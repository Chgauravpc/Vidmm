# Interval Memory for Long Video

A **queryable temporal state layer** for long video: an explicit, citable record
of *what was true over which time intervals*, maintained persistently across
hours of footage. Ask "what was true at 4:20, and how do you know?" and get an
answer anchored to the frames that produced it.

## The idea in one paragraph

Text memory systems stamp every fact at a single **point** in time and assume it
holds forever ("newest wins"). Video needs facts that occupy **intervals** with
real beginnings and ends, where conflict comes from **overlap**, not succession
("highest confidence wins"). This project shows the text design is the
*degenerate special case* of the interval design — and builds the general one.

The conflict resolver (`src/video_memory/resolve.py`) is a direct generalization
of a text temporal-knowledge-graph resolver: feed it open-ended, equal-confidence
intervals and it reproduces "newest wins" exactly
(`tests/test_resolve.py::test_text_is_degenerate_case`); feed it real intervals
with calibrated confidences and it does sweep-line overlap resolution text cannot
express.

## Status

| Layer | State |
|-------|-------|
| Interval schema (`types.py`) | ✅ done, CPU-only, zero deps |
| Sweep-line resolver (`resolve.py`) | ✅ done, 9/9 characterization tests pass |
| Perception cascade (SigLIP → Qwen2-VL) | ✅ code done, 17/17 logic tests pass; **not yet run on a GPU** |
| Feature packs (`pack.py`) | ✅ done, round-trip tested |
| Kaggle/Colab ingest notebook | ✅ `notebooks/ingest_kaggle.ipynb` |
| Interval TKG store (SQLite + numpy) | ⬜ planned |
| Retriever + cited-answer generation | ⬜ planned |

**No measured numbers yet.** The cascade's logic is tested end-to-end on
synthetic observations; the model wiring has not been executed on real video.
Escalation rate, ×realtime, and accuracy are unmeasured until the notebook runs.

The memory core has **no third-party dependencies** and runs anywhere Python
runs. The perception/ingest layers run offline on a GPU (Kaggle/Colab) and emit
committed *feature packs* so a reviewer can run the query side with no GPU.

## Run the tests

```bash
python tests/test_resolve.py            # standalone runner, no pytest needed
# or
python -m pytest tests/ -v
```

## Design notes

Full research plan (background from zero, related work, hypotheses, ablation
matrix, risk register) lives outside the repo in the planning notes. Key
inherited-but-fixed decisions:

- **Media time, not ingest time.** Facts are stamped in seconds-from-start of the
  media (valid time), kept separate from wall-clock ingest time (bi-temporal).
- **Closure is explicit.** An interval's end is `OBSERVED_END` (we saw it stop),
  `INFERRED_END` (a rival took over), or `OPEN` (still holding at last frame).
- **Evidence is load-bearing.** Every assertion carries the frame ids that
  produced it, so every answer can cite its source.
