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
| Interval store (`store.py`, SQLite) | ✅ done |
| Retriever + 5-signal ranker | ✅ done, signals inspectable |
| Cited answers (`answer.py`) | ✅ done, template-composed |
| Demo CLI (`demo.py`) | ✅ `python -m video_memory.demo --demo` |

**No measured numbers yet.** The cascade's logic is tested end-to-end on
synthetic observations; the model wiring has not been executed on real video.
Escalation rate, ×realtime, and accuracy are unmeasured until the notebook runs.

The memory core has **no third-party dependencies** and runs anywhere Python
runs. The perception/ingest layers run offline on a GPU (Kaggle/Colab) and emit
committed *feature packs* so a reviewer can run the query side with no GPU.

## Try it

No GPU, no models, no dataset — the synthetic memory is built in:

```bash
cd src
python -m video_memory.demo --demo --timeline          # see the whole memory
python -m video_memory.demo --demo                     # sample questions
python -m video_memory.demo --demo -q "what was I holding at 1:20" --explain
```

Against a real ingested pack:

```bash
python -m video_memory.demo --pack ../packs/clip -q "when was I in the kitchen"
```

## Run the tests

```bash
python tests/test_resolve.py            # standalone runners, no pytest needed
python tests/test_cascade.py
python tests/test_query.py
python tests/test_carry_forward.py
# or
python -m pytest tests/ -v
```

58 checks, all CPU, no network.

## Answers cite, and admit gaps

```
Q: what was I holding at 0:52
Nothing is asserted at 00:52.00. The closest fact is HOLDS=a phone over
[01:00.00 - 01:35.00]. Note this gap could be a genuine absence or a missed
detection - the memory cannot distinguish them.
```

Answers are composed from templates, not generated. A language model at the end
would let fluent prose paper over a bad retrieval, which is the failure mode
this project argues against. Every number traces to a row; the citation is the
product and the sentence is packaging.

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
