# packs/synthetic-60s-nocarry

The first real ingest run (Kaggle, T4), kept as a fixture because it is the
artifact that exposed the gate/tau coupling bug.

**The tags are meaningless.** The input was a synthetic clip of coloured
rectangles, which is out of distribution for every prompt in the bank, so
SigLIP scored ~0.0001 on all 34 prompts and every frame escalated to the VLM
(`escalation_rate: 1.0`). Do not read the objects as results.

What it *is* good for: it shows the pre-fix failure mode. 24 assertions, eight
1-second islands per relation, covering 8s of a 60s video, because the motion
gate dropped 52 of 60 frames and `tau=2.0` could not bridge the survivors.

    python -m video_memory.demo --pack ../packs/synthetic-60s-nocarry --timeline
