# RESULTS scratch (hand-edit as runs land)

This file is the ground-truth scratch table that both papers read from.
It is **manually maintained** alongside the auto-generated
`paper/_figs/RESULTS.md` produced by `make_paper_figures.py`. Use this
file for the headline numbers that go into the paper bodies.

> Status legend: `RUN` (in flight), `OK`, `FAIL`, `TODO` (not started).

## L1. Forecast quality (EchoNet-Dynamic test)

| Method | h=1 L1 ↓ | h=1 1−cos ↓ | h=4 L1 ↓ | h=4 1−cos ↓ | n_videos |
|--------|---------:|------------:|---------:|------------:|---------:|
| Persistence baseline           | TODO | TODO | TODO | TODO | TODO |
| SonoState (frozen enc, S4_frozen) | TODO | TODO | TODO | TODO | TODO |
| SonoState (full,   S4_ftuned)     | TODO | TODO | TODO | TODO | TODO |

## L2. Phase recovery (EchoNet ED/ES)

| Metric | Value |
|--------|------:|
| Circular variance @ ED          | TODO |
| Circular variance @ ES          | TODO |
| Mean angular separation (deg)   | TODO |
| ED-vs-ES classification accuracy| TODO |
| Shuffled-label control          | TODO |

## L3. Heart-rate decoding

| Cohort | Pearson r | Spearman ρ | MAE (bpm) | n |
|--------|----------:|-----------:|----------:|--:|
| EchoNet-Dynamic VAL | TODO | TODO | TODO | TODO |

## L4. Geometry

| Metric | Value |
|--------|------:|
| Intrinsic dim (TwoNN)      | TODO |
| Loop closure (mean)        | TODO |
| View silhouette (cosine)   | TODO |

## L5. Downstream EF probe

| Backbone | MAE (% EF) ↓ | R² ↑ |
|----------|-------------:|-----:|
| Raw mean-pool (D=1024)         | TODO | TODO |
| SonoState state head (d=256)   | TODO | TODO |

## C1. Shuffled-time control

If shuffled forecast L1 ≈ persistence L1 we have ruled out projection-artefact loops.

| Variant | h=1 forecast L1 | persistence L1 |
|---------|---------------:|---------------:|
| true pairs    | TODO | TODO |
| shuffled pairs| TODO | TODO |

## C2. Identity-init drift

After 150 epochs, learnable `transition.log_scale` should be > -4.6 if the
model has chosen to depart from persistence. Report final value.

| Cell | log_scale at convergence |
|------|-------------------------:|
| identity init  | TODO |
| random init    | TODO |
| no-residual    | TODO |

## C3. Cycle consistency

| h | round-trip L1 ↓ |
|---|----------------:|
| 1 | TODO |
| 2 | TODO |
| 4 | TODO |
| 8 | TODO |

## OOD safety monitor (Discussion vignette)

| Cohort | n | mean forecast L1 | AUC vs ID |
|--------|--:|-----------------:|----------:|
| ID  (EchoNet-Dynamic test)        | TODO | TODO | — |
| OOD (EchoNet-Pediatric)           | TODO | TODO | TODO |
| OOD (low-quality / motion-shifted)| TODO | TODO | TODO |
