# MI4MedFM 2026 Submission Plan --- SonoState as a Mech-Interp Probe

**Venue:** The 1st MICCAI Workshop on Mechanistic Interpretability for
Medical Foundation Models (MI4MedFM), Abu Dhabi, UAE.
**Site:** https://mi4medfm.github.io/2026
**Submission deadline:** Jul 15 2026 (00:00 UTC).
**Format:** LNCS, double-blind, 8 pages excluding references (verify on CFP).

---

## Headline claim

SonoState is a **post-hoc state-space probe** of a frozen echocardiographic
FM. The probe is *identity-initialized*, so any structure it surfaces is
information already in the encoder. It uncovers:

1. Cardiac-cycle manifold (closed loops in PCA, ED-vs-ES decode at high
   accuracy, mean angular separation ≈ 180°).
2. View-conditioned sub-attractors (per-view silhouette > 0).
3. Phase-locked attention circuits.
4. Bounded heart-rate decoding and bounded round-trip error under a learned
   inverse — locally isometric flow on the cycle.
5. Operational safety signal: probe's own h=1 forecast error as an OOD
   detector vs. pediatric / motion-corrupted scans.

## Differentiation from the MWM submission

- MI4MedFM: interpretability, frozen encoder, probe-as-instrument,
  safety vignette.
- MWM: world-model framing, forecasting, intervention conditioning, cycle
  consistency as planning property.

## Required experiments before submission

`bash experiments/submit_all.sh` after setting `DATA_ROOT` and `EXP_ROOT`.

| # | Sweep / analysis | Status | Script |
|---|---|---|---|
| 1 | S1_dim — d ∈ {32,64,128,256,512} | TODO | gen + analyse |
| 2 | S2_lambda_u — uniformity weight | TODO | gen + analyse |
| 3 | S4_encoder — {frozen, fine-tuned} | TODO | gen + analyse |
| 4 | S5_init — {identity, random, no-residual} | TODO | gen + analyse |
| 5 | C1_shuffled_time — temporal-causality control | TODO | gen + analyse |
| 6 | Phase decoding from ED/ES | TODO | `experiments/analysis/phase_decoding.py` |
| 7 | Heart-rate decoding | TODO | `experiments/analysis/heart_rate.py` |
| 8 | Intrinsic dim + view silhouette + loop closure | TODO | `experiments/analysis/geometry.py` |
| 9 | EF ridge probe vs raw mean-pool | TODO | `experiments/analysis/ef_probe.py` |
| 10 | Cycle consistency (post-hoc inverse) | TODO | `experiments/analysis/cycle_consistency.py` |
| 11 | OOD safety monitor (Pediatric, motion-corrupted) | TODO | `experiments/analysis/ood_monitor.py` |

## Required figures

- `fig:trajectories`: cardiac-cycle PCA loops.
- `fig:views`: same projection colored by view.
- `fig:attention`: EchoJEPA given/received attention across cycle.
- `fig:phase`: ED/ES histogram + decode accuracy.
- `fig:dim_sweep`: TwoNN intrinsic dim + decode acc vs $d$.
- `fig:ood` (optional): forecast-error histogram ID vs OOD.

## Writing TODOs

- [ ] Replace placeholders in Tabs. forecast, phase, HR, OOD.
- [ ] Tighten Background with concrete MI references (activation patching,
  sparse autoencoders).
- [ ] Decide whether the OOD vignette stays in the body or moves to
  appendix.
- [ ] Verify CFP page limit (8 vs 10).

## Build

```
cd paper/mi4medfm2026
make
```
