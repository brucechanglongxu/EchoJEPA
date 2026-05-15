# SonoState — Collaborator Status (May 13, 2026)

**TL;DR:** Sweep finished (21 cells, 6 analyses each). Several headline
claims need to be reframed before circulating: the transition operator
**does not beat persistence**, and ED/ES phase decoding from a single clip
is at chance unless the encoder is fine-tuned. Three findings remain
strong and publishable as-is.

## What was run

- 21 sweep cells (S1–S6 + C1) trained on EchoNet-Dynamic (V-JEPA-2 ViT-L
  backbone, 8×H100, ~12h each).
- 6 per-cell analyses: forecast curves, phase decoding, heart-rate
  decoding, geometry, EF probe, cycle consistency.
- All metrics in `/mnt/vast/exp/brucexu/EchoJEPA/experiments_2026/*/*/analysis_*/metrics.json`.
- Aggregated tables + figures in `paper/_figs/` (`RESULTS.md`,
  `dim_sweep.pdf`, `uniformity_sweep.pdf`, `init_ablation.pdf`,
  `shuffled_control.pdf`, plus per-analysis `.csv`).

## Findings — strong (ready to discuss)

1. **Closed cardiac-cycle manifold.** Mean cosine loop closure 0.75,
   TwoNN intrinsic dimension ≈ 3.0 across the d∈{32…512} sweep.
   Disabling the uniformity term (λ_u=0) inflates ID to 6.1 and lowers
   loop closure to 0.55 — the first clean ablation that confirms the
   uniformity loss is what gives the manifold its compact closed shape.

2. **Uniformity term is essential for downstream EF probe.** Ridge probe
   on z gives R²=0.14 at λ_u=0 vs R²=0.48 at λ_u=0.1; MAE 8.10 vs 7.00.

3. **Fine-tuning the encoder dominates everything.** EF probe MAE drops
   from 7.55 (frozen, d=128) to **5.37 (fine-tuned)**; R² 0.38 → **0.67**.
   Same fine-tuned cell is also the only one that decodes ED-vs-ES above
   chance (66% vs 51% shuffled).

## Findings — weak / negative (need reframing)

4. **Forecast operator does NOT beat persistence (any horizon, any
   cell).** Forecast L1 = 0.046, persistence L1 = 0.025 at h=1 for the
   default cell. Same direction at h=4 (0.066 vs 0.025). Two competing
   explanations:
   - clip stride (~1s) on ~2s clips makes the encoder state move very
     little between adjacent clips, so persistence is artificially
     strong;
   - the residual transition is over-fitting in-clip noise.

   **Action item for collaborators:** decide whether to (a) reframe the
   forecast section as a *probe* finding ("frozen V-JEPA-2 has no
   single-step predictability beyond persistence"), (b) re-train with
   beat-to-beat strides, or (c) switch the metric to *displacement
   prediction* (predict z_{t+1}−z_t directly).

5. **Phase decoding is at chance for frozen-encoder cells.** ED vs ES
   logistic decoder gets 0.40–0.42 accuracy across all 19 frozen-encoder
   cells (shuffled control 0.48–0.51). Only the fine-tuned cell decodes
   (0.66). Mean angular separation only 12–20° (a true antipodal pair
   would be ~180°). Closed-loop ≠ linear phase separability.

6. **Heart-rate from latent angular speed: weak.** Per-video pearson r ≈
   0.08–0.14 against the ED-ES-derived bpm proxy; MAE 45–65 bpm. The
   period-finding gives the right order of magnitude (median 100–140 bpm)
   but no fine resolution.

7. **Cycle-consistency RT_1 is paradoxically smallest under collapse.**
   The two ablations that visibly collapse the manifold (λ_u=0 and
   no-residual init) both give RT_1 ≈ 10⁻⁴–10⁻⁵, which a naive reading
   would call "best". Collaborators may want to redefine the RT metric
   (e.g., normalize by displacement, or report only on non-collapsed
   cells).

## Recommended discussion before submission

- **mwm2026 (Medical World Models workshop):** the world-model framing
  hinges on (4). If we cannot get a forecast win, we should pivot the
  paper's contribution to (1)–(3): an *interpretable, low-dimensional
  state space whose geometry is shaped by the uniformity loss and whose
  fine-tuned variant gives SOTA EF regression on EchoNet*. That is still
  a clean MICCAI workshop paper, but it is a different paper than the
  current draft.

- **mi4medfm2026 (interpretability workshop):** the negative findings
  (4)–(5) are *publishable as findings* in an interpretability venue.
  Recommend leaning into "the closed-loop appearance does not survive a
  linear phase decoder; the fine-tuned encoder fixes this". The current
  draft already partially supports this framing.

- **Paths forward (technical):**
  - Beat-to-beat forecast retrain: change clip stride, re-run S1+S2 (~1
    day GPU).
  - Displacement-prediction metric retrofit: 1 hour of code.
  - Better phase-decoder (PCA→circular, not raw): ~30 min.
  - Aligned ECG cohort for ground-truth phase: data-acquisition
    question.

## What is in this directory

- `paper/mwm2026/main.tex` — workshop paper (LNCS 8 pages). Tables for
  EF probe, ablations, cycle-consistency, geometry **filled with real
  numbers**. Forecast and phase tables now show the negative findings
  honestly.
- `paper/mi4medfm2026/main.tex` — interpretability paper (LNCS).
  Findings 1–7 reflect the new data; HR table filled.
- `paper/_figs/RESULTS.md` — ground-truth markdown of every metric
  collected (108 metrics.json files across 6 analyses).
- `paper/_figs/*.pdf` — auto-generated figures.

## What is *not* yet done

- `ood_monitor` analysis (not run; would need a separate OOD cohort
  CSV).
- Figure 1 (PCA loops on a panel of patients) — placeholder PNGs in
  both papers.
- Figure 2 / dim_sweep.pdf is generated but is not yet the publication
  figure (axes/legend need polish).
- `forecast_curves` for one cell (`S5_init/identity` was the stuck job
  I cancelled); 20/21 cells covered.
