# SonoState experimental program

This directory contains the **post-paper-skeleton** experimental
infrastructure that turns SonoState from a proof-of-concept into a
submission-grade study. Everything here is additive — none of it modifies
the upstream `app/`, `src/`, `evals/`, or `configs/` trees. New training
configs live in `experiments/configs/`, new analyses in
`experiments/analysis/`, Condor submission files in `experiments/condor/`,
and the figure pipeline in `experiments/figures/`.

## Project axes

The MWM and MI4MedFM submissions stand on five legs of evidence. Each leg
has dedicated experiments below.

| Leg | Question | Primary script | Output |
|-----|----------|----------------|--------|
| L1. Forecast quality | Does the transition beat persistence at multi-step? | `analysis/forecast_curves.py` | `RESULTS.md` Tab. 1 |
| L2. Phase recovery | Do `z_t` trajectories carry cardiac phase (vs ED/ES labels)? | `analysis/phase_decoding.py` | `RESULTS.md` Tab. 2, Fig. phase |
| L3. Heart-rate decoding | Does the angular speed around the loop encode HR? | `analysis/heart_rate.py` | `RESULTS.md` Tab. 3 |
| L4. Geometry | Intrinsic dimension, view-conditioned attractors, silhouette | `analysis/geometry.py` | `RESULTS.md` Tab. 4 |
| L5. Downstream transfer | EF ridge regression from `z_t` vs mean-pool baseline | `analysis/ef_probe.py` | `RESULTS.md` Tab. 5 |

Plus three controls / ablations:

| Control | Question | Script |
|--------|----------|--------|
| C1. Shuffled-time | Are loops genuinely temporal, or a projection artefact? | `analysis/shuffled_time.py` |
| C2. Identity-init  | Does the model ever depart from persistence? | `analysis/init_drift.py` |
| C3. Cycle-consistency | Forward then backward — does drift accumulate? | `analysis/cycle_consistency.py` |

## Sweeps

We use Condor to run all sweeps in parallel on H100 nodes. The matrix is
defined in `condor/sweep_matrix.yaml` and consumed by the submit files.
Each cell in the matrix produces a fully self-contained checkpoint
directory under `$EXP_ROOT/{sweep_name}/{cell_id}/` and a
`metrics.json` written by the analysis stage.

| Sweep | Cells | Config template |
|-------|-------|-----------------|
| S1. Latent dim | d ∈ {32, 64, 128, 256, 512} | `configs/sweep_dim.yaml` |
| S2. Uniformity weight | λ_u ∈ {0.0, 0.1, 1.0, 5.0} | `configs/sweep_lambda_u.yaml` |
| S3. Forecast loss | {L1, cos, L1+cos} | `configs/sweep_forecast_loss.yaml` |
| S4. Encoder | {frozen, fine-tuned} | `configs/sweep_encoder.yaml` |
| S5. Transition init | {identity, random, no-residual} | `configs/sweep_init.yaml` |
| S6. Schedule (long) | {150 ep, 300 ep cosine cooldown} | `configs/sweep_schedule.yaml` |

## Quickstart on Condor

```bash
# 1. Set environment
export EXP_ROOT=/scratch/bxu/project/EchoJEPA/experiments_2026
export DATA_ROOT=/scratch/bxu/project/echonet
export REPO=/scratch/bxu/project/EchoJEPA
export PYTHONPATH=$REPO

# 2. Submit all sweeps (creates per-cell checkpoint dirs and analysis jobs)
cd $REPO
condor_submit_dag experiments/condor/sweeps.dag

# 3. After everything finishes, build paper figures and tables
python experiments/figures/make_paper_figures.py \
    --root $EXP_ROOT \
    --out  $REPO/paper/_figs
```

## Conventions

* **Reproducibility.** Every config sets `meta.seed` and we pin
  `torch.backends.cudnn.deterministic` only in eval scripts (training is
  bf16-stochastic by design).
* **Cohorts.** EchoNet-Dynamic train/val/test splits are read from the
  upstream `FileList.csv`; pediatric is from EchoNet-Pediatric (A4C).
* **Outputs.** Each analysis emits `metrics.json` (machine readable) and
  `summary.md` (human readable) into the run directory. The figure
  pipeline globs for these.
* **Privacy.** No PHI ever lands in this directory; only encoder
  activations and aggregate statistics.
