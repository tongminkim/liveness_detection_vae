# AGENTS.md — Multi-Agent Plan for One-Class VAE Deepfake/Liveness Detection

This document defines a multi-agent collaboration plan to deliver a one-class Variational Autoencoder (VAE) for deepfake/liveness detection using lip landmark time series. It aligns primarily with the main objective in `readme3.md` (two-head LF/HF VAE with one-class training) while offering optional extensions described in `readme1.md` (three-band split: LF/BP/HF) and `readme2.md` (hierarchical/ladder VAE and wavelet variants).

## Executive Summary

- Objective: Build a compact, real-time-capable, one-class VAE that learns live lip dynamics from landmark sequences and detects spoofs by reconstruction-based anomaly scoring.
- Core design (from readme3):
  - Input: lip landmarks `x ∈ R^{T×K×2}` (T≈48–64 frames, K≈20–30 points at 25–30fps).
  - Shared time encoder: TCN (dilations [1,2,4,8], channels 64→128) → per-frame embeddings `h_t`.
  - Two VAE heads:
    - LF-VAE (clip-level latent, small dimension ~32) → reconstruct low-frequency trajectory `x̂_LF`.
    - HF-VAE (frame-level latents, ~64) → reconstruct high-frequency residual `r̂_HF`.
  - Composite reconstruction: `x̂ = x̂_LF + r̂_HF`.
  - Losses: LPF target for LF, residual/HPF for HF, STFT emphasis for HF band, temporal derivatives, KL with β-scheduling, full reconstruction, latent decorrelation, and jitter consistency.
  - One-class training (live only). Inference uses decomposed errors and derived scores (DRS, ΔE, complexity C) → threshold τ.

- Extensions:
  - `readme1.md`: Three-band band-split with dedicated VAE heads (LF/BP/HF) and band decorrelation.
  - `readme2.md`: Ladder/Hierarchical VAE with top-down priors and wavelet/frequency mask regularizers.

---

## Agent Roster, Responsibilities, and Deliverables

Each agent owns a set of artifacts with clear inputs/outputs, acceptance criteria, and handoffs.

### 0) Program Manager (PM) Agent
- Responsibilities:
  - Define OKRs, milestones, and cross-agent dependencies.
  - Maintain risk register, change control, and weekly demo cadence.
- Deliverables:
  - Project roadmap and Gantt.
  - RACI matrix, status dashboards.
- Acceptance:
  - Milestones met within scope/quality; blockers resolved within SLA.

### 1) Data Agent
- Responsibilities:
  - Curate datasets and extract lip landmarks sequences `x`.
    - Sources referenced: GRID, LRS2, AI Hub (see `readme3.md`).
  - Preprocessing: normalization, optional EMA/Kalman smoothing (light), Δ/Δ² derivation (optional).
  - Data splits: train (live-only), dev/val/test (live+spoof if available), cross-domain partitions.
  - Augmentations: mild time jitter ±1f, 5–10% frame drop, speed 0.95–1.05×, small spatial noise.
- Deliverables:
  - Reproducible pipeline for landmarks generation and `.npy/.pt` datasets.
  - Metadata (FPS, T, K), augmentation specs.
- Acceptance:
  - Landmarks cover lips robustly; sequence stats validated; splits deterministic.

### 2) Signal Processing Agent
- Responsibilities:
  - Implement LPF/HPF filtering to form targets:
    - LF target: `x̃_LP = LPF(x)` for `x̂_LF` supervision.
    - HF target: `r_HF = x − x̃_LP` or `HPF(x)` for `r̂_HF`.
  - STFT configuration and frequency weighting for HF loss (upper third boosted).
  - Optional advanced variants:
    - Three-band split (LF/BP/HF) with decorrelation (from `readme1.md`).
    - Wavelet decomposition (Haar/Db4), learnable filterbanks with passband masks (from `readme2.md`).
- Deliverables:
  - Filtering utilities, STFT helpers, reproducibility notes (cutoffs, taps, windows).
- Acceptance:
  - Verified band separation with minimal leakage and aligned time delays.

### 3) Model Architect Agent
- Responsibilities:
  - Define module interfaces and build core network:
    - `SharedTCNEncoder` → `H ∈ R^{T×D}` with dilations [1,2,4,8], D≈128.
    - `LFVAEHead` (clip-level latent d≈32): pooling (mean+attention), reparam, LF decoder (TCN/GRU).
    - `HFVAEHead` (frame-level latents d≈64): per-frame stats, reparam, HF decoder (TCN).
    - Output composition: `x̂ = x̂_LF + r̂_HF`.
  - Parameter budget target: ~1.2–1.8M params (from `readme3.md`).
  - Optional architecture flags:
    - Three-band heads (LF/BP/HF) as in `readme1.md`.
    - Ladder VAE prior modules (top-down) as in `readme2.md`.
- Deliverables:
  - Modular PyTorch components with unit tests; config presets.
- Acceptance:
  - Shapes match; forward pass deterministic; parameters within target budget.

### 4) Loss & Training Agent
- Responsibilities:
  - Implement composite loss:
    - LF reconstruction: `L_rec^LF = || x̂_LF − x̃_LP ||_2^2`.
    - HF reconstruction: `L_rec^HF = || r̂_HF − r_HF ||_2^2`.
    - STFT emphasis for HF: weighted spectral L2 on high bands.
    - Temporal derivative alignment: L1 on ∇, ∇², ∇³ of `x̂` vs `x`.
    - KL terms: `β_LF≈4–8` (strong), `β_HF≈0.5–1.0` (weak), with warmup/anneal.
    - Full reconstruction: `L_full = || x̂ − x ||_1` (small weight).
    - Latent decorrelation: correlation or cross-covariance penalty between upsampled LF latent and HF latent/time series.
    - Jitter consistency on live: `|| r̂_HF(x′) − r̂_HF(x) ||_1` with time perturbations.
  - Optimizer & schedule: AdamW (lr 1e-3), cosine decay, batch 16–32, AMP, gradient clipping, early stopping on val loss.
- Deliverables:
  - Loss composition with documented weights; training loop with logging and checkpoints.
- Acceptance:
  - Stable training without posterior collapse; loss curves smooth; KL warmup effective.

### 5) Metrics & Evaluation Agent
- Responsibilities:
  - Anomaly scores and decision thresholds:
    - `E_smooth = || x − x̂_LF ||`.
    - `E_detail = || r_HF − r̂_HF ||`.
    - Detail-to-roughness score (DRS): `E_smooth / (E_detail + ε)`.
    - `ΔE`: HF error increase under time jitter input.
    - Complexity `C`: sample entropy/spectral flatness indicators.
    - Final score: `S = α·DRS − β·ΔE + γ·C`.
  - Metrics: APCER, BPCER, ACER, EER, ROC-AUC, PR-AUC. Threshold τ via Youden index, min-ACER, or target FPR.
  - Calibration: optional Platt/Isotonic for environments that require probability outputs.
- Deliverables:
  - Evaluation scripts and reports per domain; ablations:
    - One-class S only vs optional classifier S_cls vs fusion.
    - With/without HF STFT term, derivatives, jitter consistency.
- Acceptance:
  - Reproducible numbers with confidence intervals; clear τ selection rationale.

### 6) Optional Feature/Classifier Agent (Stage 2, from `readme1.md`/`readme2.md`)
- Responsibilities:
  - Construct clip-level feature vector `φ` from VAE outputs:
    - Band error stats (mean/p90/max), ratios (HF/LF), KL summaries, ΔSpec, band energy ratios.
    - Optional: jerk/acceleration variance, latent summaries (PCA), cross-corr terms.
  - Train a small classifier (logistic/MLP/XGBoost) on `φ` using live+spoof (few-shot allowed).
  - Fusion: `S = w·S_cls + (1−w)·Norm(S_ano)` with `w` tuned on dev.
- Deliverables:
  - `φ` extractor, classifier training, calibration, and fusion recipe.
- Acceptance:
  - Improves ACER/EER over one-class alone on dev; avoids overfitting (dropout/L2, K-fold).

### 7) Robustness & Security Agent
- Responsibilities:
  - Stress tests: compression levels, frame-rate changes, camera noise, small misalignments.
  - Attack modeling: replay artifacts, rendering flicker, mouth-only manipulations.
  - Domain shift tests and drift monitoring guidelines.
- Deliverables:
  - Robustness report and recommended augmentations and operating thresholds per environment.
- Acceptance:
  - Defined safe operating area; documented failure modes and mitigations.

### 8) Inference & Packaging Agent
- Responsibilities:
  - Export runtime packages: TorchScript/ONNX, CPU-friendly kernels, optional quantization.
  - Real-time constraints: batch=1 latency targets with sliding window (T≈48–64).
  - API: `encode(x) → H`, `reconstruct(x) → {x̂_LF, r̂_HF, x̂}`, `score(x) → S`.
- Deliverables:
  - Inference module with CLI and minimal dependencies; latency/throughput benchmarks.
- Acceptance:
  - Meets latency budget on target hardware; numerically matches training within tolerance.

### 9) MLOps Agent
- Responsibilities:
  - Repository structure, CI for lint/tests, data versioning hooks, experiment tracking, model registry.
  - Reproducible seeds, environment lockfiles, release process.
- Deliverables:
  - CI workflows, testing harness, artifact storage conventions, release tags.
- Acceptance:
  - Green CI; deterministic training/inference; documented runbooks.

### 10) Documentation Agent
- Responsibilities:
  - Maintain READMEs, API docs, diagrams, and this AGENTS plan.
  - Prepare quickstart, config cookbook, and deployment guide.
- Deliverables:
  - Up-to-date docs aligned with code and experiments.
- Acceptance:
  - New engineer can reproduce baseline and evaluate within one day.

---

## Cross-Agent Handoffs (Baseline Path)

1. Data Agent → Signal Processing Agent:
   - Landmarks dataset spec and normalization stats.

2. Signal Processing Agent → Model Architect Agent:
   - LPF/HPF utilities, STFT configs.

3. Model Architect Agent → Loss & Training Agent:
   - Model modules ready; forward interfaces finalized.

4. Loss & Training Agent → Metrics & Evaluation Agent:
   - Checkpoint, training logs, loss component traces.

5. Metrics & Evaluation Agent → PM + Inference Agent:
   - Scores, thresholds τ, and recommended operating points.

6. Inference Agent → MLOps + Documentation Agent:
   - Packaged runtime, API docs, latency report.

7. Optional: Loss & Training + Metrics → Feature/Classifier Agent:
   - `φ` extraction hooks and dev labels for semi-supervised classifier.

---

## Milestones and Sprints

- M1: Data & Filtering (Week 1)
  - Datasets ready, LPF/HPF targets validated, unit tests pass.

- M2: Baseline Model (Week 2)
  - Shared TCN + LF/HF heads; dry-run forward; <2M params.

- M3: Loss & Training (Weeks 3–4)
  - Composite loss with KL warmup; stable convergence on live-only data.

- M4: Evaluation & Thresholding (Week 5)
  - Anomaly scores, ACER/EER; dev threshold τ; robustness smoke tests.

- M5: Inference Packaging (Week 6)
  - Exported runtime, latency OK; documentation first pass.

- M6: Optional Extensions (Weeks 7–8)
  - Three-band or ladder VAE variant ablations; Stage-2 classifier; calibration.

---

## Acceptance Criteria (Baseline)

- Functional:
  - One-class model trains on live-only, converges without collapse.
  - Produces decomposed reconstructions `x̂_LF`, `r̂_HF`, and final `x̂`.

- Quality:
  - Achieves target ACER/EER on dev; threshold τ documented.
  - Robustness: small performance degradation under mild jitter/compression.

- Performance:
  - Parameter count ~1.2–1.8M; real-time feasible on target device.
  - Inference latency meets budget for T≈48–64 on CPU.

- Reproducibility:
  - End-to-end script reproduces results within tolerance on fresh machine.

---

## Risks and Mitigations

- Posterior collapse:
  - KL warmup, free-bits, β-balancing (higher β on LF, lower on HF), skip-context in decoders.

- Band leakage / misalignment:
  - Use filtfilt/symmetric padding; equal FIR taps or explicit delay correction; frequency-mask penalties (optional).

- Overfitting (few domains):
  - Strong regularization, augmentations, early stop; consider Stage-2 classifier with φ only (no raw taps), and model freeze.

- Domain drift:
  - Calibrate per environment; maintain dev-set thresholds; include on-device recalibration (temperature scaling).

---

## Artifacts and Naming Conventions

- Core modules:
  - `SharedTCNEncoder`, `LFVAEHead`, `HFVAEHead`, `VAEReparam`, `LFDecoder`, `HFDecoder`.
  - `BandTargets` for LPF/HPF/residual; `STFTHelper` for spectral losses.
  - `LossComputer` composing: `L_rec^LF`, `L_rec^HF`, `L_stft`, `L_grad`, `L_KL^LF`, `L_KL^HF`, `L_full`, `L_decor`, `L_jitter`.
  - `Scorer` computing `E_smooth`, `E_detail`, `DRS`, `ΔE`, `C`, and final `S`.

- Config presets:
  - `baseline_lf_hf.yaml` (readme3-aligned).
  - `variant_3band.yaml` (readme1).
  - `variant_ladder_wavelet.yaml` (readme2).

- Tests:
  - Shape/grad tests for encoders/decoders and reparam; filters equivalence tests; loss term invariants.

---

## Optional Advanced Tracks

- Three-band multi-head VAE (from `readme1.md`):
  - Add `BP-VAE` (mid-band) with band decorrelation and synthesis `x̂ = x̂_LF + x̂_BP + x̂_HF`.

- Ladder/Hierarchical VAE (from `readme2.md`):
  - Top-down priors `p(z_s | z_<s)`, multi-scale decoders, wavelet routing, frequency-mask penalties, and scale-specific structural regularizers (smoothness for LF, sparsity for HF).

- Stage-2 classifier (from `readme1.md`, `readme2.md`):
  - φ extraction, small MLP/logistic/XGBoost, calibration, and score fusion.

---

## Quickstart (Baseline Flow of Work)

1. Data Agent: produce `x` sequences and normalization stats; define T, K, FPS.
2. Signal Processing Agent: build `LPF/HPF(x)` and STFT helpers; validate band targets.
3. Model Architect Agent: implement `SharedTCNEncoder`, `LFVAEHead`, `HFVAEHead`, composition `x̂ = x̂_LF + r̂_HF`.
4. Loss & Training Agent: wire composite loss, KL schedules, and trainer; run live-only training.
5. Metrics & Evaluation Agent: compute `E_smooth`, `E_detail`, `DRS`, `ΔE`, `C`, determine τ; report ACER/EER.
6. Inference & Packaging Agent: export model; provide API and latency benchmarks.
7. Documentation Agent: finalize quickstart, config tips, and deployment notes.
8. (Optional) Feature/Classifier Agent: build φ-vector classifier and fusion; provide calibrated outputs.

---

## Definitions of Done (DoD) Checklist

- Reproducible training on live-only data with stable losses and validated checkpoints.
- Documented anomaly scoring and threshold selection with rationale.
- Passing unit tests for all critical components and filters.
- Inference package meets latency and memory budgets on target hardware.
- Comprehensive README and AGENTS documentation kept current with changes.
- Optional enhancements gated behind configs with ablation results.

---