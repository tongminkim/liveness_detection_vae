# AGENTS.md — FIR Tri-Band VAEs with Latent Fusion, Reconstructing in TCN Feature Space (H)

This document defines the plan to build a one-class, time-domain-only liveness/deepfake detector that:
- Splits the landmark time series into LF/BP/HF bands using fixed FIR filters (no frequency-domain losses).
- Encodes each band with a shared TCN into feature sequences h_b[t,:] in feature space H.
- Trains three VAEs (one per band) to reconstruct their band’s encoder features ĥ_b (not raw coordinates).
- Fuses reconstructed band features via a learnable projection (h_mix = Σ_b W_b ĥ_b) to match full-band features h.
- Uses Stage 2 fine-tuning options for few-shot spoof data (margin/center/BCE), while keeping the core model one-class.

Aligned with:
- readme6: H-space reconstruction loss suite and projection-based fusion.
- readme7: Reproducible data preprocessing pipeline (segmentation, normalization, z-score, saving).


--------------------------------------------------------------------------------
## 0) Executive Summary

- Time-domain only: all constraints and supervision are on sequences in the encoder feature space H.
- Three VAEs (LF/BP/HF) operate on FIR-filtered inputs; each reconstructs its own band’s H features.
- A lightweight fusion head combines band-wise reconstructions to match full-band H (consistency).
- Stage 1: Live-only training with reconstruction + KL + fusion-consistency + band decorrelation.
- Stage 2 (optional): Few-shot spoof fine-tuning with margin/center/BCE objectives on top of Stage 1.


--------------------------------------------------------------------------------
## 1) Data Pipeline (from readme7)

Input
- Video(s) with multiple utterances at fixed fps.
- JSON containing utterance-level start_time and end_time in seconds.

Segmentation (per utterance)
- Convert timestamps to frame indices: start_idx = floor(start_time * fps), end_idx = floor(end_time * fps).
- Extract frames for [start_idx, end_idx).

Landmark extraction (per frame)
- Use a face landmark detector (e.g., MediaPipe/OpenFace/Dlib).
- Extract:
  - Lip landmarks (e.g., 20–30 points; ensure indices consistent).
  - Normalization references:
    - C_t: face center (e.g., nose tip or mouth center).
    - E_L, E_R: baseline points (e.g., eye or mouth corners) for orientation/scale.
- Handle failed frames with linear interpolation to maintain sequence continuity.

Per-frame geometric normalization (roll/scale/translation)
- For each landmark X_{t,k} (2D XY):
  - Rotate to correct roll using R(θ_t) defined by E_L, E_R.
  - Translate by subtracting C_t.
  - Scale by dividing with ||E_R - E_L|| (face width).
  - X_norm_{t,k} = R(θ_t) (X_{t,k} - C_t) / ||E_R - E_L||.

Global z-score normalization (dataset-level)
- Compute mean and std over TRAIN split only (per coordinate/channel).
- Apply z = (X_norm - μ_train) / σ_train to train/dev/test (use train stats).

Saving (per utterance)
- Save normalized landmarks as .npy arrays with shape (T, K, 2).
- Optionally use .npz with metadata (utterance id, fps, indices).

Reproducibility guidelines
- Fix detector settings (fps, thresholds, smoothing); keep deterministic rounding and transforms.
- Maintain separate train/dev/test splits; compute μ/σ on train only.


--------------------------------------------------------------------------------
## 2) Architecture (from readme6; H-space reconstruction)

Notation
- x ∈ R[B, T, K, 2]: normalized landmarks; reshape to [B, T, C], C = K×2 (optionally concat Δx, Δ²x).
- FIR BandSplit: x → {x_LF, x_BP, x_HF} using fixed FIR filters (Hamming, taps ≈ 127).
- Shared TCN encoder f_TCN(·): maps a time sequence to H-space features [B, T, D].

Feature targets
- h      = f_TCN(x)       ∈ R[B, T, D]      (full-band)
- h_b    = f_TCN(x_b)     ∈ R[B, T, D]      (band target per b ∈ {LF, BP, HF})

Per-band VAEs (operate in H-space)
- For each band b:
  - Encoder: h_b → q(z_b | h_b)
    - LF (clip latent; e.g., dim 32)
    - BP (frame/window latent; e.g., dim 32–48)
    - HF (frame latent; e.g., dim 64)
  - Decoder: z_b → ĥ_b ∈ R[B, T, D]  (reconstruct band’s H target)
  - Structural bias:
    - LF favor smoothness (slow dynamics).
    - BP moderate smoothness and jerk allowance.
    - HF preserve high-frequency detail (avoid oversmoothing).

Projection-based fusion (consistency head)
- h_mix = W_LF ĥ_LF + W_BP ĥ_BP + W_HF ĥ_HF, with learnable W_b ∈ R[D×D].
- Goal: h_mix ≈ h (full-band H) to encourage cross-band compatibility.

Parameter targets
- D (TCN feature dim) ≈ 128.
- Total params ≲ 2M (baseline).


--------------------------------------------------------------------------------
## 3) Loss Functions (H-space only, from readme6)

Per-band reconstruction (robust L1 recommended)
- L_rec = Σ_{b ∈ {LF,BP,HF}} λ_b · ||h_b − ĥ_b||_1

VAE KL regularization
- L_KL = Σ_b β_b · KL(q(z_b | h_b) || N(0, I))

Projection-based fusion consistency
- h_mix = Σ_b W_b ĥ_b
- L_mix = ζ · ||h − h_mix||_1

Band decorrelation (reduce redundancy across reconstructed bands)
- L_decorr = γ · Σ_{i<j} corr(ĥ_i, ĥ_j)^2
  - corr computed over time and feature dims (batch-averaged)

Total loss (Stage 1)
- L_total = L_rec + L_KL + L_mix + L_decorr

Default weights (tune on dev)
- λ_LF/BP/HF = 1.0
- β_LF/BP/HF = 1.0 (can bias β_LF higher, β_HF lower if needed)
- ζ (mix) = 0.5–1.0
- γ (decorr) = 0.1–0.5


--------------------------------------------------------------------------------
## 4) Training Stages (from readme6)

Stage 1 — One-class pretraining (Live only)
- Optimize L_total on live-only data.
- Anomaly score for dev/test:
  - E_b = ||h_b − ĥ_b|| (e.g., mean or p90 over T and D)
  - KL_sum = Σ_b mean KL(q(z_b)||N(0,I)) over time
  - S_ano = w1 · E_LF + w2 · E_BP + w3 · E_HF + w4 · KL_sum
- Choose τ on dev via min-ACER or target FPR.

Stage 2 — Few-shot spoof fine-tuning (optional)
- Initialize from Stage 1; freeze backbone or use small LR for stability.
- Choose one separation head (only one at a time is recommended):
  1) Score Margin Loss:
     - L_margin = E_live[max(0, S(x) − τ_L)] + E_spoof[max(0, τ_S − S(x))]
  2) Feature Center Loss (with clip-level φ features from H-space errors/latents):
     - L_center = ||φ(live) − c||_2^2 + max(0, m − ||φ(spoof) − c||_2)
  3) Binary Classifier (BCE/Focal) on φ:
     - L_BCE = −Σ[y log p + (1 − y) log (1 − p)]
- Stage 2 objective:
  - L_stage2 = L_total (from Stage 1) + ρ · L_{margin|center|BCE}
- Typical ρ: 0.5–1.0 (tune on dev)


--------------------------------------------------------------------------------
## 5) Metrics & Scoring (H-space)

Core error summaries (per clip)
- E_LF, E_BP, E_HF: mean/p90/max of ||h_b − ĥ_b|| over time and D.
- KL_sum: sum/mean of band KLs.
- Optional ratios: E_HF / (E_LF + ε), E_BP / (E_LF + ε).

Final score (examples)
- S_ano = w1 E_LF + w2 E_BP + w3 E_HF + w4 KL_sum
- For Stage 2 classifier: fuse calibrated classifier score with normalized S_ano if helpful.

Evaluation
- Report APCER, BPCER, ACER, EER; ROC-AUC/PR-AUC.
- Pick τ with min-ACER or target FPR on dev; provide domain-wise breakdowns.


--------------------------------------------------------------------------------
## 6) Agents, Responsibilities, and Deliverables

0) Program Manager (PM)
- Responsibilities: Scope, milestones, dependencies; weekly reviews; risks/change control.
- Deliverables: Roadmap, RACI, dashboards.
- Acceptance: Milestones achieved; blockers resolved within SLA; scope/quality maintained.

1) Data Agent (aligns with readme7)
- Responsibilities:
  - Implement utterance segmentation from JSON timestamps.
  - Landmark extraction and failure handling (interpolation).
  - Per-frame roll/scale/translation normalization; train-only z-score.
  - Dataset splits; deterministic saving as .npy/.npz (T,K,2).
- Deliverables: Reproducible datasets, μ/σ stats, pipeline scripts.
- Acceptance: Deterministic outputs; validated stats; consistent FPS/T.

2) FIR/Signal Processing Agent
- Responsibilities:
  - Design FIR filters for LF (0–2.5 Hz), BP (2.5–6 Hz), HF (>6 Hz up to f_N).
  - Ensure matched group delay across bands (Hamming taps ≈ 127).
  - Provide Δx, Δ²x utilities (optional) and windowing helpers.
- Deliverables: Kernels/taps, alignment validation, unit tests.
- Acceptance: Minimal band leakage; verified delay alignment.

3) Model Architect
- Responsibilities:
  - Implement shared TCN encoder f_TCN(x) → H[T,D] (dilations [1,2,4,8], D≈128).
  - Implement 3 VAEs (LF/BP/HF) in H-space; configure latents (LF clip-level; BP/HF frame-level).
  - Implement projection fusion h_mix = Σ_b W_b ĥ_b with learnable W_b.
- Deliverables: Modular PyTorch components; config presets; unit tests.
- Acceptance: Correct shapes; stable forward; params ≲ 2M.

4) Loss & Training
- Responsibilities:
  - Implement L_rec, L_KL, L_mix, L_decorr with logging per component.
  - Training loop (AdamW, lr≈1e-3, cosine decay, batch 16–32, AMP, grad clip).
  - Early stopping on dev; loss weight search.
- Deliverables: Trainer scripts; checkpoints; curves.
- Acceptance: Stable convergence; no collapse; reproducible runs.

5) Metrics & Evaluation
- Responsibilities:
  - Compute E_b, KL_sum; S_ano; choose τ on dev.
  - Stage 2: evaluate margin/center/BCE variants; calibrate if needed.
- Deliverables: Eval reports; threshold selection rationale; ablations.
- Acceptance: Reproducible metrics; clear operating points.

6) Inference & Packaging
- Responsibilities:
  - Export model (TorchScript/ONNX); CPU-first latency; sliding-window infer (T≈48–64, hop 8–16).
  - Provide APIs:
    - preprocess(x) → {x_LF, x_BP, x_HF}
    - encode(x|x_b) → {h, h_LF, h_BP, h_HF}
    - forward(x) → {ĥ_LF, ĥ_BP, ĥ_HF, h_mix, S}
- Deliverables: CLI tools; latency/memory report.
- Acceptance: Meets latency budget; numerical parity with training.

7) MLOps
- Responsibilities: Repo structure; CI (lint/tests, smoke-train); experiment tracking; artifact registry; env lockfiles.
- Deliverables: CI workflows; versioned datasets/models; release playbooks.
- Acceptance: Green CI; deterministic runs; traceable artifacts.

8) Documentation
- Responsibilities: Maintain this plan, architecture notes, preprocessing specs, quickstart, configs, troubleshooting.
- Deliverables: READMEs/diagrams; examples.
- Acceptance: New engineer can reproduce baseline in < 1 day.

9) Optional Feature/Classifier Agent (Stage 2)
- Responsibilities:
  - Build φ from H-space signals (E_b stats, KL_sum, ratios, simple latent summaries).
  - Train a small classifier (logistic/MLP/XGBoost); calibrate and optionally fuse with S_ano.
- Deliverables: φ extractor; classifier training; calibration.
- Acceptance: Improves ACER/EER without overfitting (dropout/L2, K-fold).


--------------------------------------------------------------------------------
## 7) Cross-Agent Handoffs

1) Data → FIR/Signal:
- Normalized sequences (T,K,2), FPS, μ/σ stats.

2) FIR/Signal → Model Architect:
- FIR kernels/taps, delay reports; Δ/Δ² utilities.

3) Model Architect → Loss & Training:
- f_TCN, VAE heads, fusion head; configs; tests.

4) Loss & Training → Metrics:
- Checkpoints, logs, per-loss traces, dev curves.

5) Metrics → PM + Inference:
- τ, operating points, ablation outcomes.

6) Inference → MLOps + Documentation:
- Exported artifacts, API docs, latency benchmarks.

7) (Optional) Loss & Metrics → Feature/Classifier:
- φ hooks, dev labels, calibration set guidance.


--------------------------------------------------------------------------------
## 8) Milestones and Sprints

- M1 (Week 1): Preprocessing pipeline implemented (readme7); FIR kernels validated; unit tests pass.
- M2 (Week 2): TCN + 3 VAEs + fusion head implemented; forward stable; params ≲ 2M.
- M3 (Weeks 3–4): Loss suite (L_rec, L_KL, L_mix, L_decorr) integrated; Stage 1 training stable.
- M4 (Week 5): Scoring and τ selection; baseline ACER/EER reported; ablations complete.
- M5 (Week 6): Inference packaging (CPU-first); latency OK; docs finalized.
- M6 (Weeks 7–8, optional): Stage 2 (margin/center/BCE) fine-tuning; calibration; robustness report.


--------------------------------------------------------------------------------
## 9) Acceptance Criteria

Functional
- Band-wise VAEs reconstruct H-space features (ĥ_LF/BP/HF) close to h_LF/BP/HF.
- Fusion head produces h_mix ≈ h with stable training dynamics.

Quality
- Meets dev-set ACER/EER targets; ablation justifies design choices.
- Robust to mild jitter/frame-drop; stable across reasonable FPS variations.

Performance
- Param count ≲ 2M; CPU inference within window latency budget.

Reproducibility
- Deterministic preprocessing and training; CI green; artifacts versioned.

Documentation
- Clear quickstart and configs; new engineer reproducible in < 1 day.


--------------------------------------------------------------------------------
## 10) Risks and Mitigations

Band leakage/misalignment
- Align group delays across FIR bands; verify with synthetic sweeps; enforce L_decorr and L_mix.

Posterior collapse
- KL warmup or free-bits; balance β across bands (LF ≥ BP ≥ HF); residual connections in decoders.

HF oversmoothing
- Avoid strong smoothness on HF; prefer minimal amplitude regularization.

Data drift/domain shift
- Monitor S_ano distribution; recalibrate τ per domain; Stage 2 few-shot adaptation.

Overfitting in Stage 2
- Freeze or low LR for backbone; strong regularization; K-fold domain splits; early stopping.


--------------------------------------------------------------------------------
## 11) Default Configs and Practical Tips

- FIR: Hamming taps ≈ 127; LF 0–2.5 Hz; BP 2.5–6 Hz; HF >6 Hz (tune by fps).
- TCN: dilations [1,2,4,8], kernel size 3, channels 64→128, residual blocks; D ≈ 128.
- Latents: d_LF=32 (clip), d_BP=32–48 (frame/window), d_HF=64 (frame).
- Optim: AdamW (lr 1e-3), cosine decay, batch 16–32, AMP, grad clipping.
- Loss weights: λ_b=1.0; β_b ∈ [0.5, 2.0] (tune); ζ∈[0.5,1.0]; γ∈[0.1,0.5].
- Windowing: T≈48–64; hop 8–16; smooth scores across hops for stability.
- Scoring: Use E_b and KL_sum summaries; optionally add ratios; choose τ via min-ACER/target FPR.
