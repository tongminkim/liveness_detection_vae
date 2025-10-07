# AGENTS.md — FIR Tri-Band VAEs with Latent Integration, Reconstructing in TCN Feature Space (H)

This plan defines a one-class, time-domain-only liveness/deepfake detection system that:
- Splits the landmark time series with fixed FIR filters into LF/BP/HF bands.
- Encodes sequences with a shared TCN into a feature sequence H[t,:].
- Trains three VAEs (one per band) to reconstruct the encoder features (H), not the raw time-domain coordinates.
- Integrates the three band latents at the final stage to reconstruct the full-band encoder features (H_raw).
- Computes all reconstruction and consistency losses strictly in the encoder feature space (H).

We do NOT use frequency-domain features or losses. All constraints are time-domain sequences (in feature space).

--------------------------------------------------------------------------------

## 1) Architecture Overview (H-space Reconstruction)

Notation
- x ∈ R[B, T, K, 2]: lip landmarks; reshape to x ∈ R[B, T, C], C=K×2 (optionally concat Δ, Δ² → C’)
- FIR split (fixed, time-domain only): x → {x_LF, x_BP, x_HF}
- Shared TCN Encoder E(·): maps a time sequence to feature sequence H ∈ R[B, T, D]
- Teacher-Student Encoders (for stable H targets):
  - Teacher encoder E_τ: EMA of student encoder E; produces H* targets; stop-gradient
  - Student encoder E: feeds band VAEs; updated by backprop

Targets (no gradients through targets)
- H_raw* = E_τ(x)              ∈ R[B, T, D]
- H_LF*  = E_τ(x_LF)           ∈ R[B, T, D]
- H_BP*  = E_τ(x_BP)           ∈ R[B, T, D]
- H_HF*  = E_τ(x_HF)           ∈ R[B, T, D]

Band encodings for VAE inputs (student)
- H_LF_in  = E(x_LF)           ∈ R[B, T, D]
- H_BP_in  = E(x_BP)           ∈ R[B, T, D]
- H_HF_in  = E(x_HF)           ∈ R[B, T, D]

Band VAEs (time-sequence VAEs in H-space)
- LF-VAE:
  - Posterior q(z_LF | H_LF_in) with clip-level latent z_LF ∈ R[d_LF] (e.g., 32)
  - Decoder D_LF(z_LF, T) → Ĥ_LF ∈ R[B, T, D] (reconstructs H_LF*)
  - Structural bias: smoothness over time (low-frequency dynamics)
- BP-VAE:
  - Posterior q(z_BP,t | H_BP_in) with frame/window latents z_BP,t ∈ R[d_BP] (e.g., 32–48)
  - Decoder D_BP({z_BP,t}) → Ĥ_BP ∈ R[B, T, D] (reconstructs H_BP*)
  - Structural bias: moderate smoothness and jerk allowance
- HF-VAE:
  - Posterior q(z_HF,t | H_HF_in) with frame latents z_HF,t ∈ R[d_HF] (e.g., 64)
  - Decoder D_HF({z_HF,t}) → Ĥ_HF ∈ R[B, T, D] (reconstructs H_HF*)
  - Structural bias: preserve rapid small-amplitude variations (avoid oversmoothing)

Latent-level Integration (late fusion in latent space)
- Fuse z = {z_LF, z_BP, z_HF} → z_INT (clip-level; concatenate + small MLP or gated fusion)
- Consistency decoder D_INT(z_INT, T) → Ĥ_INT ∈ R[B, T, D]
  - Ĥ_INT should approximate H_raw* (global encoded full-band target)

Optional mixing head (to relate band reconstructions to integrated path)
- Ĥ_MIX = G_MIX(Ĥ_LF, Ĥ_BP, Ĥ_HF) ∈ R[B, T, D] (e.g., 1×1 temporal conv over feature channels)
  - Encourages compatibility between band-wise reconstructions and integrated reconstruction

Streaming note
- For online use, employ causal FIR filters (with delay compensation) and causal TCN; teacher E_τ is kept synchronized (EMA) and used for targets at inference as well.

--------------------------------------------------------------------------------

## 2) Agents, Responsibilities, and Deliverables

### 0) Program Manager (PM) Agent
- Responsibilities: Scope, milestones, dependencies; weekly reviews; risk/change control.
- Deliverables: Roadmap, RACI, status dashboards.
- Acceptance: Milestones met; blockers resolved within SLA; scope/quality maintained.

### 1) Data Agent
- Responsibilities:
  - Prepare x ∈ R[B,T,K,2] (and optional Δ,Δ²) with normalization; windowing T≈48–64 at 25–30 fps.
  - Train/dev/test splits (live-only for train; live+spoof for eval), domain partitions.
  - Time-domain augmentations: ±1 frame jitter, 5–10% frame drop, 0.95–1.05× speed.
- Deliverables: Reproducible datasets (.npy/.pt), normalization stats, augmentation configs.
- Acceptance: Deterministic splits, validated stats, consistent FPS/T across sets.

### 2) FIR/Signal Processing Agent
- Responsibilities:
  - Design fixed FIR filters for LF: 0–2.5 Hz, BP: 2.5–6 Hz, HF: >6 Hz (up to f_N); Hamming taps N≈127.
  - Ensure matched group delay across bands; symmetric padding offline, causal buffers for streaming.
  - Provide utilities for time derivatives (Δ, Δ², jerk) in feature space and in inputs as needed.
- Deliverables: Filter kernels, validation on synthetic sweeps, delay alignment report.
- Acceptance: Minimal band leakage; verified delay; deterministic outputs.

### 3) Model Architect Agent
- Responsibilities:
  - Implement teacher-student encoders E_τ (EMA) and E (student) with TCN (dilations [1,2,4,8], channels 64→128).
  - Build LF/BP/HF VAEs in H-space and the integration module (z_INT + D_INT).
  - Provide a small mixing head G_MIX (optional) for band-to-integrated alignment.
- Deliverables: Modular PyTorch components, unit tests (shape/grad), config presets.
- Acceptance: Stable forward pass; parameter budget ≲ 2M; clean interfaces.

### 4) Loss & Training Agent (H-space only)
- Responsibilities:
  - Targets: Compute H_* with teacher E_τ and stop-gradient: H_raw*, H_LF*, H_BP*, H_HF*.
  - Band reconstruction losses (H-space):
    - L_rec^LF = ||Ĥ_LF − H_LF*||_p
    - L_rec^BP = ||Ĥ_BP − H_BP*||_p
    - L_rec^HF = ||Ĥ_HF − H_HF*||_p
      - p ∈ {1,2}, default p=1 (robust)
  - KL terms (β-VAE style, with warmup 10–20 epochs):
    - β_LF (strong, e.g., 4–8), β_BP (moderate, 1–3), β_HF (weak, 0.5–1.0)
  - Global integration consistency:
    - L_int_raw = ||Ĥ_INT − H_raw*||_1
    - Optional band-sum compatibility via mixing:
      - Ĥ_MIX = G_MIX(Ĥ_LF, Ĥ_BP, Ĥ_HF)
      - L_int_mix = ||Ĥ_INT − Ĥ_MIX||_1
  - Temporal dynamics alignment in H-space:
    - L_grad = Σ_{k=1..3} λ_k · ||∇_t^k(Ĥ_INT) − ∇_t^k(H_raw*)||_1
      - Optionally add per-band gradient alignment on Ĥ_b vs H_b*
  - Band role separation (time-domain decorrelation in H-space):
    - L_decor = γ · [corr_t(Ĥ_LF, Ĥ_BP)^2 + corr_t(Ĥ_LF, Ĥ_HF)^2 + corr_t(Ĥ_BP, Ĥ_HF)^2]
      - corr_t computed across time and features (batch-averaged)
    - Optional latent cross-cov penalty among {z_LF, z_BP, z_HF} (off-diagonal Barlow Twins style)
  - Live-only jitter consistency (H-space):
    - For a lightly jittered x′ (±1f, drop), compute Ĥ_b(x′) and enforce:
      - L_jitter = ||Ĥ_BP(x′) − Ĥ_BP(x)||_1 + ||Ĥ_HF(x′) − Ĥ_HF(x)||_1 (small weight)
  - Total loss (example weights; tune on dev):
    - L_total = 1.0·L_rec^LF + 0.8·L_rec^BP + 0.6·L_rec^HF
             + β_LF·KL_LF + β_BP·KL_BP + β_HF·KL_HF
             + 0.5·L_int_raw + 0.2·L_int_mix
             + 0.3·L_grad + 0.1·L_decor + 0.1·L_jitter
  - Optimization:
    - AdamW (lr 1e-3), cosine decay, batch 16–32, AMP, grad clipping; EMA decay for E_τ (e.g., 0.99→0.999 schedule).
- Deliverables: Training loop with logging of each loss component; checkpoints; EMA management.
- Acceptance: Stable convergence; no posterior collapse; smooth loss curves; reproducible runs.

### 5) Metrics & Evaluation Agent (H-space scoring)
- Responsibilities:
  - Compute errors in H-space (mean, p90, max over time and features):
    - E_LF = ||H_LF* − Ĥ_LF||, E_BP = ||H_BP* − Ĥ_BP||, E_HF = ||H_HF* − Ĥ_HF||
    - E_INT = ||H_raw* − Ĥ_INT||
  - Ratios and sensitivity:
    - R_HF = E_HF / (E_LF + ε), R_BP = E_BP / (E_LF + ε)
    - ΔE_BP = E_BP(x′) − E_BP(x), ΔE_HF = E_HF(x′) − E_HF(x) (jitter sensitivity)
  - Final score S (example):
    - S = w_LF·E_LF + w_BP·E_BP + w_HF·E_HF + u·E_INT + v·ΔE_BP + u’·ΔE_HF
  - Thresholding and metrics:
    - Pick τ via min-ACER or target FPR; report APCER, BPCER, ACER, EER, ROC-AUC/PR-AUC.
  - Ablations:
    - With/without L_decor, L_int_mix, jitter consistency; window size T variations; causal vs non-causal encoders.
- Deliverables: Evaluation scripts/reports; τ selection rationale; confidence intervals.
- Acceptance: Reproducible numbers; clear operating points; ablation insights.

### 6) Inference & Packaging Agent
- Responsibilities:
  - Export model including E (student), E_τ (teacher target encoder), VAEs, and integrator.
  - Real-time path: causal FIR + causal TCN; window T≈48–64, hop 8–16; online EMA for E_τ retained.
  - APIs:
    - preprocess(x) → {x_LF, x_BP, x_HF}
    - encode_teacher({x, x_LF, x_BP, x_HF}) → {H_raw*, H_LF*, H_BP*, H_HF*}
    - forward(x) → {Ĥ_LF, Ĥ_BP, Ĥ_HF, Ĥ_INT, S}
  - Benchmarks: CPU-first latency, memory footprint, streaming behavior.
- Deliverables: Packaged runtime (TorchScript/ONNX), CLI tools, latency report.
- Acceptance: Meets latency budgets; numerical parity with training within tolerance.

### 7) MLOps Agent
- Responsibilities: Repo structure, unit/integration tests, experiment tracking, artifact registry, env lockfiles; CI (lint/tests, smoke-train).
- Deliverables: CI workflows; dataset/model versioning; release playbooks; seed management.
- Acceptance: Green CI; deterministic runs; traceable artifacts.

### 8) Documentation Agent
- Responsibilities: Maintain this plan; architecture diagrams; config cookbook; quickstart; troubleshooting.
- Deliverables: Up-to-date READMEs; examples and best practices.
- Acceptance: New engineer can train/evaluate baseline in < 1 day.

### 9) Optional Feature/Classifier Agent (semi-supervised enhancement)
- Responsibilities:
  - Build clip-level feature φ purely from H-space errors and stats:
    - {E_LF, E_BP, E_HF, E_INT} summaries; ratios R_HF, R_BP; ΔE features; L_decor magnitude proxies.
  - Train small classifier (logistic/MLP) with few-shot spoof labels; calibrate; fuse with S.
- Deliverables: φ extractor; classifier training; calibration; fusion recipe.
- Acceptance: Improves ACER/EER without overfitting (dropout/L2, K-fold domain splits).

--------------------------------------------------------------------------------

## 3) Cross-Agent Handoffs

1) Data → FIR/Signal:
- x sequences, FPS, normalization stats.

2) FIR/Signal → Model Architect:
- FIR kernels, delay compensation notes, derivative ops.

3) Model Architect → Loss & Training:
- Encoders (E, E_τ), VAEs, integrator, mixing head; forward interfaces.

4) Loss & Training → Metrics:
- Checkpoints; loss traces; dev curves; EMA logs.

5) Metrics → PM + Inference:
- Thresholds τ; operating points; ablation decisions.

6) Inference → MLOps + Documentation:
- Exported artifacts; latency/memory benchmarks; API docs.

7) (Optional) Loss & Metrics → Feature/Classifier:
- φ hooks; dev labels; calibration sets.

--------------------------------------------------------------------------------

## 4) Milestones and Sprints

- M1 (Week 1): Data ready; FIR validated (delay-aligned; unit tests).
- M2 (Week 2): Teacher-student encoders + 3 H-space VAEs + integrator; forward stable.
- M3 (Weeks 3–4): H-space loss suite wired; KL warmup; stable training; no collapse.
- M4 (Week 5): H-space scoring, τ selection; baseline ACER/EER; ablations.
- M5 (Week 6): Inference packaging (teacher included); latency OK; docs pass.
- M6 (Weeks 7–8, optional): φ-classifier; fusion; domain robustness report.

--------------------------------------------------------------------------------

## 5) Acceptance Criteria (Baseline)

- Functional:
  - E_τ provides stable H targets; three VAEs reconstruct H_LF*, H_BP*, H_HF*.
  - Latent integrator reconstructs H_raw* (Ĥ_INT), consistent with band reconstructions.

- Quality:
  - Target ACER/EER met on dev; τ documented; ablations justify design choices.
  - Robust under mild jitter/frame-drop; stable across realistic FPS variations.

- Performance:
  - Params ≲ 2M; CPU inference within window latency budget.

- Reproducibility:
  - Deterministic training/inference; CI green; versioned artifacts; fixed seeds.

--------------------------------------------------------------------------------

## 6) Default Configs and Practical Tips

- FIR:
  - Taps N=127, Hamming; LF: 0–2.5 Hz; BP: 2.5–6 Hz; HF: >6 Hz (tune by FPS).
  - Offline: symmetric padding (near-zero phase). Online: causal with matched delay buffers.

- Encoders:
  - TCN: dilations [1,2,4,8], kernel size 3, channels 64→128, residual connections.
  - Teacher E_τ uses EMA decay ∈ [0.99, 0.999]; compute H* with stop-gradient.

- Latents:
  - d_LF=32 (clip), d_BP=32–48 (frame/window), d_HF=64 (frame).
  - z_INT from [z_LF, pooled z_BP, pooled z_HF] via small MLP or gated fusion.

- Loss weights (starting point):
  - As in Section 4; adjust on dev; keep HF regularization light to preserve detail.

- Scoring:
  - Use E_b and E_INT summaries in H-space; include jitter sensitivity terms.
  - Choose τ by min-ACER or target FPR; calibrate if needed.

--------------------------------------------------------------------------------

## 7) Risks and Mitigations

- Target drift / trivial solutions:
  - Use teacher E_τ (EMA) for fixed targets H* (stop-gradient); do not backprop into E_τ.
  - Optionally prewarm E with live-only autoencoding before VAE training.

- Posterior collapse:
  - KL warmup/free-bits; β-balancing (stronger on LF, weaker on HF); residual connections in decoders.

- Band leakage or misalignment:
  - Verify FIR group delays; enforce L_decor among Ĥ_b; use G_MIX to reconcile band → global path.

- Over-smoothing HF features:
  - Avoid aggressive smoothness on HF; prefer mild amplitude regularization.

- Domain drift:
  - Maintain per-domain τ; monitor E_INT drift; recalibrate periodically.

--------------------------------------------------------------------------------

## 8) Quickstart Responsibilities

- Data: Provide x, FPS, normalization; windowing config.
- FIR/Signal: Deliver x_LF, x_BP, x_HF utilities; delay compensation; derivative ops.
- Model: Implement E, E_τ (EMA), LF/BP/HF VAEs, integrator, optional G_MIX.
- Training: Wire H-space losses; KL warmup; checkpoints; logging.
- Metrics: Compute H-space scores; select τ; report ACER/EER and ROC/PR.
- Inference: Export full graph (incl. teacher); verify latency; provide CLI.
- Docs: Keep quickstart/configs current; troubleshooting common pitfalls.

--------------------------------------------------------------------------------

## 9) Definition of Done (DoD)

- End-to-end training converges with stable H-space reconstructions per band and strong Ĥ_INT vs H_raw* alignment.
- Latent integration improves or matches band-wise-only performance on dev metrics.
- Unit tests cover FIR alignment, encoder/decoder shapes, loss invariants, EMA behavior.
- Inference meets latency/memory budgets; outputs match training within tolerance.
- Documentation enables a newcomer to reproduce baseline metrics in < 1 day.
