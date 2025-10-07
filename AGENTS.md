# AGENTS.md — Time-Domain-Only FIR Tri-Band VAE Plan (LF/BP/HF) with Latent-Level Integration

This document defines the multi-agent collaboration plan to deliver a one-class, time-domain-only liveness/deepfake detector using three VAEs operating on FIR-filtered subband sequences: Low (LF), Band-Pass (BP), and High (HF). We explicitly avoid frequency-domain (e.g., STFT) losses. All supervision and consistency constraints are defined in the time domain.

Scope summary:
- Input: lip landmark time series x ∈ R[B, T, K, 2] → reshape to [B, T, C], C = K×2 (optionally concat Δ, Δ²).
- Fixed, non-learned FIR filters split x → x_LF, x_BP, x_HF (time-domain subbands).
- Three VAEs (LF-VAE, BP-VAE, HF-VAE), each trained to reconstruct its subband target in time domain.
- Latent-level integration: final-stage latent fusion of z_LF, z_BP, z_HF → z_INT; small consistency decoder enforces global time-domain reconstruction and cross-band coherence.
- No frequency-domain features or losses (no STFT/CQT/wavelets in loss). All constraints are time-domain (reconstruction, derivatives, decorrelation, smoothness/sparsity, jitter consistency).
- One-class training (live only). Scoring is reconstruction-based from time-domain errors.

--------------------------------------------------------------------------------

## Architecture at a Glance (Time Domain Only)

- Preprocessing:
  - Normalize coordinates per sequence or globally (mean/var).
  - Optional lightweight smoothing (EMA/Kalman); avoid over-smoothing (preserve HF).
  - Optional temporal derivatives Δx, Δ²x as extra channels.

- FIR tri-band split (fixed, time domain):
  - FPS = f, Nyquist f_N = f/2.
  - Recommended bands:
    - LF: 0–2.5 Hz (low-pass)
    - BP: 2.5–6 Hz (band-pass)
    - HF: >6 Hz (high-pass) up to f_N
  - Filters: Hamming-windowed FIR, taps N≈127 (balanced delay across bands). Offline: zero-phase via symmetric padding; Streaming: matched group delay buffers.

- Encoder and VAEs:
  - Shared time encoder (TCN) over x or over per-band inputs; heads can be conditioned on shared H[T,D] for stability.
  - LF-VAE:
    - Clip-level latent z_LF ∈ R[d_LF] (e.g., 32).
    - Time-domain decoder reconstructs x̂_LF ≈ x_LF.
    - Structure regularization favors smooth trajectories.
  - BP-VAE:
    - Frame-level or short-window latent z_BP,t ∈ R[d_BP] (e.g., 32–48).
    - Time-domain decoder reconstructs x̂_BP ≈ x_BP.
    - Regularization for mid-scale dynamics (moderate smoothness, moderate jerk allowance).
  - HF-VAE:
    - Frame-level latent z_HF,t ∈ R[d_HF] (e.g., 64).
    - Time-domain decoder reconstructs x̂_HF ≈ x_HF.
    - Regularization emphasizes rapid, small-amplitude variations (encourage detail, avoid oversmoothing).

- Latent-level integration (late fusion):
  - Integrator combines {z_LF, z_BP, z_HF} → z_INT (clip-level, or short sequence aligned to T).
  - Fusion strategies (configurable):
    - Concatenate + MLP (default, simple and robust).
    - Product-of-Experts Gaussian (for clip-level fusion).
    - Gated fusion/attention over per-band latents (optional).
  - Consistency decoder (small TCN/GRU) maps z_INT → x̂_INT ∈ R[B, T, C] to enforce final global reconstruction consistency in time domain.

- Composition constraint:
  - Band sum consistency: x̂_SUM = x̂_LF + x̂_BP + x̂_HF should approximate x (time-domain).
  - Optional: x̂_INT should also approximate x; use consistency terms to align x̂_INT and x̂_SUM.

--------------------------------------------------------------------------------

## Agents, Responsibilities, and Deliverables

### 0) Program Manager (PM) Agent
- Responsibilities:
  - Define scope, milestones, and dependencies across agents.
  - Maintain risk register, weekly reviews, and change control.
- Deliverables:
  - Roadmap, RACI, progress dashboards.
- Acceptance:
  - Milestones met, blockers triaged within SLA, scope/quality maintained.

### 1) Data Agent
- Responsibilities:
  - Collect/curate video datasets; extract lip landmark sequences x ∈ R[B, T, K, 2].
  - Normalization stats, optional Δ/Δ² features; sequence windowing (T≈48–64).
  - Data splits: train (live-only), dev/test (live+spoof if available), domain partitions.
  - Augmentation (time domain): small jitter (±1f), 5–10% frame drop, speed 0.95–1.05×.
- Deliverables:
  - Reproducible datasets (.npy/.pt), metadata (FPS, T, K), augmentation configs.
- Acceptance:
  - Deterministic splits; coverage of lip landmarks; validated sequence statistics.

### 2) FIR/Signal Processing Agent
- Responsibilities:
  - Design and implement fixed FIR filters for LF/BP/HF (time domain only).
  - Ensure matched delay across bands; provide zero-phase offline variant and causal streaming variant.
  - Provide utilities for Δ, Δ² computation, total variation, and time-derivative ops.
- Deliverables:
  - Filter kernels/taps, reference plots (impulse/step response), and validation checks (band isolation on synthetic signals).
- Acceptance:
  - Minimal band leakage; consistent alignment across bands; reproducible, tested utilities.

### 3) Model Architect Agent
- Responsibilities:
  - Implement shared time encoder (TCN) and three VAE heads (LF/BP/HF).
    - Shared encoder: dilations [1,2,4,8], channels 64→128 (configurable).
    - LF-VAE (clip-latent), BP-VAE (frame/window latent), HF-VAE (frame latent).
    - Heads accept band-specific inputs (x_LF, x_BP, x_HF) and may condition on shared H[T,D].
  - Implement latent-level integrator:
    - Fusion of {z_LF, z_BP, z_HF} → z_INT (clip-level by default).
    - Small consistency decoder mapping z_INT → x̂_INT.
- Deliverables:
  - Modular PyTorch components with clear interfaces and unit tests.
- Acceptance:
  - Shape correctness, parameter budget reasonable (<~2M baseline), stable forwards.

### 4) Loss & Training Agent (Time Domain Only)
- Responsibilities:
  - Define time-domain-only objectives (no frequency-domain losses):
    - Band reconstructions:
      - L_rec^LF = ||x̂_LF − x_LF|| (L1 or L2)
      - L_rec^BP = ||x̂_BP − x_BP||
      - L_rec^HF = ||x̂_HF − x_HF||
    - KL terms (β-VAE style, with warmup 10–20 epochs):
      - Strong β_LF (e.g., 4–8) to enforce low-capacity smooth representation.
      - Moderate β_BP (e.g., 1–3).
      - Weak β_HF (e.g., 0.5–1.0) to preserve fine details.
    - Band sum consistency:
      - L_mix = ||x − (x̂_LF + x̂_BP + x̂_HF)||₁
    - Integration consistency (latent fusion):
      - L_int = ||x̂_INT − x||₁  and/or  ||x̂_INT − (x̂_LF + x̂_BP + x̂_HF)||₁
    - Temporal derivative alignment (time-domain dynamics):
      - L_grad = Σ_{k=1..3} λ_k · ||∇_t^k x̂ − ∇_t^k x||₁   (applied to x̂_SUM or x̂_INT)
    - Band role separation (time-domain decorrelation):
      - L_decor = γ · [corr(x̂_LF, x̂_BP)^2 + corr(x̂_LF, x̂_HF)^2 + corr(x̂_BP, x̂_HF)^2] (computed over time and channels)
      - Optional latent cross-covariance penalty between {z_LF, z_BP, z_HF} (Barlow Twins off-diagonal style) to reduce redundancy.
    - Structure regularizers:
      - LF smoothness: TV or second-derivative penalty on x̂_LF (small weight).
      - HF sparsity-of-amplitude and jerk allowance: small L1 on x̂_HF, but no heavy smoothing (preserve detail).
      - BP moderate smoothness (between LF and HF).
    - Jitter consistency (live-only augmentation):
      - L_jitter = ||x̂_HF(x′) − x̂_HF(x)||₁ + ||x̂_BP(x′) − x̂_BP(x)||₁ (x′ has small temporal jitter/drop)
  - Training setup:
    - AdamW (lr 1e-3), cosine decay, batch 16–32, AMP, gradient clipping; early stopping on dev loss.
- Deliverables:
  - Loss composition with documented weights; training loop with robust logging and checkpoints.
- Acceptance:
  - Stable optimization; no posterior collapse; monotone loss trends; reproducible runs.

### 5) Metrics & Evaluation Agent
- Responsibilities:
  - One-class scoring (time-domain metrics only):
    - E_LF = ||x_LF − x̂_LF|| (mean/p90/max over time)
    - E_BP = ||x_BP − x̂_BP||
    - E_HF = ||x_HF − x̂_HF||
    - Ratios: E_HF/(E_LF+ε), E_BP/(E_LF+ε)
    - ΔE under jitter: E_HF(x′) − E_HF(x), E_BP(x′) − E_BP(x)
    - Optional complexity scores computed in time domain (e.g., sample entropy of residuals)
  - Final score S (example):
    - S = w_LF·E_LF + w_BP·E_BP + w_HF·E_HF + u·ΔE_BP + v·ΔE_HF  (weights tuned on dev)
  - Protocols and metrics:
    - Threshold τ set on dev via min-ACER or target FPR; report APCER/BPCER/ACER, EER, ROC-AUC/PR-AUC.
- Deliverables:
  - Evaluation scripts, reports per domain, ablations (with/without latent integration, with/without decorrelation, etc.).
- Acceptance:
  - Reproducible numbers with CI; clear τ selection; ablation justifications.

### 6) Inference & Packaging Agent
- Responsibilities:
  - Export inference graph (TorchScript/ONNX) for time-domain only pipeline.
  - Real-time option: causal TCN and causal FIR with compensated delay; streaming windowing (T≈48–64, hop 8–16).
  - Provide APIs:
    - preprocess(x) → {x_LF, x_BP, x_HF}
    - forward(x) → {x̂_LF, x̂_BP, x̂_HF, x̂_INT, S}
- Deliverables:
  - Inference module with CLI; latency/throughput benchmarks on target hardware (CPU-first).
- Acceptance:
  - Latency within budget; numerical parity to training within tolerance.

### 7) MLOps Agent
- Responsibilities:
  - Repo structure; unit tests; experiment tracking; artifact versioning; env lockfiles.
  - CI: lint/tests on commits; nightly training smoke tests.
- Deliverables:
  - CI workflows, dataset/model registry conventions, release playbooks.
- Acceptance:
  - Green CI; deterministic training/inference; traceable artifacts.

### 8) Documentation Agent
- Responsibilities:
  - Maintain this AGENTS plan, architecture overview, configuration cookbook, and quickstart.
- Deliverables:
  - Up-to-date READMEs; diagrams; example configs; troubleshooting guide.
- Acceptance:
  - New engineer can train and evaluate baseline in <1 day.

### 9) Optional Feature/Classifier Agent (if small spoof set available)
- Responsibilities:
  - Build clip-level feature vector φ strictly from time-domain signals and errors:
    - [E_LF, E_BP, E_HF] stats (mean/p90/max), ratios, ΔE features, smoothness/sparsity penalties magnitudes, decorrelation terms.
  - Train small classifier (logistic/MLP) and calibrate; fuse with one-class score if beneficial.
- Deliverables:
  - φ extractor, classifier training, calibration, and fusion recipe.
- Acceptance:
  - Improves ACER/EER on dev; avoids overfitting (dropout/L2, K-fold).

--------------------------------------------------------------------------------

## Cross-Agent Handoffs

1) Data → FIR/Signal:
- Landmark sequences, FPS, normalization stats.

2) FIR/Signal → Model Architect:
- FIR kernels, delay characteristics, derivative utilities.

3) Model Architect → Loss & Training:
- Modules ready; forward interfaces; parameter budget.

4) Loss & Training → Metrics:
- Checkpoints and logs; loss component magnitudes; dev curves.

5) Metrics → PM + Inference:
- Thresholds τ, operating points, ablation insights.

6) Inference → MLOps + Documentation:
- Exported models, benchmarks, quickstart API docs.

7) (Optional) Loss & Metrics → Feature/Classifier:
- Feature hooks and dev labels for semi-supervised classifier.

--------------------------------------------------------------------------------

## Milestones and Sprints

- M1 (Week 1): Data ready; FIR kernels validated (delay-aligned; unit tests).
- M2 (Week 2): Shared encoder + 3 VAEs implemented; forward pass stable.
- M3 (Weeks 3–4): Time-domain loss suite wired; training stable; no collapse.
- M4 (Week 5): Scoring + thresholding; baseline ACER/EER; ablations.
- M5 (Week 6): Inference packaging; latency OK; docs pass.
- M6 (Weeks 7–8, optional): Classifier φ-features; fusion; domain robustness report.

--------------------------------------------------------------------------------

## Acceptance Criteria (Baseline)

- Functional:
  - Three VAEs reconstruct LF/BP/HF bands from FIR-split sequences in time domain.
  - Latent integrator produces z_INT and x̂_INT; compositions are consistent.

- Quality:
  - Meets target ACER/EER on dev; τ documented; ablations justify design.
  - Robust to mild jitter/frame-drop; stable across reasonable FPS variations.

- Performance:
  - Parameter count ≲ 2M; CPU inference within window latency budget (T≈48–64).

- Reproducibility:
  - Deterministic training/inference; CI green; artifacts versioned.

--------------------------------------------------------------------------------

## Default Configs and Practical Tips

- FIR:
  - Taps N=127, Hamming; cutoffs: LF 2.5 Hz, BP 2.5–6 Hz, HF >6 Hz (tune by FPS).
  - Offline: symmetric padding (near-zero phase). Streaming: causal FIR + delay compensation.

- Encoder (TCN):
  - Dilations [1,2,4,8], kernel size 3, channels 64→128, residual connections.
  - Causal=False for offline; True for streaming.

- Latents:
  - d_LF=32 (clip-level), d_BP=32–48 (frame/window), d_HF=64 (frame).

- Loss weights (starting point, tune on dev):
  - L = 1.0·L_rec^LF + 0.8·L_rec^BP + 0.6·L_rec^HF
      + 0.3·L_grad + 0.5·L_mix + 0.3·L_int
      + β_LF·KL_LF + β_BP·KL_BP + β_HF·KL_HF
      + 0.1·L_decor + 0.1·(LF smooth) + 0.05·(HF sparsity) + 0.1·L_jitter

- Scoring:
  - S = w_LF·E_LF + w_BP·E_BP + w_HF·E_HF + u·ΔE_BP + v·ΔE_HF
  - Choose τ via min-ACER or target FPR on dev.

--------------------------------------------------------------------------------

## Risks and Mitigations

- Band leakage or misalignment:
  - Equalize group delays; verify with synthetic sweeps; enforce time-domain decorrelation; rely on L_mix and L_int to maintain decomposition.

- Posterior collapse:
  - KL warmup, free-bits if needed; stronger β on LF, weaker on HF.

- Over-smoothing HF:
  - Avoid strong smoothness on HF; use gentle amplitude regularization only.

- Domain drift:
  - Recalibrate τ per environment; maintain dev-set checkpoints per domain.

--------------------------------------------------------------------------------

## Quickstart Responsibilities

- Data: Provide x, FPS, normalizers, and windowing.
- FIR/Signal: Deliver x_LF, x_BP, x_HF utilities aligned in time; derivative ops.
- Model: Implement shared encoder, 3 VAEs, latent integrator, consistency decoder.
- Training: Wire losses (time-domain only), schedules, logging, checkpoints.
- Metrics: Compute scores, pick τ, produce ACER/EER and ROC/PR.
- Inference: Export models, verify latency; CLI tools for batch/streaming.
- Docs: Quickstart and configs kept current; troubleshooting for common pitfalls.

--------------------------------------------------------------------------------

## Definition of Done (DoD)

- End-to-end training on live-only data converges with stable, interpretable band reconstructions.
- Latent integration improves or matches band-sum consistency; both paths reconstruct x.
- Unit tests cover FIR alignment, encoder/decoder shapes, loss invariants.
- Inference meets latency/memory budgets; outputs match training within tolerance.
- Documentation enables a newcomer to reproduce baseline metrics in <1 day.
