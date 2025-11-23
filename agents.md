# Agents Guide

This repository has been trimmed down to code-only assets and is now organized into the functional agents below. Each agent groups the scripts, entry points, and outputs that collaborate to build the Band-Split VAE liveness detection pipeline.

## Agent Map

| Agent | Main Files | Primary Outputs |
| --- | --- | --- |
| Data Acquisition | `preprocess_data*.py`, `preprocess_fake_data.py`, `split_test_balanced.py`, `data_split.json`, `check_progress.sh` | Landmark NPZ files, deterministic train/test splits |
| Dataset Assembly | `dataset.py`, `dataset_bandvae.py`, `dataset_stage2.py`, `config*.py` | Torch-ready tensors (single-stream + band-split) |
| Model Core | `model.py`, `model_bandvae.py`, `seaborn.py` | FeatureSpaceVAE, BandSplitVAE, Butterworth filter bank, loss utilities |
| Stage 1 Training | `train.py`, `train_bandvae.py`, `train_stage1_pretrain.py`, `train_both_bandvae.sh`, `auto_train_simple.sh` | `runs/vae_*` and `runs/stage1_pretrain` checkpoints/logs |
| Stage 2 Learners | `train_stage2_method1_margin.py`, `train_stage2_method2_discriminator.py` | `runs/stage2_method*` fine-tuned checkpoints |
| Evaluation & Filtering | `evaluate_stage2_methods.py`, `evaluate_and_visualize_stage2.py` (`--audiodriven-only`), `evaluate_gaussian_filters.py`, `final.py`, `evaluate_*`, `visualize_*.py` | Metrics JSON, comparison plots, PCA/gaussian filter reports |
| Hyperparameter & Diagnostics | `tune_gaussian*.py`, `tune_hp_T300.py`, `tune_hyperparameters.py`, `analyze_fake_bandloss.py`, `debug_nan.py`, shell helpers | Search sweeps, debugging stats |
| Documentation & Requirements | `README.md`, `PROJECT.md`, `STAGE2_PIPELINE.md`, `HYPERPARAMETER_TUNING.md`, `pip_requirements.txt`, `conda_requirements.txt`, diagrams/images | Human guidance + dependency manifests |

## Data Acquisition Agent

- **Landmark extraction pipelines**  
  - `preprocess_data.py` implements the reference MediaPipe workflow: fixed-length segments, Procrustes normalization, and multi-feature engineering (position, velocity, acceleration, angle, angle-rate).  
  - `preprocess_data_fast.py` mirrors the same logic but uses multiprocessing + pid/log management for large-scale runs.  
  - `preprocess_fake_data.py` mirrors the live-data pipeline for fake sources.  
  - All scripts emit `.npz` packages with lips-only landmarks plus meta-data (`fps`, frame size) that downstream datasets expect.

- **Split management**  
  - `split_test_balanced.py` (see inline documentation) consumes `/home/elicer/.../test_balanced_npz` and writes the canonical `data_split.json` with stage2-train vs final-test partitions plus the fake directory whitelist.
  - `check_progress.sh` is a monitoring helper for long-running preprocess jobs (PID tracking, disk utilization, most recent progress lines).

- **What to run**  
  - Use `python3 preprocess_data_fast.py --input <raw_dir> --output <npz_dir>` (arguments are parsed at the bottom of the script) to build the processed corpus.  
  - Run `python3 split_test_balanced.py` to refresh `data_split.json` whenever the balanced dataset changes.

## Dataset Assembly Agent

- **Single-stream dataset** (`dataset.py`)  
  - `LipLivenessDataset` wraps `.npz` files, executes optional Procrustes alignment (`normalize_landmarks_procrustes`), and materializes feature stacks via `compute_features` + `features_to_array`.  
  - Works with configs from `config.py` (simple vs full feature sets) to ensure the input dimensionality matches the current model.

- **Band-split dataset** (`dataset_bandvae.py`)  
  - Extends the above by invoking `ButterworthFilterBank` to create LF/BP/HF streams, computing derivative-rich features via `compute_band_features`, and returning `(x_lf, x_bp, x_hf)` tensors already shaped as `[B, C_in, T]`.

- **Stage 2 dataset** (`dataset_stage2.py`)  
  - Consumes `data_split.json`, stitches together real/fake paths, and returns `(x_lf, x_bp, x_hf, label)` for discriminative fine-tuning.  
  - Includes deterministic crop/pad logic, Butterworth filtering, and optional feature toggles (acceleration, angle, angle-rate).

- **Configuration helpers**  
  - `config.py` (`FeatureSpaceVAE`) and `config_bandvae.py` expose `FullFeatureConfig`/`SimpleFeatureConfig` classes (paths, T-fixed, dilations, KL weights) so every training script pulls identical hyperparameters.

## Model Core Agent

- **Single-stream FeatureSpaceVAE** (`model.py`)  
  - Provides `TCNEncoder`, `TCNDecoder`, `FeatureSpaceVAE`, and helper losses (`reconstruction_loss`, `vae_loss`).  
  - Used by the original proof-of-concept one-class model and Stage 1 legacy scripts.

- **BandSplitVAE stack** (`model_bandvae.py`)  
  - `ButterworthFilterBank` implements low/band/high filters with SciPy's SOS form.  
  - `SingleBandVAE` (TCN encoder/decoder + reparameterization) is instantiated three times.  
  - `BandSplitVAE` handles fusion heads, per-band KL weights, and exports `band_split_vae_loss` which Stage 1/2 scripts rely on.

- **Lightweight seaborn stub** (`seaborn.py`)  
  - Supplies a minimal `heatmap` wrapper for deployments that lack the real seaborn dependency; imported by `final.py`.

## Stage 1 Training Agent

- **One-class POC**  
  - `train.py` trains `FeatureSpaceVAE` on processed live data only (see `config.py`).  
  - `auto_train_simple.sh` waits for a FULL run to finish before starting the SIMPLE mode automatically.

- **Band-split training**  
  - `train_bandvae.py` loads `LipLivenessBandDataset`, trains `BandSplitVAE`, and writes checkpoints under `runs/bandvae_full` or `runs/bandvae_simple`.  
  - `train_both_bandvae.sh` chains the full → simple runs for convenience.

- **Stage 1 extension**  
  - `train_stage1_pretrain.py` reproduces the Stage 1 experiment directly on `/home/elicer/.../20GBprocessed`, using T=300, random cropping, AMP, and storing results in `runs/stage1_pretrain`.

- **Outputs**  
  - All training scripts save `best.pt` (plus logs) into `runs/`. Downstream agents read these paths explicitly, so keep the directory layout intact.

## Stage 2 Learner Agents

- **Margin-based learners**  
  - `train_stage2_method1_margin.py` is the canonical implementation: it loads `runs/stage1_pretrain/stage1_pretrained.pt`, fine-tunes with `margin_loss`, and now includes the improved batch-wise tensor handling that used to live in the `_fixed` variant—so no alternate copies are required.

- **Latent discriminator learner**  
  - `train_stage2_method2_discriminator.py` adds `LatentDiscriminator` (max/avg pooled latent concatenation → MLP) and mixes BCE with reconstruction loss via `LAMBDA_CLS`.  
  - Shares the same dataset + config but emits an extra discriminator checkpoint and accuracy metrics.

- **Common patterns**  
  - Every Stage 2 script instantiates `Stage2Dataset` with `split_name='stage2_train'` for training and `'final_test'` for validation, so `data_split.json` is the single source of truth.  
  - Mixed precision (`torch.cuda.amp`) and per-band logging mirror Stage 1 for reproducibility.

## Evaluation & Filtering Agent

- **Method comparison**  
  - `evaluate_stage2_methods.py` loads both Stage 2 checkpoints, runs inference on the final-test split, and emits `runs/stage2_comparison/{comparison_metrics.json, tsne_comparison.png, score_distributions.png}`.
  - `evaluate_and_visualize_stage2.py` expands on this with histograms, ROC curves, and t-SNE plots saved into `runs/stage2_evaluation*`; pass `--audiodriven-only` when you want the filtered audio-driven analysis that previously lived in a separate file.

- **Gaussian filtering & PCA**  
  - `evaluate_gaussian_filters.py`, `tune_gaussian_filters.py`, `tune_gaussian_advanced.py`, and `final.py` implement the two-stage rejection scheme: Stage 1 uses 3D reconstruction-loss Gaussians, Stage 2 uses PCA-reduced latent Gaussians. Outputs go under `runs/gaussian_filtering*` and `runs/final_evaluation_pca10d/`.

- **Classifier & misc evaluation**  
  - `evaluate_classifier.py`, `evaluate_proper.py`, and `evaluate_with_full_train.py` focus on alternative checkpoints/datasets.  
  - Visualization helpers (`visualize_2d_lf_hf.py`, `visualize_3d_all_bands.py`, `visualize_test_balanced.py`, `visualize_trial.py`) produce the `.png` assets checked into the repo (`lf_vs_hf_2d_scatter.png`, etc.).

- **Artifacts**  
  - `band_split_vae.dot` + `band_split_vae_tikz.tex` describe the architecture graphically; `bd.png`, `classifier_*.png`, `real_vs_fake_bandloss.png` capture published figures.

## Hyperparameter & Diagnostics Agent

- **Tuning scripts**  
  - `tune_hp_T300.py`, `tune_hyperparameters.py`, and `tune_gaussian_advanced.py` sweep learning rates, dilation patterns, filter confidences, and PCA dimensions. All scripts reuse the dataset/model factories, so configuration changes stay centralized.

- **Analysis utilities**  
  - `analyze_fake_bandloss.py` replays videos through `VideoLandmarkExtractor` and `BandSplitVAE` to study per-band reconstruction gaps between real and fake samples.  
  - `debug_nan.py` isolates NaN-producing sequences.  
  - `check_progress.sh` (also listed above) and other shell stubs help with long runs.

- **Legacy prototype**  
  - `req.py` is a self-contained proof-of-concept one-class TCN-VAE (monolithic script with its own dataset class, config, and training loop). Useful for rapid experiments without touching the main modules.

## Documentation & Requirements Agent

- **Guides**  
  - `README.md` gives the Korean/English overview, `PROJECT.md` documents the implementation scope, `STAGE2_PIPELINE.md` covers the negative sampling plan, and `HYPERPARAMETER_TUNING.md` summarizes experiments.

- **Requirements**  
  - `pip_requirements.txt` and `conda_requirements.txt` list the Python dependencies (PyTorch ≥ 2.0, NumPy ≥ 1.24, Mediapipe, SciPy, scikit-learn, matplotlib, seaborn/tqdm, etc.).  
  - Use `conda create --name bandvae --file conda_requirements.txt` or `pip install -r pip_requirements.txt` to mirror the training environment.

- **Outputs**  
  - Generated results live under `runs/` (kept intentionally) and are referenced by evaluation scripts. Do not rename or move these directories unless you also update the constants near the top of each script.

---

With this breakdown you can quickly locate the right entry point for any task—collecting data, training, tuning, filtering, or generating reports—without searching the entire repository.
