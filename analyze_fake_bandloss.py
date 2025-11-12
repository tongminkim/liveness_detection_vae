"""
Real vs Fake 영상 대역별 Reconstruction Loss 비교 분석
- 100개의 real 영상 샘플에서 랜드마크 추출 (full version)
- 100개의 fake 영상 샘플에서 랜드마크 추출 (full version)
- BandSplitVAE로 대역별 reconstruction loss 계산
- Real vs Fake 비교 히스토그램 시각화 (PoC 논문 스타일)
"""

import os
import sys
import glob
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import from existing modules
from preprocess_data import (
    VideoLandmarkExtractor,
    normalize_landmarks_procrustes,
    compute_features,
    features_to_array,
    SELECTED_IDX
)
from model_bandvae import BandSplitVAE, ButterworthFilterBank
from config_bandvae import FullFeatureConfig


def extract_landmark_features_from_video(video_path, extractor, filter_bank):
    """
    단일 비디오에서 랜드마크 추출 및 대역 분리

    Returns:
        features_lf, features_bp, features_hf: [T, K*F_dim] numpy arrays
        None if extraction failed
    """
    try:
        # Extract landmarks from first segment
        for segment_data in extractor.extract_from_video(video_path):
            landmarks = segment_data['landmarks']  # [T, K, 2]

            # Need at least 30 frames
            if landmarks.shape[0] < 30:
                return None

            # Select lips landmarks (40 points: outer + inner, matching config)
            # SELECTED_IDX = LEFT_EYE(16) + RIGHT_EYE(16) + LIPS_OUTER(21) + LIPS_INNER(21) + FACE_CENTER(1)
            # Take only first 40 lips landmarks (20 outer + 20 inner)
            lips_start = 16 + 16  # after eyes
            lips_landmarks = landmarks[:, lips_start:lips_start+40, :]  # [T, 40, 2]

            # Normalize with Procrustes
            normalized = normalize_landmarks_procrustes(lips_landmarks, ref_frames=10)

            # Compute features (FULL version)
            features = compute_features(normalized, fps=30)
            features_array = features_to_array(features)  # [T, 42, 8]

            # Flatten to [T, 42*8=336]
            T, K, F = features_array.shape
            features_flat = features_array.reshape(T, K * F)

            # Extract position only for filtering [T, K, 2]
            position = normalized  # [T, K, 2]

            # Apply frequency decomposition
            pos_lf, pos_bp, pos_hf = filter_bank.apply(position)

            # Reconstruct features for each band
            # We need to replace position component in features_array
            features_lf = features_array.copy()
            features_lf[:, :, :2] = pos_lf

            features_bp = features_array.copy()
            features_bp[:, :, :2] = pos_bp

            features_hf = features_array.copy()
            features_hf[:, :, :2] = pos_hf

            # Flatten
            features_lf_flat = features_lf.reshape(T, K * F)
            features_bp_flat = features_bp.reshape(T, K * F)
            features_hf_flat = features_hf.reshape(T, K * F)

            return features_lf_flat, features_bp_flat, features_hf_flat

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

    return None


def load_model(checkpoint_path, config):
    """Load trained BandSplitVAE model"""
    # Model parameters from config
    C_in = config.C_in_per_band  # FULL: 320 (40 landmarks × 8 features)

    model = BandSplitVAE(
        C_in_per_band=C_in,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


@torch.no_grad()
def compute_band_losses(model, features_lf, features_bp, features_hf, debug=False):
    """
    Compute per-band reconstruction losses

    Args:
        model: BandSplitVAE
        features_lf, features_bp, features_hf: [T, C_in] numpy arrays
        debug: print debug info

    Returns:
        losses: dict with 'lf', 'bp', 'hf' reconstruction losses (scalars)
    """
    # Per-band normalization (zero mean, unit variance) - CRITICAL!
    # This matches the normalization used during training
    features_lf = (features_lf - features_lf.mean()) / (features_lf.std() + 1e-8)
    features_bp = (features_bp - features_bp.mean()) / (features_bp.std() + 1e-8)
    features_hf = (features_hf - features_hf.mean()) / (features_hf.std() + 1e-8)

    # Convert to torch tensors [1, C_in, T]
    x_lf = torch.from_numpy(features_lf.T).unsqueeze(0).float()  # [1, C_in, T]
    x_bp = torch.from_numpy(features_bp.T).unsqueeze(0).float()
    x_hf = torch.from_numpy(features_hf.T).unsqueeze(0).float()

    if debug:
        print(f"  Tensor LF: shape={x_lf.shape}, min={x_lf.min():.4f}, max={x_lf.max():.4f}, has_nan={torch.isnan(x_lf).any()}")
        print(f"  Tensor BP: shape={x_bp.shape}, min={x_bp.min():.4f}, max={x_bp.max():.4f}, has_nan={torch.isnan(x_bp).any()}")
        print(f"  Tensor HF: shape={x_hf.shape}, min={x_hf.min():.4f}, max={x_hf.max():.4f}, has_nan={torch.isnan(x_hf).any()}")

    # Forward pass
    recons, mus, logvars, fused = model(x_lf, x_bp, x_hf)

    if debug:
        print(f"  Recon LF: shape={recons['lf'].shape}, min={recons['lf'].min():.4f}, max={recons['lf'].max():.4f}, has_nan={torch.isnan(recons['lf']).any()}")
        print(f"  Recon BP: shape={recons['bp'].shape}, min={recons['bp'].min():.4f}, max={recons['bp'].max():.4f}, has_nan={torch.isnan(recons['bp']).any()}")
        print(f"  Recon HF: shape={recons['hf'].shape}, min={recons['hf'].min():.4f}, max={recons['hf'].max():.4f}, has_nan={torch.isnan(recons['hf']).any()}")

    # Compute L1 reconstruction losses per band
    loss_lf = torch.abs(recons['lf'] - x_lf).mean().item()
    loss_bp = torch.abs(recons['bp'] - x_bp).mean().item()
    loss_hf = torch.abs(recons['hf'] - x_hf).mean().item()

    if debug:
        print(f"  Losses: LF={loss_lf}, BP={loss_bp}, HF={loss_hf}")

    return {
        'lf': loss_lf,
        'bp': loss_bp,
        'hf': loss_hf
    }


def process_videos(video_dir, num_samples=100, checkpoint_path=None, video_type="fake"):
    """
    Process videos and compute band-wise reconstruction losses

    Args:
        video_dir: Directory containing videos
        num_samples: Number of samples to process (default 100)
        checkpoint_path: Path to trained model checkpoint
        video_type: "real" or "fake" (for logging)

    Returns:
        results: dict with 'lf', 'bp', 'hf' loss arrays
    """
    # Find all videos
    all_videos = sorted(glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True))

    if len(all_videos) == 0:
        raise ValueError(f"No videos found in {video_dir}")

    print(f"Found {len(all_videos)} {video_type} videos")

    # Randomly sample
    random.seed(42)
    sampled_videos = random.sample(all_videos, min(num_samples, len(all_videos)))

    print(f"Processing {len(sampled_videos)} samples...")

    # Initialize extractors
    extractor = VideoLandmarkExtractor(segment_duration=15, fps=30)
    filter_bank = ButterworthFilterBank(fps=30, fc_low=2.0, fc_high=8.0, order=4)

    # Load model
    if checkpoint_path is None:
        # Try to find best checkpoint
        checkpoint_path = "/home/elicer/liveness_detection/model1/runs/bandvae_full/best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "/home/elicer/liveness_detection/model1/runs/live_vae/best.pt"

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    config = FullFeatureConfig()
    model = load_model(checkpoint_path, config)

    # Process videos
    losses_lf = []
    losses_bp = []
    losses_hf = []

    successful = 0
    failed = 0

    for video_path in tqdm(sampled_videos, desc=f"Processing {video_type} videos"):
        result = extract_landmark_features_from_video(video_path, extractor, filter_bank)

        if result is None:
            failed += 1
            continue

        features_lf, features_bp, features_hf = result

        # Debug: check shapes and stats
        if successful == 0:  # Print for first sample only
            print(f"\nDebug - First sample:")
            print(f"  features_lf: shape={features_lf.shape}, min={features_lf.min():.4f}, max={features_lf.max():.4f}, mean={features_lf.mean():.4f}, has_nan={np.isnan(features_lf).any()}")
            print(f"  features_bp: shape={features_bp.shape}, min={features_bp.min():.4f}, max={features_bp.max():.4f}, mean={features_bp.mean():.4f}, has_nan={np.isnan(features_bp).any()}")
            print(f"  features_hf: shape={features_hf.shape}, min={features_hf.min():.4f}, max={features_hf.max():.4f}, mean={features_hf.mean():.4f}, has_nan={np.isnan(features_hf).any()}")

        # Compute losses
        try:
            # Check for NaN in features
            if np.isnan(features_lf).any() or np.isnan(features_bp).any() or np.isnan(features_hf).any():
                print(f"Warning: NaN detected in features for {video_path}, skipping...")
                failed += 1
                continue

            band_losses = compute_band_losses(model, features_lf, features_bp, features_hf, debug=(successful==0))

            # Check if losses are valid
            if np.isnan(band_losses['lf']) or np.isnan(band_losses['bp']) or np.isnan(band_losses['hf']):
                print(f"Warning: NaN in computed losses for {video_path}, skipping...")
                failed += 1
                continue

            losses_lf.append(band_losses['lf'])
            losses_bp.append(band_losses['bp'])
            losses_hf.append(band_losses['hf'])

            successful += 1

            # Stop when we have enough samples
            if successful >= num_samples:
                break

        except Exception as e:
            print(f"Error computing losses for {video_path}: {e}")
            failed += 1
            continue

    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    return {
        'lf': np.array(losses_lf),
        'bp': np.array(losses_bp),
        'hf': np.array(losses_hf)
    }


def plot_comparative_histograms(results_real, results_fake, output_path="real_vs_fake_bandloss.png"):
    """
    Plot comparative reconstruction loss histograms (Real vs Fake)

    논문 PoC 스타일:
    - 3개의 subplot (LF, BP, HF)
    - Real과 Fake 오버레이 히스토그램
    - 통계 정보 표시
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bands = ['lf', 'bp', 'hf']
    titles = ['Low-Frequency (LF)', 'Band-Pass (BP)', 'High-Frequency (HF)']

    for idx, (band, title) in enumerate(zip(bands, titles)):
        ax = axes[idx]

        losses_real = results_real[band]
        losses_fake = results_fake[band]

        # Plot overlaid histograms
        bins = np.linspace(
            min(losses_real.min(), losses_fake.min()),
            max(losses_real.max(), losses_fake.max()),
            30
        )

        ax.hist(losses_real, bins=bins, alpha=0.6, color='#2ecc71',
                edgecolor='black', label='Real', density=False)
        ax.hist(losses_fake, bins=bins, alpha=0.6, color='#e74c3c',
                edgecolor='black', label='Fake', density=False)

        # Compute statistics
        mean_real = np.mean(losses_real)
        mean_fake = np.mean(losses_fake)
        std_real = np.std(losses_real)
        std_fake = np.std(losses_fake)

        # Add vertical lines for means
        ax.axvline(mean_real, color='green', linestyle='--', linewidth=2.5,
                   label=f'Real μ: {mean_real:.4f}')
        ax.axvline(mean_fake, color='red', linestyle='--', linewidth=2.5,
                   label=f'Fake μ: {mean_fake:.4f}')

        # Add text box with statistics
        stats_text = (f'Real: μ={mean_real:.4f}, σ={std_real:.4f}\n'
                     f'Fake: μ={mean_fake:.4f}, σ={std_fake:.4f}\n'
                     f'Δμ = {abs(mean_real - mean_fake):.4f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Reconstruction Loss (L1)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Real vs Fake: Reconstruction Loss by Frequency Band\n(100 samples each, Full Feature Version)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparative histogram saved to: {output_path}")

    # Print comparative statistics
    print("\n" + "="*80)
    print("REAL vs FAKE RECONSTRUCTION LOSS COMPARISON (by Frequency Band)")
    print("="*80)

    for band, title in zip(bands, titles):
        losses_real = results_real[band]
        losses_fake = results_fake[band]

        print(f"\n{title}:")
        print(f"  {'Metric':<15} {'Real':>12} {'Fake':>12} {'Difference':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'Mean:':<15} {np.mean(losses_real):>12.6f} {np.mean(losses_fake):>12.6f} {abs(np.mean(losses_real) - np.mean(losses_fake)):>12.6f}")
        print(f"  {'Std:':<15} {np.std(losses_real):>12.6f} {np.std(losses_fake):>12.6f} {abs(np.std(losses_real) - np.std(losses_fake)):>12.6f}")
        print(f"  {'Median:':<15} {np.median(losses_real):>12.6f} {np.median(losses_fake):>12.6f} {abs(np.median(losses_real) - np.median(losses_fake)):>12.6f}")
        print(f"  {'Min:':<15} {np.min(losses_real):>12.6f} {np.min(losses_fake):>12.6f} {abs(np.min(losses_real) - np.min(losses_fake)):>12.6f}")
        print(f"  {'Max:':<15} {np.max(losses_real):>12.6f} {np.max(losses_fake):>12.6f} {abs(np.max(losses_real) - np.max(losses_fake)):>12.6f}")

        # Statistical test hint
        if np.mean(losses_real) > np.mean(losses_fake):
            print(f"  → Real shows HIGHER reconstruction loss (expected for one-class VAE)")
        else:
            print(f"  → Fake shows HIGHER reconstruction loss (unexpected)")

    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare Real vs Fake videos with band-split VAE")
    parser.add_argument(
        "--real_dir",
        type=str,
        default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_원본",
        help="Directory containing real videos"
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_변조",
        help="Directory containing fake videos"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to process per class"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="real_vs_fake_bandloss.png",
        help="Output histogram path"
    )

    args = parser.parse_args()

    # Header
    print("\n" + "="*80)
    print("REAL vs FAKE VIDEO COMPARISON: BAND-WISE RECONSTRUCTION LOSS ANALYSIS")
    print("="*80)
    print(f"Real video directory: {args.real_dir}")
    print(f"Fake video directory: {args.fake_dir}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Checkpoint: {args.checkpoint or 'auto-detect'}")
    print("="*80 + "\n")

    # Process REAL videos
    print("\n" + "="*80)
    print("STEP 1: Processing REAL videos")
    print("="*80)
    results_real = process_videos(
        video_dir=args.real_dir,
        num_samples=args.num_samples,
        checkpoint_path=args.checkpoint,
        video_type="real"
    )

    # Process FAKE videos
    print("\n" + "="*80)
    print("STEP 2: Processing FAKE videos")
    print("="*80)
    results_fake = process_videos(
        video_dir=args.fake_dir,
        num_samples=args.num_samples,
        checkpoint_path=args.checkpoint,
        video_type="fake"
    )

    # Plot comparative histograms
    print("\n" + "="*80)
    print("STEP 3: Generating comparative visualization")
    print("="*80)
    plot_comparative_histograms(results_real, results_fake, output_path=args.output)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
