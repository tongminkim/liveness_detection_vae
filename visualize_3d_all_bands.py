"""
3D Visualization: LF vs BP vs HF Reconstruction Loss
Real vs Fake 영상의 3개 주파수 대역 reconstruction loss 분포 비교

- Real: processed_live_test에서 200개 샘플
- Fake: train_변조 비디오에서 200개 샘플 (전처리 포함)
- Model: bandvae_full/best.pt
- Plot: 3D scatter (LF, BP, HF)
"""

import os
import sys
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm

# Import from existing modules
from preprocess_data import (
    VideoLandmarkExtractor,
    normalize_landmarks_procrustes,
    compute_features,
    features_to_array,
)
from preprocess_data_fast import process_single_video
from model_bandvae import BandSplitVAE, ButterworthFilterBank
from config_bandvae import FullFeatureConfig


def load_npz_and_extract_features(npz_path, filter_bank):
    """
    .npz 파일에서 landmarks를 로드하고 3개 대역 features 추출

    Returns:
        (features_lf, features_bp, features_hf): 3 x [T, K*F_dim] numpy arrays
        None if extraction failed
    """
    try:
        data = np.load(npz_path)

        # Load lips landmarks (40 points: 20 outer + 20 inner)
        lips_outer = data['lips_outer']  # [T, 21, 2]
        lips_inner = data['lips_inner']  # [T, 21, 2]

        # Remove last point (duplicate)
        lips = np.concatenate([
            lips_outer[:, :-1, :],  # [T, 20, 2]
            lips_inner[:, :-1, :]   # [T, 20, 2]
        ], axis=1)  # [T, 40, 2]

        T, K, _ = lips.shape

        # Need at least 30 frames
        if T < 30:
            return None

        # Normalize with Procrustes
        normalized = normalize_landmarks_procrustes(lips, ref_frames=10)

        # Compute features (FULL version)
        features = compute_features(normalized, fps=30)
        features_array = features_to_array(features)  # [T, K, 8]

        # Extract position for filtering [T, K, 2]
        position = normalized  # [T, K, 2]

        # Apply frequency decomposition - ALL 3 BANDS
        pos_lf, pos_bp, pos_hf = filter_bank.apply(position)

        # Reconstruct features for ALL 3 bands
        features_lf = features_array.copy()
        features_lf[:, :, :2] = pos_lf

        features_bp = features_array.copy()
        features_bp[:, :, :2] = pos_bp

        features_hf = features_array.copy()
        features_hf[:, :, :2] = pos_hf

        # Flatten to [T, K*F]
        T, K, F = features_array.shape
        features_lf_flat = features_lf.reshape(T, K * F)
        features_bp_flat = features_bp.reshape(T, K * F)
        features_hf_flat = features_hf.reshape(T, K * F)

        return features_lf_flat, features_bp_flat, features_hf_flat

    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None


def preprocess_video_to_npz(video_path, temp_dir):
    """
    비디오를 전처리하여 .npz 파일 생성

    Returns:
        npz_path: 생성된 .npz 파일 경로
        None if failed
    """
    try:
        # Create temp directory
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        # Process video
        args = (video_path, temp_dir, 15, 2)  # segment_duration=15, frame_skip=2
        result = process_single_video(args)

        if result.startswith("✗"):
            return None

        # Find generated .npz file
        video_name = Path(video_path).stem
        npz_files = sorted(glob.glob(os.path.join(temp_dir, f"{video_name}_seg*.npz")))

        if len(npz_files) == 0:
            return None

        # Return first segment
        return npz_files[0]

    except Exception as e:
        print(f"Error preprocessing {video_path}: {e}")
        return None


def load_model(checkpoint_path, config):
    """Load trained BandSplitVAE model"""
    C_in = config.C_in_per_band  # FULL: 320

    model = BandSplitVAE(
        C_in_per_band=C_in,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


@torch.no_grad()
def compute_all_band_losses(model, features_lf, features_bp, features_hf):
    """
    Compute ALL 3 band reconstruction losses

    Returns:
        (loss_lf, loss_bp, loss_hf): tuple of 3 scalars
    """
    # Per-band normalization
    features_lf = (features_lf - features_lf.mean()) / (features_lf.std() + 1e-8)
    features_bp = (features_bp - features_bp.mean()) / (features_bp.std() + 1e-8)
    features_hf = (features_hf - features_hf.mean()) / (features_hf.std() + 1e-8)

    # Convert to torch tensors [1, C_in, T]
    x_lf = torch.from_numpy(features_lf.T).unsqueeze(0).float()
    x_bp = torch.from_numpy(features_bp.T).unsqueeze(0).float()
    x_hf = torch.from_numpy(features_hf.T).unsqueeze(0).float()

    # Forward pass
    recons, mus, logvars, fused = model(x_lf, x_bp, x_hf)

    # Compute L1 reconstruction losses for ALL 3 bands
    loss_lf = torch.abs(recons['lf'] - x_lf).mean().item()
    loss_bp = torch.abs(recons['bp'] - x_bp).mean().item()
    loss_hf = torch.abs(recons['hf'] - x_hf).mean().item()

    return loss_lf, loss_bp, loss_hf


def process_real_samples(npz_dir, num_samples, model, filter_bank):
    """
    Process Real samples from .npz files

    Returns:
        losses: list of (loss_lf, loss_bp, loss_hf) tuples
    """
    # Find all .npz files
    all_npz = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

    if len(all_npz) == 0:
        raise ValueError(f"No .npz files found in {npz_dir}")

    print(f"Found {len(all_npz)} Real .npz files")

    # Randomly sample
    random.seed(42)
    sampled_npz = random.sample(all_npz, min(num_samples, len(all_npz)))

    print(f"Processing {len(sampled_npz)} Real samples...")

    losses = []
    successful = 0
    failed = 0

    for npz_path in tqdm(sampled_npz, desc="Processing Real"):
        result = load_npz_and_extract_features(npz_path, filter_bank)

        if result is None:
            failed += 1
            continue

        features_lf, features_bp, features_hf = result

        # Check for NaN
        if np.isnan(features_lf).any() or np.isnan(features_bp).any() or np.isnan(features_hf).any():
            failed += 1
            continue

        try:
            loss_lf, loss_bp, loss_hf = compute_all_band_losses(model, features_lf, features_bp, features_hf)

            # Check if losses are valid
            if np.isnan(loss_lf) or np.isnan(loss_bp) or np.isnan(loss_hf):
                failed += 1
                continue

            losses.append((loss_lf, loss_bp, loss_hf))
            successful += 1

            if successful >= num_samples:
                break

        except Exception as e:
            print(f"Error computing losses: {e}")
            failed += 1
            continue

    print(f"Real samples - Success: {successful}, Failed: {failed}")

    return losses


def process_fake_samples(video_dir, num_samples, model, filter_bank, temp_dir="temp_fake_npz_3d"):
    """
    Process Fake samples from videos (with preprocessing)

    Returns:
        losses: list of (loss_lf, loss_bp, loss_hf) tuples
    """
    # Find all videos
    all_videos = sorted(glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True))

    if len(all_videos) == 0:
        raise ValueError(f"No videos found in {video_dir}")

    print(f"Found {len(all_videos)} Fake videos")

    # Randomly sample (same seed as 2D version for consistency)
    random.seed(42)
    sampled_videos = random.sample(all_videos, min(num_samples * 2, len(all_videos)))

    print(f"Processing Fake samples (may take a while due to preprocessing)...")

    losses = []
    successful = 0
    failed = 0

    for video_path in tqdm(sampled_videos, desc="Processing Fake"):
        # Preprocess video to .npz
        npz_path = preprocess_video_to_npz(video_path, temp_dir)

        if npz_path is None:
            failed += 1
            continue

        # Extract features
        result = load_npz_and_extract_features(npz_path, filter_bank)

        if result is None:
            failed += 1
            continue

        features_lf, features_bp, features_hf = result

        # Check for NaN
        if np.isnan(features_lf).any() or np.isnan(features_bp).any() or np.isnan(features_hf).any():
            failed += 1
            continue

        try:
            loss_lf, loss_bp, loss_hf = compute_all_band_losses(model, features_lf, features_bp, features_hf)

            # Check if losses are valid
            if np.isnan(loss_lf) or np.isnan(loss_bp) or np.isnan(loss_hf):
                failed += 1
                continue

            losses.append((loss_lf, loss_bp, loss_hf))
            successful += 1

            if successful >= num_samples:
                break

        except Exception as e:
            print(f"Error computing losses: {e}")
            failed += 1
            continue

    print(f"Fake samples - Success: {successful}, Failed: {failed}")

    # Clean up temp directory
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

    return losses


def plot_3d_scatter(losses_real, losses_fake, output_path="lf_bp_hf_3d_scatter.png"):
    """
    Plot 3D scatter: LF (x) vs BP (y) vs HF (z)

    Args:
        losses_real: list of (loss_lf, loss_bp, loss_hf) tuples for Real
        losses_fake: list of (loss_lf, loss_bp, loss_hf) tuples for Fake
    """
    # Convert to arrays
    real_lf = np.array([x[0] for x in losses_real])
    real_bp = np.array([x[1] for x in losses_real])
    real_hf = np.array([x[2] for x in losses_real])

    fake_lf = np.array([x[0] for x in losses_fake])
    fake_bp = np.array([x[1] for x in losses_fake])
    fake_hf = np.array([x[2] for x in losses_fake])

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot scatter
    ax.scatter(real_lf, real_bp, real_hf, alpha=0.6, s=80, c='#2ecc71',
               edgecolors='black', linewidth=0.5, label=f'Real (n={len(real_lf)})',
               marker='o')
    ax.scatter(fake_lf, fake_bp, fake_hf, alpha=0.6, s=80, c='#e74c3c',
               edgecolors='black', linewidth=0.5, label=f'Fake (n={len(fake_lf)})',
               marker='^')

    # Compute statistics
    real_lf_mean, real_bp_mean, real_hf_mean = np.mean(real_lf), np.mean(real_bp), np.mean(real_hf)
    fake_lf_mean, fake_bp_mean, fake_hf_mean = np.mean(fake_lf), np.mean(fake_bp), np.mean(fake_hf)

    # Add mean markers (larger)
    ax.scatter([real_lf_mean], [real_bp_mean], [real_hf_mean], marker='*', s=800,
               c='green', edgecolors='black', linewidth=2.5, label='Real Mean', zorder=10)
    ax.scatter([fake_lf_mean], [fake_bp_mean], [fake_hf_mean], marker='*', s=800,
               c='red', edgecolors='black', linewidth=2.5, label='Fake Mean', zorder=10)

    # Labels
    ax.set_xlabel('Low-Frequency (LF) Loss', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Band-Pass (BP) Loss', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('High-Frequency (HF) Loss', fontsize=14, fontweight='bold', labelpad=10)

    ax.set_title('Real vs Fake: 3D Distribution (LF vs BP vs HF Reconstruction Loss)\n'
                 f'200 samples each | Model: bandvae_full/best.pt',
                 fontsize=16, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # View angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 3D scatter plot saved to: {output_path}")

    # Print statistics
    print("\n" + "="*100)
    print("REAL vs FAKE: 3-BAND (LF/BP/HF) RECONSTRUCTION LOSS COMPARISON")
    print("="*100)

    print(f"\n{'Band':<10} {'Real Mean':<15} {'Real Std':<15} {'Fake Mean':<15} {'Fake Std':<15} {'Δ Mean':<15}")
    print("-"*100)

    for band, real_data, fake_data in [
        ('LF', real_lf, fake_lf),
        ('BP', real_bp, fake_bp),
        ('HF', real_hf, fake_hf)
    ]:
        real_mean, real_std = np.mean(real_data), np.std(real_data)
        fake_mean, fake_std = np.mean(fake_data), np.std(fake_data)
        delta_mean = abs(real_mean - fake_mean)

        print(f"{band:<10} {real_mean:<15.6f} {real_std:<15.6f} {fake_mean:<15.6f} {fake_std:<15.6f} {delta_mean:<15.6f}")

    print("="*100)

    # Detailed stats table
    print("\n" + "="*100)
    print("DETAILED STATISTICS")
    print("="*100)

    for band, real_data, fake_data in [
        ('LF', real_lf, fake_lf),
        ('BP', real_bp, fake_bp),
        ('HF', real_hf, fake_hf)
    ]:
        print(f"\n{band} Band:")
        print(f"  {'Metric':<15} {'Real':<20} {'Fake':<20}")
        print(f"  {'-'*15} {'-'*20} {'-'*20}")
        print(f"  {'Mean:':<15} {np.mean(real_data):<20.6f} {np.mean(fake_data):<20.6f}")
        print(f"  {'Std:':<15} {np.std(real_data):<20.6f} {np.std(fake_data):<20.6f}")
        print(f"  {'Median:':<15} {np.median(real_data):<20.6f} {np.median(fake_data):<20.6f}")
        print(f"  {'Min:':<15} {np.min(real_data):<20.6f} {np.min(fake_data):<20.6f}")
        print(f"  {'Max:':<15} {np.max(real_data):<20.6f} {np.max(fake_data):<20.6f}")

    print("="*100 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="3D visualization: LF vs BP vs HF reconstruction loss")
    parser.add_argument(
        "--real_dir",
        type=str,
        default="/home/elicer/liveness_detection/model1/processed_live_test",
        help="Directory containing Real .npz files"
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_변조",
        help="Directory containing Fake videos"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples per class"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/elicer/liveness_detection/model1/runs/bandvae_full/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lf_bp_hf_3d_scatter.png",
        help="Output plot path"
    )

    args = parser.parse_args()

    # Header
    print("\n" + "="*100)
    print("3D VISUALIZATION: LF vs BP vs HF RECONSTRUCTION LOSS (REAL vs FAKE)")
    print("="*100)
    print(f"Real .npz directory: {args.real_dir}")
    print(f"Fake video directory: {args.fake_dir}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*100 + "\n")

    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    config = FullFeatureConfig()
    model = load_model(args.checkpoint, config)

    # Initialize filter bank
    filter_bank = ButterworthFilterBank(fps=30, fc_low=2.0, fc_high=8.0, order=4)

    # Process REAL samples
    print("\n" + "="*100)
    print("STEP 1: Processing REAL samples (from .npz files)")
    print("="*100)
    losses_real = process_real_samples(args.real_dir, args.num_samples, model, filter_bank)

    if len(losses_real) < args.num_samples:
        print(f"Warning: Only got {len(losses_real)} Real samples (requested {args.num_samples})")

    # Process FAKE samples
    print("\n" + "="*100)
    print("STEP 2: Processing FAKE samples (from videos with preprocessing)")
    print("="*100)
    losses_fake = process_fake_samples(args.fake_dir, args.num_samples, model, filter_bank)

    if len(losses_fake) < args.num_samples:
        print(f"Warning: Only got {len(losses_fake)} Fake samples (requested {args.num_samples})")

    # Plot 3D scatter
    print("\n" + "="*100)
    print("STEP 3: Generating 3D scatter plot")
    print("="*100)
    plot_3d_scatter(losses_real, losses_fake, output_path=args.output)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
