"""
Real vs Fake 분류기 평가 (Step 5 & 6 포함)
- Step 5: Learnable Weighted Fusion 사용
- Step 6: Anomaly Scoring으로 분류
- Train/Test split 적용
- 성능 메트릭 계산 (Accuracy, Precision, Recall, F1, ROC-AUC)
"""

import os
import sys
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

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


def extract_features_from_video(video_path, extractor, filter_bank):
    """Extract and prepare features from video"""
    try:
        for segment_data in extractor.extract_from_video(video_path):
            landmarks = segment_data['landmarks']

            if landmarks.shape[0] < 30:
                return None

            # Select lips landmarks (40 points)
            lips_start = 16 + 16
            lips_landmarks = landmarks[:, lips_start:lips_start+40, :]

            # Normalize
            normalized = normalize_landmarks_procrustes(lips_landmarks, ref_frames=10)

            # Compute features
            features = compute_features(normalized, fps=30)
            features_array = features_to_array(features)

            # Apply frequency decomposition
            position = normalized
            pos_lf, pos_bp, pos_hf = filter_bank.apply(position)

            # Reconstruct per-band features
            T, K, F = features_array.shape

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
        print(f"Error: {e}")
        return None

    return None


@torch.no_grad()
def compute_anomaly_score(model, features_lf, features_bp, features_hf,
                          betas={'lf': 1.0, 'bp': 1.0, 'hf': 1.0}):
    """
    Step 6: Anomaly Scoring

    S_ano = Σ_b [α·‖F_b - F̂_b‖₁ + β_b·KL_b]

    Returns:
        score: anomaly score (higher = more anomalous)
    """
    # Normalize per band
    features_lf = (features_lf - features_lf.mean()) / (features_lf.std() + 1e-8)
    features_bp = (features_bp - features_bp.mean()) / (features_bp.std() + 1e-8)
    features_hf = (features_hf - features_hf.mean()) / (features_hf.std() + 1e-8)

    # Convert to tensors
    x_lf = torch.from_numpy(features_lf.T).unsqueeze(0).float()
    x_bp = torch.from_numpy(features_bp.T).unsqueeze(0).float()
    x_hf = torch.from_numpy(features_hf.T).unsqueeze(0).float()

    # Forward pass (Step 5: Learnable Weighted Fusion 포함)
    recons, mus, logvars, fused = model(x_lf, x_bp, x_hf)

    score = 0.0

    # Per-band scoring
    for band, x_in in zip(['lf', 'bp', 'hf'], [x_lf, x_bp, x_hf]):
        # Reconstruction error (L1)
        recon_error = torch.abs(recons[band] - x_in).mean()

        # KL divergence
        kl_div = -0.5 * (1 + logvars[band] - mus[band].pow(2) - logvars[band].exp()).mean()

        # Weighted sum
        score += recon_error + betas[band] * kl_div

    return score.item()


def evaluate_classifier(real_dir, fake_dir, checkpoint_path, num_samples=100,
                       test_ratio=0.5, seed=42):
    """
    Real vs Fake 분류기 평가

    Args:
        real_dir: Real video directory
        fake_dir: Fake video directory
        checkpoint_path: Model checkpoint
        num_samples: Total samples per class
        test_ratio: Test set ratio (0.5 = 50% test)
        seed: Random seed

    Returns:
        results: dict with scores and metrics
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find videos
    real_videos = sorted(glob.glob(os.path.join(real_dir, "**/*.mp4"), recursive=True))
    fake_videos = sorted(glob.glob(os.path.join(fake_dir, "**/*.mp4"), recursive=True))

    print(f"Total real videos: {len(real_videos)}")
    print(f"Total fake videos: {len(fake_videos)}")

    # Sample
    real_sample = random.sample(real_videos, min(num_samples, len(real_videos)))
    fake_sample = random.sample(fake_videos, min(num_samples, len(fake_videos)))

    # Split train/test
    num_test = int(num_samples * test_ratio)
    num_train = num_samples - num_test

    real_train = real_sample[:num_train]
    real_test = real_sample[num_train:]
    fake_train = fake_sample[:num_train]  # Not used in one-class learning
    fake_test = fake_sample[num_train:]

    print(f"\nTrain set: {num_train} real (fake not used in one-class)")
    print(f"Test set: {num_test} real + {num_test} fake")

    # Initialize
    extractor = VideoLandmarkExtractor(segment_duration=15, fps=30)
    filter_bank = ButterworthFilterBank(fps=30, fc_low=2.0, fc_high=8.0, order=4)

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    config = FullFeatureConfig()
    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n✓ Model loaded!")
    print(f"  Fusion weights: {torch.softmax(model.fusion_weights, dim=0).detach().numpy()}")

    # Process TRAIN real videos (for threshold calibration)
    print(f"\n{'='*80}")
    print("Processing TRAIN Real videos (for threshold calibration)...")
    print(f"{'='*80}")

    train_scores = []
    for video_path in tqdm(real_train, desc="Train real"):
        result = extract_features_from_video(video_path, extractor, filter_bank)
        if result is None:
            continue

        features_lf, features_bp, features_hf = result

        if np.isnan(features_lf).any() or np.isnan(features_bp).any() or np.isnan(features_hf).any():
            continue

        try:
            score = compute_anomaly_score(model, features_lf, features_bp, features_hf)
            if not np.isnan(score):
                train_scores.append(score)
        except Exception as e:
            print(f"Error: {e}")
            continue

    train_scores = np.array(train_scores)
    print(f"\nTrain real scores: n={len(train_scores)}, mean={train_scores.mean():.4f}, std={train_scores.std():.4f}")

    # Compute threshold (논문: 20th percentile, but let's try mean)
    threshold_20th = np.percentile(train_scores, 20)
    threshold_mean = train_scores.mean()
    threshold_median = np.median(train_scores)

    print(f"\nThreshold candidates:")
    print(f"  20th percentile: {threshold_20th:.4f}")
    print(f"  Median: {threshold_median:.4f}")
    print(f"  Mean: {threshold_mean:.4f}")

    # Process TEST real videos
    print(f"\n{'='*80}")
    print("Processing TEST Real videos...")
    print(f"{'='*80}")

    test_real_scores = []
    for video_path in tqdm(real_test, desc="Test real"):
        result = extract_features_from_video(video_path, extractor, filter_bank)
        if result is None:
            continue

        features_lf, features_bp, features_hf = result

        if np.isnan(features_lf).any() or np.isnan(features_bp).any() or np.isnan(features_hf).any():
            continue

        try:
            score = compute_anomaly_score(model, features_lf, features_bp, features_hf)
            if not np.isnan(score):
                test_real_scores.append(score)
        except Exception as e:
            continue

    # Process TEST fake videos
    print(f"\n{'='*80}")
    print("Processing TEST Fake videos...")
    print(f"{'='*80}")

    test_fake_scores = []
    for video_path in tqdm(fake_test, desc="Test fake"):
        result = extract_features_from_video(video_path, extractor, filter_bank)
        if result is None:
            continue

        features_lf, features_bp, features_hf = result

        if np.isnan(features_lf).any() or np.isnan(features_bp).any() or np.isnan(features_hf).any():
            continue

        try:
            score = compute_anomaly_score(model, features_lf, features_bp, features_hf)
            if not np.isnan(score):
                test_fake_scores.append(score)
        except Exception as e:
            continue

    test_real_scores = np.array(test_real_scores)
    test_fake_scores = np.array(test_fake_scores)

    print(f"\nTest real scores: n={len(test_real_scores)}, mean={test_real_scores.mean():.4f}, std={test_real_scores.std():.4f}")
    print(f"Test fake scores: n={len(test_fake_scores)}, mean={test_fake_scores.mean():.4f}, std={test_fake_scores.std():.4f}")

    # Combine labels and scores
    y_true = np.concatenate([
        np.ones(len(test_real_scores)),   # Real = 1
        np.zeros(len(test_fake_scores))   # Fake = 0
    ])

    y_scores = np.concatenate([test_real_scores, test_fake_scores])

    # Evaluate different thresholds
    print(f"\n{'='*80}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*80}")

    results = {}

    for threshold_name, threshold in [
        ("20th percentile", threshold_20th),
        ("Median", threshold_median),
        ("Mean", threshold_mean)
    ]:
        # 논문 전략: 낮은 score = fake (하지만 우리 결과는 fake가 더 높음)
        # 전략 1: score > threshold → real
        y_pred_high = (y_scores > threshold).astype(int)

        acc_high = accuracy_score(y_true, y_pred_high)
        prec_high = precision_score(y_true, y_pred_high, zero_division=0)
        rec_high = recall_score(y_true, y_pred_high, zero_division=0)
        f1_high = f1_score(y_true, y_pred_high, zero_division=0)

        # 전략 2: score < threshold → real (논문 방식)
        y_pred_low = (y_scores < threshold).astype(int)

        acc_low = accuracy_score(y_true, y_pred_low)
        prec_low = precision_score(y_true, y_pred_low, zero_division=0)
        rec_low = recall_score(y_true, y_pred_low, zero_division=0)
        f1_low = f1_score(y_true, y_pred_low, zero_division=0)

        print(f"\nThreshold: {threshold_name} = {threshold:.4f}")
        print(f"  Strategy 1 (score > threshold → real):")
        print(f"    Accuracy:  {acc_high:.4f}")
        print(f"    Precision: {prec_high:.4f}")
        print(f"    Recall:    {rec_high:.4f}")
        print(f"    F1:        {f1_high:.4f}")
        print(f"  Strategy 2 (score < threshold → real, like paper):")
        print(f"    Accuracy:  {acc_low:.4f}")
        print(f"    Precision: {prec_low:.4f}")
        print(f"    Recall:    {rec_low:.4f}")
        print(f"    F1:        {f1_low:.4f}")

        results[threshold_name] = {
            'threshold': threshold,
            'strategy_high': {'acc': acc_high, 'prec': prec_high, 'rec': rec_high, 'f1': f1_high},
            'strategy_low': {'acc': acc_low, 'prec': prec_low, 'rec': rec_low, 'f1': f1_low}
        }

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f"\n{'='*80}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"{'='*80}")

        results['roc_auc'] = roc_auc
    except:
        print("\nCannot compute ROC-AUC (need more samples)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Score distribution
    ax = axes[0]
    ax.hist(test_real_scores, bins=30, alpha=0.6, color='green', label='Real (Test)', edgecolor='black')
    ax.hist(test_fake_scores, bins=30, alpha=0.6, color='red', label='Fake (Test)', edgecolor='black')
    ax.axvline(threshold_mean, color='blue', linestyle='--', linewidth=2, label=f'Threshold (Mean={threshold_mean:.3f})')
    ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Test Set: Anomaly Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ROC curve
    if 'roc_auc' in results:
        ax = axes[1]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('classifier_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Evaluation plot saved to: classifier_evaluation.png")

    return {
        'train_scores': train_scores,
        'test_real_scores': test_real_scores,
        'test_fake_scores': test_fake_scores,
        'results': results,
        'y_true': y_true,
        'y_scores': y_scores
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str,
                       default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_원본")
    parser.add_argument("--fake_dir", type=str,
                       default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_변조")
    parser.add_argument("--checkpoint", type=str,
                       default="/home/elicer/liveness_detection/model1/runs/bandvae_full/best.pt")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Total samples per class")
    parser.add_argument("--test_ratio", type=float, default=0.5,
                       help="Test set ratio (0.5 = 50%% test)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("REAL vs FAKE CLASSIFIER EVALUATION (Step 5 & 6)")
    print("="*80)
    print(f"Real directory: {args.real_dir}")
    print(f"Fake directory: {args.fake_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Test ratio: {args.test_ratio} ({int(args.num_samples * args.test_ratio)} test samples per class)")
    print(f"Random seed: {args.seed}")
    print("="*80 + "\n")

    results = evaluate_classifier(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print("\n✓ Evaluation complete!")
