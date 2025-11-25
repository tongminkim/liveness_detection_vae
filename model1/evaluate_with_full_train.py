"""
올바른 Anomaly Score 기반 분류
1. 전체 학습 데이터(processed_live)로 threshold 계산
2. 새로운 test real + fake로 평가
"""

import os
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

from model_bandvae import BandSplitVAE, ButterworthFilterBank
from config_bandvae import FullFeatureConfig
from dataset_bandvae import compute_band_features


@torch.no_grad()
def compute_anomaly_score_from_npz(model, npz_path, filter_bank, config):
    """
    NPZ 파일로부터 anomaly score 계산
    """
    try:
        data = np.load(npz_path)

        # Load lips landmarks
        lips_outer = data['lips_outer']  # [T, 21, 2]
        lips_inner = data['lips_inner']  # [T, 20, 2]

        # Combine (take first 20 from each)
        lips_landmarks = np.concatenate([
            lips_outer[:, :20, :],
            lips_inner[:, :20, :]
        ], axis=1)  # [T, 40, 2]

        T = lips_landmarks.shape[0]
        if T < 30:
            return None

        # Apply frequency decomposition on position
        position = lips_landmarks  # [T, K, 2]
        pos_lf, pos_bp, pos_hf = filter_bank.apply(position)

        # Compute per-band features
        features_lf = compute_band_features(
            pos_lf, fps=30,
            use_acceleration=config.use_acceleration,
            use_angle=config.use_angle,
            use_angle_rate=config.use_angle_rate
        )

        features_bp = compute_band_features(
            pos_bp, fps=30,
            use_acceleration=config.use_acceleration,
            use_angle=config.use_angle,
            use_angle_rate=config.use_angle_rate
        )

        features_hf = compute_band_features(
            pos_hf, fps=30,
            use_acceleration=config.use_acceleration,
            use_angle=config.use_angle,
            use_angle_rate=config.use_angle_rate
        )

        # Stack features
        def stack_features(feat_dict):
            parts = [feat_dict['position']]  # [T, K, 2]
            if config.use_velocity:
                parts.append(feat_dict['velocity'])
            if config.use_acceleration:
                parts.append(feat_dict['acceleration'])
            if config.use_angle:
                parts.append(feat_dict['angle'])
            if config.use_angle_rate:
                parts.append(feat_dict['angle_rate'])
            return np.concatenate(parts, axis=2)  # [T, K, F]

        x_lf = stack_features(features_lf)
        x_bp = stack_features(features_bp)
        x_hf = stack_features(features_hf)

        # Flatten to [T, K*F]
        T, K, F = x_lf.shape
        x_lf_flat = x_lf.reshape(T, K * F)
        x_bp_flat = x_bp.reshape(T, K * F)
        x_hf_flat = x_hf.reshape(T, K * F)

        # Normalize per band (CRITICAL!)
        x_lf_flat = (x_lf_flat - x_lf_flat.mean()) / (x_lf_flat.std() + 1e-8)
        x_bp_flat = (x_bp_flat - x_bp_flat.mean()) / (x_bp_flat.std() + 1e-8)
        x_hf_flat = (x_hf_flat - x_hf_flat.mean()) / (x_hf_flat.std() + 1e-8)

        # Convert to tensors
        x_lf_t = torch.from_numpy(x_lf_flat.T).unsqueeze(0).float()
        x_bp_t = torch.from_numpy(x_bp_flat.T).unsqueeze(0).float()
        x_hf_t = torch.from_numpy(x_hf_flat.T).unsqueeze(0).float()

        # Forward
        recons, mus, logvars, _ = model(x_lf_t, x_bp_t, x_hf_t)

        # Compute anomaly score
        score = 0.0
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        for band, x_in in zip(['lf', 'bp', 'hf'], [x_lf_t, x_bp_t, x_hf_t]):
            recon_error = torch.abs(recons[band] - x_in).mean()
            kl_div = -0.5 * (1 + logvars[band] - mus[band].pow(2) - logvars[band].exp()).mean()
            score += recon_error + betas[band] * kl_div

        return score.item()

    except Exception as e:
        print(f"Error: {e}")
        return None


def compute_train_anomaly_scores(processed_dir, model, filter_bank, config, max_samples=None):
    """
    전체 학습 데이터의 anomaly score 계산
    """
    npz_files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))

    if max_samples:
        npz_files = npz_files[:max_samples]

    print(f"Computing anomaly scores for {len(npz_files)} training samples...")

    scores = []
    for npz_path in tqdm(npz_files, desc="Train data"):
        score = compute_anomaly_score_from_npz(model, npz_path, filter_bank, config)
        if score is not None and not np.isnan(score):
            scores.append(score)

    return np.array(scores)


def evaluate_with_full_train(processed_dir, real_test_dir, fake_test_dir,
                             checkpoint_path, num_test_samples=50, seed=42):
    """
    올바른 평가:
    1. 전체 train 데이터로 threshold 계산
    2. Test real + fake로 평가
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
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

    filter_bank = ButterworthFilterBank(fps=30, fc_low=2.0, fc_high=8.0, order=4)

    print(f"✓ Model loaded!")
    print(f"  Fusion weights: {torch.softmax(model.fusion_weights, dim=0).detach().numpy()}")

    # Step 1: Compute train scores (전체 또는 일부)
    print(f"\n{'='*80}")
    print("STEP 1: Computing anomaly scores for TRAINING data")
    print(f"{'='*80}")

    # 전체 데이터는 너무 많으므로 샘플링 (또는 전체 사용 가능)
    train_scores = compute_train_anomaly_scores(
        processed_dir, model, filter_bank, config,
        max_samples=5000  # 전체 57K는 너무 많으니 5000개만
    )

    print(f"\nTrain scores: n={len(train_scores)}, mean={train_scores.mean():.4f}, std={train_scores.std():.4f}")

    # Compute thresholds
    threshold_20th = np.percentile(train_scores, 20)
    threshold_50th = np.percentile(train_scores, 50)
    threshold_80th = np.percentile(train_scores, 80)
    threshold_mean = train_scores.mean()

    print(f"\nThreshold candidates (from FULL train data):")
    print(f"  20th percentile: {threshold_20th:.4f}")
    print(f"  50th percentile: {threshold_50th:.4f}")
    print(f"  80th percentile: {threshold_80th:.4f}")
    print(f"  Mean: {threshold_mean:.4f}")

    # Step 2: Load test videos
    print(f"\n{'='*80}")
    print("STEP 2: Loading TEST videos")
    print(f"{'='*80}")

    from preprocess_data import (
        VideoLandmarkExtractor,
        normalize_landmarks_procrustes,
        compute_features,
        features_to_array,
        SELECTED_IDX
    )

    real_videos = sorted(glob.glob(os.path.join(real_test_dir, "**/*.mp4"), recursive=True))
    fake_videos = sorted(glob.glob(os.path.join(fake_test_dir, "**/*.mp4"), recursive=True))

    # Sample test videos (새로운 샘플)
    random.seed(seed + 100)  # Different seed for test
    real_test = random.sample(real_videos, min(num_test_samples, len(real_videos)))
    fake_test = random.sample(fake_videos, min(num_test_samples, len(fake_videos)))

    print(f"Test set: {len(real_test)} real + {len(fake_test)} fake")

    # Step 3: Process test videos
    print(f"\n{'='*80}")
    print("STEP 3: Processing TEST videos")
    print(f"{'='*80}")

    extractor = VideoLandmarkExtractor(segment_duration=15, fps=30)

    def extract_and_score(video_path):
        """Extract features and compute score"""
        try:
            for segment_data in extractor.extract_from_video(video_path):
                landmarks = segment_data['landmarks']

                if landmarks.shape[0] < 30:
                    return None

                # Select lips
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

                T, K, F = features_array.shape

                features_lf = features_array.copy()
                features_lf[:, :, :2] = pos_lf

                features_bp = features_array.copy()
                features_bp[:, :, :2] = pos_bp

                features_hf = features_array.copy()
                features_hf[:, :, :2] = pos_hf

                # Flatten
                x_lf = features_lf.reshape(T, K * F)
                x_bp = features_bp.reshape(T, K * F)
                x_hf = features_hf.reshape(T, K * F)

                # Normalize
                x_lf = (x_lf - x_lf.mean()) / (x_lf.std() + 1e-8)
                x_bp = (x_bp - x_bp.mean()) / (x_bp.std() + 1e-8)
                x_hf = (x_hf - x_hf.mean()) / (x_hf.std() + 1e-8)

                # Convert to tensors
                x_lf_t = torch.from_numpy(x_lf.T).unsqueeze(0).float()
                x_bp_t = torch.from_numpy(x_bp.T).unsqueeze(0).float()
                x_hf_t = torch.from_numpy(x_hf.T).unsqueeze(0).float()

                # Forward
                recons, mus, logvars, _ = model(x_lf_t, x_bp_t, x_hf_t)

                # Score
                score = 0.0
                betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

                for band, x_in in zip(['lf', 'bp', 'hf'], [x_lf_t, x_bp_t, x_hf_t]):
                    recon_error = torch.abs(recons[band] - x_in).mean()
                    kl_div = -0.5 * (1 + logvars[band] - mus[band].pow(2) - logvars[band].exp()).mean()
                    score += recon_error + betas[band] * kl_div

                return score.item()
        except:
            return None
        return None

    # Process test real
    test_real_scores = []
    for video_path in tqdm(real_test, desc="Test real"):
        score = extract_and_score(video_path)
        if score is not None and not np.isnan(score):
            test_real_scores.append(score)

    # Process test fake
    test_fake_scores = []
    for video_path in tqdm(fake_test, desc="Test fake"):
        score = extract_and_score(video_path)
        if score is not None and not np.isnan(score):
            test_fake_scores.append(score)

    test_real_scores = np.array(test_real_scores)
    test_fake_scores = np.array(test_fake_scores)

    print(f"\nTest real scores: n={len(test_real_scores)}, mean={test_real_scores.mean():.4f}, std={test_real_scores.std():.4f}")
    print(f"Test fake scores: n={len(test_fake_scores)}, mean={test_fake_scores.mean():.4f}, std={test_fake_scores.std():.4f}")

    # Step 4: Evaluate
    print(f"\n{'='*80}")
    print("STEP 4: CLASSIFICATION RESULTS")
    print(f"{'='*80}")

    y_true = np.concatenate([
        np.ones(len(test_real_scores)),
        np.zeros(len(test_fake_scores))
    ])
    y_scores = np.concatenate([test_real_scores, test_fake_scores])

    for threshold_name, threshold in [
        ("20th percentile", threshold_20th),
        ("50th percentile", threshold_50th),
        ("80th percentile", threshold_80th),
        ("Mean", threshold_mean)
    ]:
        # Strategy 1: score > threshold → real
        y_pred = (y_scores > threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\nThreshold: {threshold_name} = {threshold:.4f}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f"\n{'='*80}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"{'='*80}")
    except:
        roc_auc = None

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train distribution
    ax = axes[0]
    ax.hist(train_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(threshold_mean, color='red', linestyle='--', linewidth=2, label=f'Mean={threshold_mean:.3f}')
    ax.axvline(threshold_20th, color='orange', linestyle='--', linewidth=2, label=f'20th={threshold_20th:.3f}')
    ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Training Data (Live)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Test distribution
    ax = axes[1]
    ax.hist(test_real_scores, bins=30, alpha=0.6, color='green', label='Real (Test)', edgecolor='black')
    ax.hist(test_fake_scores, bins=30, alpha=0.6, color='red', label='Fake (Test)', edgecolor='black')
    ax.axvline(threshold_mean, color='blue', linestyle='--', linewidth=2, label=f'Threshold={threshold_mean:.3f}')
    ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Test Data', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ROC curve
    if roc_auc:
        ax = axes[2]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('classifier_full_train.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: classifier_full_train.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str,
                       default="/home/elicer/liveness_detection/model1/processed_live")
    parser.add_argument("--real_test_dir", type=str,
                       default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_원본")
    parser.add_argument("--fake_test_dir", type=str,
                       default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_변조")
    parser.add_argument("--checkpoint", type=str,
                       default="/home/elicer/liveness_detection/model1/runs/bandvae_full/best.pt")
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("EVALUATION with FULL TRAINING DATA for Threshold")
    print("="*80)
    print(f"Processed train dir: {args.processed_dir}")
    print(f"Real test dir: {args.real_test_dir}")
    print(f"Fake test dir: {args.fake_test_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {args.num_test_samples} per class")
    print("="*80 + "\n")

    evaluate_with_full_train(
        args.processed_dir,
        args.real_test_dir,
        args.fake_test_dir,
        args.checkpoint,
        args.num_test_samples,
        args.seed
    )

    print("\n✓ Evaluation complete!")
