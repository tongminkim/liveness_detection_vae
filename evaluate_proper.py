"""
올바른 Train/Test 분리로 평가
- Train: processed_live (57K)
- Test Real: processed_live_test (190)
- Test Fake: 원천데이터/train_변조 (샘플링)
"""

import os
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    """NPZ 파일로부터 anomaly score 계산"""
    try:
        data = np.load(npz_path)
        lips_outer = data['lips_outer']
        lips_inner = data['lips_inner']

        lips_landmarks = np.concatenate([
            lips_outer[:, :20, :],
            lips_inner[:, :20, :]
        ], axis=1)

        T = lips_landmarks.shape[0]
        if T < 30:
            return None

        position = lips_landmarks
        pos_lf, pos_bp, pos_hf = filter_bank.apply(position)

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

        def stack_features(feat_dict):
            parts = [feat_dict['position']]
            if config.use_velocity:
                parts.append(feat_dict['velocity'])
            if config.use_acceleration:
                parts.append(feat_dict['acceleration'])
            if config.use_angle:
                parts.append(feat_dict['angle'])
            if config.use_angle_rate:
                parts.append(feat_dict['angle_rate'])
            return np.concatenate(parts, axis=2)

        x_lf = stack_features(features_lf)
        x_bp = stack_features(features_bp)
        x_hf = stack_features(features_hf)

        T, K, F = x_lf.shape
        x_lf_flat = x_lf.reshape(T, K * F)
        x_bp_flat = x_bp.reshape(T, K * F)
        x_hf_flat = x_hf.reshape(T, K * F)

        x_lf_flat = (x_lf_flat - x_lf_flat.mean()) / (x_lf_flat.std() + 1e-8)
        x_bp_flat = (x_bp_flat - x_bp_flat.mean()) / (x_bp_flat.std() + 1e-8)
        x_hf_flat = (x_hf_flat - x_hf_flat.mean()) / (x_hf_flat.std() + 1e-8)

        x_lf_t = torch.from_numpy(x_lf_flat.T).unsqueeze(0).float()
        x_bp_t = torch.from_numpy(x_bp_flat.T).unsqueeze(0).float()
        x_hf_t = torch.from_numpy(x_hf_flat.T).unsqueeze(0).float()

        recons, mus, logvars, _ = model(x_lf_t, x_bp_t, x_hf_t)

        score = 0.0
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        for band, x_in in zip(['lf', 'bp', 'hf'], [x_lf_t, x_bp_t, x_hf_t]):
            recon_error = torch.abs(recons[band] - x_in).mean()
            kl_div = -0.5 * (1 + logvars[band] - mus[band].pow(2) - logvars[band].exp()).mean()
            score += recon_error + betas[band] * kl_div

        return score.item()

    except Exception as e:
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str,
                       default="/home/elicer/liveness_detection/model1/processed_live")
    parser.add_argument("--test_real_dir", type=str,
                       default="/home/elicer/liveness_detection/model1/processed_live_test")
    parser.add_argument("--test_fake_dir", type=str,
                       default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_변조")
    parser.add_argument("--checkpoint", type=str,
                       default="/home/elicer/liveness_detection/model1/runs/bandvae_full/best.pt")
    parser.add_argument("--train_samples", type=int, default=5000,
                       help="Number of train samples to use (from 57K)")
    parser.add_argument("--test_fake_samples", type=int, default=190,
                       help="Number of fake test samples")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "="*80)
    print("PROPER EVALUATION with Correct Train/Test Split")
    print("="*80)
    print(f"Train dir: {args.train_dir}")
    print(f"Test real dir: {args.test_real_dir}")
    print(f"Test fake dir: {args.test_fake_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*80 + "\n")

    # Load model
    print("Loading model...")
    config = FullFeatureConfig()
    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    filter_bank = ButterworthFilterBank(fps=30, fc_low=2.0, fc_high=8.0, order=4)

    print(f"✓ Model loaded!")
    print(f"  Fusion weights: {torch.softmax(model.fusion_weights, dim=0).detach().numpy()}")

    # Step 1: Compute train scores
    print(f"\n{'='*80}")
    print(f"STEP 1: Computing anomaly scores for TRAINING data")
    print(f"{'='*80}")

    train_npz_files = sorted(glob.glob(os.path.join(args.train_dir, "*.npz")))
    print(f"Total train files: {len(train_npz_files)}")

    if args.train_samples:
        train_npz_files = random.sample(train_npz_files, min(args.train_samples, len(train_npz_files)))

    print(f"Using {len(train_npz_files)} train samples...")

    train_scores = []
    for npz_path in tqdm(train_npz_files, desc="Train"):
        score = compute_anomaly_score_from_npz(model, npz_path, filter_bank, config)
        if score is not None and not np.isnan(score):
            train_scores.append(score)

    train_scores = np.array(train_scores)
    print(f"\nTrain scores: n={len(train_scores)}, mean={train_scores.mean():.4f}, std={train_scores.std():.4f}")

    # Thresholds
    threshold_20th = np.percentile(train_scores, 20)
    threshold_50th = np.percentile(train_scores, 50)
    threshold_80th = np.percentile(train_scores, 80)
    threshold_mean = train_scores.mean()

    print(f"\nThresholds (from full train data):")
    print(f"  20th percentile: {threshold_20th:.4f}")
    print(f"  50th percentile: {threshold_50th:.4f}")
    print(f"  80th percentile: {threshold_80th:.4f}")
    print(f"  Mean: {threshold_mean:.4f}")

    # Step 2: Test real scores
    print(f"\n{'='*80}")
    print(f"STEP 2: Computing scores for TEST REAL data")
    print(f"{'='*80}")

    test_real_files = sorted(glob.glob(os.path.join(args.test_real_dir, "*.npz")))
    print(f"Test real files: {len(test_real_files)}")

    test_real_scores = []
    for npz_path in tqdm(test_real_files, desc="Test real"):
        score = compute_anomaly_score_from_npz(model, npz_path, filter_bank, config)
        if score is not None and not np.isnan(score):
            test_real_scores.append(score)

    test_real_scores = np.array(test_real_scores)
    print(f"\nTest real scores: n={len(test_real_scores)}, mean={test_real_scores.mean():.4f}, std={test_real_scores.std():.4f}")

    # Step 3: Test fake scores (process videos)
    print(f"\n{'='*80}")
    print(f"STEP 3: Processing TEST FAKE videos")
    print(f"{'='*80}")

    from preprocess_data import (
        VideoLandmarkExtractor,
        normalize_landmarks_procrustes,
        compute_features,
        features_to_array
    )

    fake_videos = sorted(glob.glob(os.path.join(args.test_fake_dir, "**/*.mp4"), recursive=True))
    print(f"Total fake videos: {len(fake_videos)}")

    fake_videos_sample = random.sample(fake_videos, min(args.test_fake_samples, len(fake_videos)))
    print(f"Using {len(fake_videos_sample)} fake samples...")

    extractor = VideoLandmarkExtractor(segment_duration=15, fps=30)

    test_fake_scores = []
    for video_path in tqdm(fake_videos_sample, desc="Test fake"):
        try:
            for segment_data in extractor.extract_from_video(video_path):
                landmarks = segment_data['landmarks']

                if landmarks.shape[0] < 30:
                    continue

                lips_start = 16 + 16
                lips_landmarks = landmarks[:, lips_start:lips_start+40, :]
                normalized = normalize_landmarks_procrustes(lips_landmarks, ref_frames=10)

                features = compute_features(normalized, fps=30)
                features_array = features_to_array(features)

                position = normalized
                pos_lf, pos_bp, pos_hf = filter_bank.apply(position)

                T, K, F = features_array.shape

                features_lf = features_array.copy()
                features_lf[:, :, :2] = pos_lf

                features_bp = features_array.copy()
                features_bp[:, :, :2] = pos_bp

                features_hf = features_array.copy()
                features_hf[:, :, :2] = pos_hf

                x_lf = features_lf.reshape(T, K * F)
                x_bp = features_bp.reshape(T, K * F)
                x_hf = features_hf.reshape(T, K * F)

                x_lf = (x_lf - x_lf.mean()) / (x_lf.std() + 1e-8)
                x_bp = (x_bp - x_bp.mean()) / (x_bp.std() + 1e-8)
                x_hf = (x_hf - x_hf.mean()) / (x_hf.std() + 1e-8)

                x_lf_t = torch.from_numpy(x_lf.T).unsqueeze(0).float()
                x_bp_t = torch.from_numpy(x_bp.T).unsqueeze(0).float()
                x_hf_t = torch.from_numpy(x_hf.T).unsqueeze(0).float()

                recons, mus, logvars, _ = model(x_lf_t, x_bp_t, x_hf_t)

                score = 0.0
                betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

                for band, x_in in zip(['lf', 'bp', 'hf'], [x_lf_t, x_bp_t, x_hf_t]):
                    recon_error = torch.abs(recons[band] - x_in).mean()
                    kl_div = -0.5 * (1 + logvars[band] - mus[band].pow(2) - logvars[band].exp()).mean()
                    score += recon_error + betas[band] * kl_div

                if not np.isnan(score.item()):
                    test_fake_scores.append(score.item())
                break  # Only first segment
        except Exception as e:
            print(f"Error processing video: {e}")
            continue

    test_fake_scores = np.array(test_fake_scores)
    print(f"\nTest fake scores: n={len(test_fake_scores)}, mean={test_fake_scores.mean():.4f}, std={test_fake_scores.std():.4f}")

    # Step 4: Evaluate
    print(f"\n{'='*80}")
    print(f"STEP 4: CLASSIFICATION RESULTS")
    print(f"{'='*80}")

    y_true = np.concatenate([
        np.ones(len(test_real_scores)),
        np.zeros(len(test_fake_scores))
    ])
    y_scores = np.concatenate([test_real_scores, test_fake_scores])

    best_acc = 0
    best_threshold = None

    for threshold_name, threshold in [
        ("20th percentile", threshold_20th),
        ("50th percentile", threshold_50th),
        ("80th percentile", threshold_80th),
        ("Mean", threshold_mean)
    ]:
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

        if acc > best_acc:
            best_acc = acc
            best_threshold = (threshold_name, threshold)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f"\n{'='*80}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"{'='*80}")
        print(f"\nBest: {best_threshold[0]} ({best_threshold[1]:.4f}) = {best_acc:.4f} accuracy")
    except:
        roc_auc = None

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(train_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(threshold_mean, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Training Data (Live)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(test_real_scores, bins=30, alpha=0.6, color='green', label='Real', edgecolor='black')
    ax.hist(test_fake_scores, bins=30, alpha=0.6, color='red', label='Fake', edgecolor='black')
    ax.axvline(threshold_mean, color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Test Data', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if roc_auc:
        ax = axes[2]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC={roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('FPR', fontsize=12, fontweight='bold')
        ax.set_ylabel('TPR', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('classifier_proper.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: classifier_proper.png")


if __name__ == "__main__":
    main()
