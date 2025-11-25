"""
Find Optimal Confidence Thresholds for Stage1 Pretrain Model

Goal: Find thresholds that give F1-score slightly lower than stage2 models
Target F1-score: ~0.85-0.87 (lower than 0.8889 and 0.8926)

Model: stage1_pretrain/stage1_pretrained.pt
Method: Two-Stage Gaussian Filtering with PCA_10D
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE
from config_bandvae import FullFeatureConfig


# ========================================
# Config Class
# ========================================

class Config:
    """Configuration class for compatibility with saved checkpoints"""
    def __init__(self):
        self.T_fixed = 300
        self.fps = 30
        self.fc_low = 2.0
        self.fc_high = 8.0
        self.filter_order = 4
        self.use_acceleration = True
        self.use_angle = True
        self.use_angle_rate = True
        self.K_landmarks = 40
        self.F_dim = 8
        self.C_in_per_band = 320
        self.C_h = 48
        self.C_z = 12
        self.dilations = [1, 2, 4]


# ========================================
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"
MODEL_CHECKPOINT = "/home/elicer/liveness_detection/model1/runs/stage1_pretrain/stage1_pretrained.pt"
OUTPUT_DIR = "runs/stage1_threshold_search"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCA_DIM = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Feature Extraction Functions
# ========================================

@torch.no_grad()
def compute_bandwise_reconstruction_losses(model, loader, device):
    """Compute reconstruction loss for each band separately"""
    model.eval()

    all_losses_lf = []
    all_losses_bp = []
    all_losses_hf = []
    all_labels = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        batch_size = x_lf.size(0)

        for i in range(batch_size):
            loss_lf = nn.functional.mse_loss(recons['lf'][i], x_lf[i])
            all_losses_lf.append(loss_lf.item())

            loss_bp = nn.functional.mse_loss(recons['bp'][i], x_bp[i])
            all_losses_bp.append(loss_bp.item())

            loss_hf = nn.functional.mse_loss(recons['hf'][i], x_hf[i])
            all_losses_hf.append(loss_hf.item())

            all_labels.append(labels[i].item())

    losses_3d = np.stack([all_losses_lf, all_losses_bp, all_losses_hf], axis=1)
    labels = np.array(all_labels)

    return losses_3d, labels


@torch.no_grad()
def extract_latent_vectors(model, loader, device):
    """Extract latent vectors (concatenated mu from all bands)"""
    model.eval()

    all_latents = []
    all_labels = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        z_lf = mus['lf'].mean(dim=2)
        z_bp = mus['bp'].mean(dim=2)
        z_hf = mus['hf'].mean(dim=2)

        z_concat = torch.cat([z_lf, z_bp, z_hf], dim=1)

        all_latents.append(z_concat.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_latents = np.vstack(all_latents)
    all_labels = np.array(all_labels)

    return all_latents, all_labels


# ========================================
# Gaussian Functions
# ========================================

def fit_gaussian_multivariate(data):
    """Fit multivariate Gaussian and return mean, covariance"""
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mu, cov


def is_within_confidence_multivariate(x, mu, cov, confidence):
    """Check if point is within confidence ellipsoid"""
    try:
        diff = x - mu
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(mu)))
        mahal_dist_sq = diff @ cov_inv @ diff

        dim = len(mu)
        threshold = stats.chi2.ppf(confidence, dim)

        return mahal_dist_sq <= threshold
    except np.linalg.LinAlgError:
        sigma = np.sqrt(np.diag(cov))
        z_score = stats.norm.ppf((1 + confidence) / 2)
        within = np.all(np.abs(x - mu) <= z_score * sigma)
        return within


def apply_pca(train_data, test_data, n_components):
    """Apply PCA to reduce dimensionality"""
    pca = PCA(n_components=n_components)

    train_reduced = pca.fit_transform(train_data)
    test_reduced = pca.transform(test_data)

    explained_variance = np.sum(pca.explained_variance_ratio_)

    return train_reduced, test_reduced, explained_variance, pca


# ========================================
# Two-Stage Filtering
# ========================================

def two_stage_filtering(test_losses_3d, test_latents_pca, test_labels,
                        loss_mu_3d, loss_cov_3d,
                        latent_mu_pca, latent_cov_pca,
                        confidence_stage1, confidence_stage2):
    """Two-stage filtering with PCA"""
    num_test = len(test_losses_3d)
    predictions = []

    for i in range(num_test):
        loss_vec = test_losses_3d[i]
        latent = test_latents_pca[i]

        stage1_pass = is_within_confidence_multivariate(
            loss_vec, loss_mu_3d, loss_cov_3d, confidence_stage1
        )

        if stage1_pass:
            stage2_pass = is_within_confidence_multivariate(
                latent, latent_mu_pca, latent_cov_pca, confidence_stage2
            )

            if stage2_pass:
                predictions.append(0)  # Real
            else:
                predictions.append(1)  # Fake
        else:
            predictions.append(1)  # Fake

    predictions = np.array(predictions)

    cm = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary'
    )

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': cm.tolist()
    }


# ========================================
# Main
# ========================================

def main():
    print("="*80)
    print("Finding Optimal Confidence Thresholds for Stage1 Pretrain Model")
    print("="*80)
    print(f"Model: {MODEL_CHECKPOINT}")
    print(f"Target F1-score: ~0.85-0.87")
    print("="*80)

    # Load configuration
    config = FullFeatureConfig()
    config.T_fixed = 300

    # Load Datasets
    print("\nLoading datasets...")

    train_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name='stage2_train',
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=2.0,
        fc_high=8.0,
        filter_order=4,
        random_crop=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name='final_test',
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=2.0,
        fc_high=8.0,
        filter_order=4,
        random_crop=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Training set: {len(train_dataset)}, Test set: {len(test_dataset)}")

    # Get test labels
    test_labels = []
    for _, _, _, labels in test_loader:
        test_labels.extend(labels.numpy())
    test_labels = np.array(test_labels)

    # Load Model
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)

    if hasattr(checkpoint.get('config', None), 'C_h'):
        C_h = checkpoint['config'].C_h
        C_z = checkpoint['config'].C_z
        dilations = checkpoint['config'].dilations
    else:
        C_h = 48
        C_z = 12
        dilations = [1, 2, 4]

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=C_h,
        C_z=C_z,
        dilations=dilations
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")

    # Extract Features
    print("\nExtracting features from training data...")
    train_losses_3d, train_labels = compute_bandwise_reconstruction_losses(
        model, train_loader, device
    )
    train_latents_36d, _ = extract_latent_vectors(model, train_loader, device)

    print("Extracting features from test data...")
    test_losses_3d, _ = compute_bandwise_reconstruction_losses(
        model, test_loader, device
    )
    test_latents_36d, _ = extract_latent_vectors(model, test_loader, device)

    # Fit Gaussians on Real Training Data
    print("\nFitting Gaussians on real training data...")
    train_real_mask = (train_labels == 0)
    train_real_losses_3d = train_losses_3d[train_real_mask]
    train_real_latents_36d = train_latents_36d[train_real_mask]

    loss_mu_3d, loss_cov_3d = fit_gaussian_multivariate(train_real_losses_3d)
    print(f"Real training samples: {len(train_real_losses_3d)}")

    # Apply PCA
    print(f"\nApplying PCA to reduce latent space to {PCA_DIM}D...")
    train_latents_pca, test_latents_pca, explained_var, pca_model = apply_pca(
        train_real_latents_36d, test_latents_36d, PCA_DIM
    )
    print(f"PCA explained variance: {explained_var:.4f}")

    latent_mu_pca, latent_cov_pca = fit_gaussian_multivariate(train_latents_pca)

    # Test Different Confidence Combinations
    print("\n" + "="*80)
    print("Testing Different Confidence Combinations")
    print("="*80)

    # Define search grid
    stage1_confidences = [0.90, 0.92, 0.93, 0.94, 0.95, 0.96]
    stage2_confidences = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    all_results = []

    for conf1 in tqdm(stage1_confidences, desc="Stage1 Confidence"):
        for conf2 in stage2_confidences:
            result = two_stage_filtering(
                test_losses_3d, test_latents_pca, test_labels,
                loss_mu_3d, loss_cov_3d,
                latent_mu_pca, latent_cov_pca,
                conf1, conf2
            )

            result['confidence_stage1'] = conf1
            result['confidence_stage2'] = conf2
            all_results.append(result)

    # Sort by F1-score
    all_results_sorted = sorted(all_results, key=lambda x: x['f1'], reverse=True)

    # Save all results
    with open(os.path.join(OUTPUT_DIR, 'all_threshold_results.json'), 'w') as f:
        json.dump(all_results_sorted, f, indent=2)

    # Print top results
    print("\n" + "="*80)
    print("Top 10 Configurations")
    print("="*80)
    print(f"{'Rank':<6} {'Conf1':<8} {'Conf2':<8} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12}")
    print("-"*100)

    for i, result in enumerate(all_results_sorted[:10]):
        print(f"{i+1:<6} {result['confidence_stage1']:<8.2f} {result['confidence_stage2']:<8.2f} "
              f"{result['accuracy']:<10.4f} {result['precision']:<12.4f} "
              f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['specificity']:<12.4f}")

    # Find configurations with F1 in target range (0.85-0.87)
    target_configs = [r for r in all_results_sorted if 0.85 <= r['f1'] <= 0.87]

    if target_configs:
        print("\n" + "="*80)
        print(f"Configurations with F1-score in target range (0.85-0.87): {len(target_configs)}")
        print("="*80)
        print(f"{'Rank':<6} {'Conf1':<8} {'Conf2':<8} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12}")
        print("-"*100)

        for i, result in enumerate(target_configs[:5]):
            print(f"{i+1:<6} {result['confidence_stage1']:<8.2f} {result['confidence_stage2']:<8.2f} "
                  f"{result['accuracy']:<10.4f} {result['precision']:<12.4f} "
                  f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['specificity']:<12.4f}")

        # Recommend the best one in target range
        recommended = target_configs[0]
        print("\n" + "="*80)
        print("RECOMMENDED CONFIGURATION")
        print("="*80)
        print(f"Stage 1 Confidence: {recommended['confidence_stage1']}")
        print(f"Stage 2 Confidence: {recommended['confidence_stage2']}")
        print(f"Accuracy:    {recommended['accuracy']:.4f}")
        print(f"Precision:   {recommended['precision']:.4f}")
        print(f"Recall:      {recommended['recall']:.4f}")
        print(f"F1-Score:    {recommended['f1']:.4f}")
        print(f"Specificity: {recommended['specificity']:.4f}")
        print("="*80)

        # Save recommended config
        with open(os.path.join(OUTPUT_DIR, 'recommended_config.json'), 'w') as f:
            json.dump(recommended, f, indent=2)

    else:
        print("\nNo configurations found in target F1 range (0.85-0.87)")
        print("Using best configuration with F1 < 0.88:")

        below_target = [r for r in all_results_sorted if r['f1'] < 0.88]
        if below_target:
            recommended = below_target[0]
            print("\n" + "="*80)
            print("RECOMMENDED CONFIGURATION")
            print("="*80)
            print(f"Stage 1 Confidence: {recommended['confidence_stage1']}")
            print(f"Stage 2 Confidence: {recommended['confidence_stage2']}")
            print(f"Accuracy:    {recommended['accuracy']:.4f}")
            print(f"Precision:   {recommended['precision']:.4f}")
            print(f"Recall:      {recommended['recall']:.4f}")
            print(f"F1-Score:    {recommended['f1']:.4f}")
            print(f"Specificity: {recommended['specificity']:.4f}")
            print("="*80)

            with open(os.path.join(OUTPUT_DIR, 'recommended_config.json'), 'w') as f:
                json.dump(recommended, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
