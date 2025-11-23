"""
Two-Stage Gaussian Filtering for Liveness Detection

Stage 1 Filter: Reconstruction loss within 95% confidence interval
Stage 2 Filter: Latent distribution within 95% confidence interval

Uses stage2_train real samples to fit Gaussian distributions
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import FullFeatureConfig


# ========================================
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"
MODEL_CHECKPOINT = "runs/stage2_method1_margin/stage2_method1_best.pt"
OUTPUT_DIR = "runs/gaussian_filtering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Gaussian Fitting Functions
# ========================================

@torch.no_grad()
def compute_reconstruction_losses(model, loader, device):
    """Compute reconstruction loss for each sample"""
    model.eval()

    all_losses = []
    all_labels = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        # Compute reconstruction loss for each sample
        batch_size = x_lf.size(0)
        for i in range(batch_size):
            single_recons = {k: v[i:i+1] for k, v in recons.items()}
            single_mus = {k: v[i:i+1] for k, v in mus.items()}
            single_logvars = {k: v[i:i+1] for k, v in logvars.items()}
            single_targets = {k: v[i:i+1] for k, v in targets.items()}
            single_x_hat_fused = x_hat_fused[i:i+1] if x_hat_fused is not None else None

            loss, loss_dict = band_split_vae_loss(
                single_recons, single_mus, single_logvars, single_targets,
                single_x_hat_fused, None, betas=betas, alpha_fusion=0.0
            )

            all_losses.append(loss_dict['total'])
            all_labels.append(labels[i].item())

    return np.array(all_losses), np.array(all_labels)


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

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        # Get mean of latent vectors across time dimension
        # mus['lf'] shape: (B, C_z, T) -> take mean over T
        z_lf = mus['lf'].mean(dim=2)  # (B, C_z)
        z_bp = mus['bp'].mean(dim=2)  # (B, C_z)
        z_hf = mus['hf'].mean(dim=2)  # (B, C_z)

        z_concat = torch.cat([z_lf, z_bp, z_hf], dim=1)  # (B, C_z * 3)

        all_latents.append(z_concat.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_latents = np.vstack(all_latents)
    all_labels = np.array(all_labels)

    return all_latents, all_labels


def fit_gaussian_1d(data):
    """Fit 1D Gaussian and return mean, std"""
    mu = np.mean(data)
    sigma = np.std(data)
    return mu, sigma


def fit_gaussian_multivariate(data):
    """Fit multivariate Gaussian and return mean, covariance"""
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mu, cov


def is_within_confidence_1d(x, mu, sigma, confidence=0.95):
    """Check if value is within confidence interval for 1D Gaussian"""
    # For 95% confidence interval: approximately 1.96 standard deviations
    z_score = stats.norm.ppf((1 + confidence) / 2)
    lower = mu - z_score * sigma
    upper = mu + z_score * sigma
    return lower <= x <= upper


def is_within_confidence_multivariate(x, mu, cov, confidence=0.95):
    """Check if point is within confidence ellipsoid for multivariate Gaussian"""
    # Mahalanobis distance
    try:
        diff = x - mu
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(mu)))  # Add small regularization
        mahal_dist_sq = diff @ cov_inv @ diff

        # Chi-squared distribution with dim degrees of freedom
        dim = len(mu)
        threshold = stats.chi2.ppf(confidence, dim)

        return mahal_dist_sq <= threshold
    except np.linalg.LinAlgError:
        # If covariance is singular, fall back to element-wise check
        print("Warning: Singular covariance matrix, using element-wise check")
        sigma = np.sqrt(np.diag(cov))
        z_score = stats.norm.ppf((1 + confidence) / 2)
        within = np.all(np.abs(x - mu) <= z_score * sigma)
        return within


# ========================================
# Two-Stage Filtering
# ========================================

def two_stage_filtering(model, train_loader, test_loader, device, confidence=0.95):
    """
    Two-stage filtering:
    1. Stage 1: Filter by reconstruction loss
    2. Stage 2: Classify by latent distribution
    """

    print("="*80)
    print("Two-Stage Gaussian Filtering")
    print("="*80)

    # ========================================
    # Fit Gaussians on Training Data (Real only)
    # ========================================
    print("\nStep 1: Fitting Gaussians on stage2_train real samples...")

    train_losses, train_labels = compute_reconstruction_losses(model, train_loader, device)
    train_latents, _ = extract_latent_vectors(model, train_loader, device)

    # Filter only real samples
    train_real_losses = train_losses[train_labels == 0]
    train_real_latents = train_latents[train_labels == 0]

    print(f"Number of real training samples: {len(train_real_losses)}")

    # Fit 1D Gaussian for reconstruction loss
    loss_mu, loss_sigma = fit_gaussian_1d(train_real_losses)
    print(f"Reconstruction loss: mu={loss_mu:.4f}, sigma={loss_sigma:.4f}")

    # Fit multivariate Gaussian for latent vectors
    latent_mu, latent_cov = fit_gaussian_multivariate(train_real_latents)
    print(f"Latent dimension: {len(latent_mu)}")
    print(f"Latent mean (first 5): {latent_mu[:5]}")

    # ========================================
    # Apply Two-Stage Filter on Test Data
    # ========================================
    print(f"\nStep 2: Applying two-stage filter on final_test samples...")

    test_losses, test_labels = compute_reconstruction_losses(model, test_loader, device)
    test_latents, _ = extract_latent_vectors(model, test_loader, device)

    num_test = len(test_losses)
    print(f"Number of test samples: {num_test}")
    print(f"  Real: {np.sum(test_labels == 0)}")
    print(f"  Fake: {np.sum(test_labels == 1)}")

    # Apply filtering
    predictions = []
    stage1_pass_count = 0

    for i in range(num_test):
        loss = test_losses[i]
        latent = test_latents[i]

        # Stage 1: Reconstruction loss filter
        stage1_pass = is_within_confidence_1d(loss, loss_mu, loss_sigma, confidence)

        if stage1_pass:
            stage1_pass_count += 1
            # Stage 2: Latent distribution filter
            stage2_pass = is_within_confidence_multivariate(latent, latent_mu, latent_cov, confidence)

            if stage2_pass:
                predictions.append(0)  # Real
            else:
                predictions.append(1)  # Fake
        else:
            # Failed stage 1: classify as fake
            predictions.append(1)  # Fake

    predictions = np.array(predictions)

    print(f"\nFiltering Statistics:")
    print(f"  Passed Stage 1 filter: {stage1_pass_count}/{num_test} ({100*stage1_pass_count/num_test:.2f}%)")
    print(f"  Classified as Real: {np.sum(predictions == 0)}/{num_test} ({100*np.sum(predictions == 0)/num_test:.2f}%)")
    print(f"  Classified as Fake: {np.sum(predictions == 1)}/{num_test} ({100*np.sum(predictions == 1)/num_test:.2f}%)")

    # ========================================
    # Compute Metrics
    # ========================================
    print(f"\nStep 3: Computing metrics...")

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)

    # Metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {tn:4d}    {fp:4d}")
    print(f"       Fake   {fn:4d}    {tp:4d}")

    return {
        'predictions': predictions,
        'labels': test_labels,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'stage1_pass_count': stage1_pass_count,
        'loss_mu': loss_mu,
        'loss_sigma': loss_sigma,
        'test_losses': test_losses,
        'test_latents': test_latents
    }


# ========================================
# Visualization
# ========================================

def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix\n(Two-Stage Gaussian Filtering)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add percentages as text
    for i in range(2):
        for j in range(2):
            percentage = cm_normalized[i, j] * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def plot_loss_distributions(results, output_path):
    """Plot reconstruction loss distributions with filtering threshold"""
    test_losses = results['test_losses']
    labels = results['labels']
    loss_mu = results['loss_mu']
    loss_sigma = results['loss_sigma']

    # 95% confidence interval
    z_score = stats.norm.ppf(0.975)  # 95% CI
    lower = loss_mu - z_score * loss_sigma
    upper = loss_mu + z_score * loss_sigma

    plt.figure(figsize=(12, 6))

    real_losses = test_losses[labels == 0]
    fake_losses = test_losses[labels == 1]

    plt.hist(real_losses, bins=50, alpha=0.6, label='Real', color='blue', edgecolor='black')
    plt.hist(fake_losses, bins=50, alpha=0.6, label='Fake', color='red', edgecolor='black')

    plt.axvline(lower, color='green', linestyle='--', linewidth=2, label=f'95% CI Lower: {lower:.4f}')
    plt.axvline(upper, color='orange', linestyle='--', linewidth=2, label=f'95% CI Upper: {upper:.4f}')
    plt.axvline(loss_mu, color='black', linestyle='-', linewidth=2, label=f'Mean: {loss_mu:.4f}')

    plt.title('Reconstruction Loss Distribution\n(Stage 1 Filter)', fontsize=14, fontweight='bold')
    plt.xlabel('Reconstruction Loss', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss distributions to: {output_path}")


# ========================================
# Main
# ========================================

def main():
    print("="*80)
    print("Two-Stage Gaussian Filtering Evaluation")
    print("="*80)
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    print()

    # Load configuration
    config = FullFeatureConfig()
    config.T_fixed = 300

    # ========================================
    # Load Datasets
    # ========================================
    print("Loading datasets...")

    # Training dataset (to fit Gaussians)
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

    # Test dataset
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

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print()

    # ========================================
    # Load Model
    # ========================================
    print("Loading model...")
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=checkpoint['config'].C_h,
        C_z=checkpoint['config'].C_z,
        dilations=checkpoint['config'].dilations
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print()

    # ========================================
    # Run Two-Stage Filtering
    # ========================================
    results = two_stage_filtering(model, train_loader, test_loader, device, confidence=0.95)

    # ========================================
    # Save Results
    # ========================================
    print("\nSaving results...")

    # Save metrics
    metrics = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'specificity': float(results['specificity']),
        'stage1_pass_rate': float(results['stage1_pass_count'] / len(results['labels'])),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'loss_mean': float(results['loss_mu']),
        'loss_std': float(results['loss_sigma'])
    }

    with open(os.path.join(OUTPUT_DIR, 'gaussian_filtering_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {os.path.join(OUTPUT_DIR, 'gaussian_filtering_metrics.json')}")

    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    )

    # Plot loss distributions
    plot_loss_distributions(
        results,
        os.path.join(OUTPUT_DIR, 'loss_distributions.png')
    )

    print()
    print("="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
