"""
Evaluate Two Models on Final Test Set
Using Two-Stage Gaussian Filtering with PCA_10D

Models:
1. method1_margin_best.pt (Margin-based training)
2. method2_discriminator_best.pt (Discriminator-based training)

Optimal Configuration (from gaussian_filtering_advanced):
- PCA dimensions: 10
- Stage 1 confidence: 0.95
- Stage 2 confidence: 0.70
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
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import FullFeatureConfig


# ========================================
# Config Class (for loading checkpoints)
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
        self.batch_size = 64
        self.lr = 1e-4
        self.epochs = 10
        self.num_workers = 4
        self.margin = 0.5
        self.lambda_margin = 1.0
        self.lambda_cls = 1.0


# ========================================
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"

# Two models to evaluate
MODELS = {
    'method1_margin': '/home/elicer/liveness_detection/model1/runs/method1_margin_best.pt',
    'method2_discriminator': '/home/elicer/liveness_detection/model1/runs/method2_discriminator_best.pt'
}

OUTPUT_DIR = "runs/two_models_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimal hyperparameters (from gaussian_filtering_advanced)
PCA_DIM = 10
CONFIDENCE_STAGE1 = 0.95
CONFIDENCE_STAGE2 = 0.70

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

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        # Compute loss for each band separately
        batch_size = x_lf.size(0)

        for i in range(batch_size):
            # LF band reconstruction loss
            loss_lf = nn.functional.mse_loss(recons['lf'][i], x_lf[i])
            all_losses_lf.append(loss_lf.item())

            # BP band reconstruction loss
            loss_bp = nn.functional.mse_loss(recons['bp'][i], x_bp[i])
            all_losses_bp.append(loss_bp.item())

            # HF band reconstruction loss
            loss_hf = nn.functional.mse_loss(recons['hf'][i], x_hf[i])
            all_losses_hf.append(loss_hf.item())

            all_labels.append(labels[i].item())

    # Stack into (N, 3) array
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

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        # Get mean of latent vectors across time dimension
        z_lf = mus['lf'].mean(dim=2)  # (B, C_z)
        z_bp = mus['bp'].mean(dim=2)  # (B, C_z)
        z_hf = mus['hf'].mean(dim=2)  # (B, C_z)

        z_concat = torch.cat([z_lf, z_bp, z_hf], dim=1)  # (B, C_z * 3)

        all_latents.append(z_concat.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_latents = np.vstack(all_latents)
    all_labels = np.array(all_labels)

    return all_latents, all_labels


# ========================================
# Gaussian Fitting Functions
# ========================================

def fit_gaussian_multivariate(data):
    """Fit multivariate Gaussian and return mean, covariance"""
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mu, cov


def is_within_confidence_multivariate(x, mu, cov, confidence):
    """Check if point is within confidence ellipsoid for multivariate Gaussian"""
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


# ========================================
# PCA Dimension Reduction
# ========================================

def apply_pca(train_data, test_data, n_components):
    """Apply PCA to reduce dimensionality"""
    pca = PCA(n_components=n_components)

    # Fit on training data
    train_reduced = pca.fit_transform(train_data)

    # Transform test data
    test_reduced = pca.transform(test_data)

    explained_variance = np.sum(pca.explained_variance_ratio_)

    return train_reduced, test_reduced, explained_variance, pca


# ========================================
# Two-Stage Filtering
# ========================================

def two_stage_filtering_pca10d(test_losses_3d, test_latents_pca, test_labels,
                                 loss_mu_3d, loss_cov_3d,
                                 latent_mu_pca, latent_cov_pca,
                                 confidence_stage1, confidence_stage2):
    """
    Two-stage filtering with PCA_10D

    Stage 1: 3D Gaussian on band-wise reconstruction losses
    Stage 2: 10D PCA-reduced latent space
    """
    num_test = len(test_losses_3d)
    predictions = []
    stage1_pass_count = 0
    stage2_pass_count = 0

    for i in range(num_test):
        loss_vec = test_losses_3d[i]  # (3,)
        latent = test_latents_pca[i]  # (10,)

        # Stage 1: 3D Gaussian on band-wise reconstruction losses
        stage1_pass = is_within_confidence_multivariate(
            loss_vec, loss_mu_3d, loss_cov_3d, confidence_stage1
        )

        if stage1_pass:
            stage1_pass_count += 1
            # Stage 2: Latent distribution filter
            stage2_pass = is_within_confidence_multivariate(
                latent, latent_mu_pca, latent_cov_pca, confidence_stage2
            )

            if stage2_pass:
                stage2_pass_count += 1
                predictions.append(0)  # Real
            else:
                predictions.append(1)  # Fake
        else:
            predictions.append(1)  # Fake

    predictions = np.array(predictions)

    # Compute metrics
    cm = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary'
    )

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'predictions': predictions,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'stage1_pass_count': stage1_pass_count,
        'stage2_pass_count': stage2_pass_count,
        'stage1_pass_rate': stage1_pass_count / num_test,
        'stage2_pass_rate': stage2_pass_count / num_test,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


# ========================================
# Model Evaluation
# ========================================

def evaluate_model(model_name, model_checkpoint, train_loader, test_loader, test_labels, device):
    """Evaluate a single model"""
    print("\n" + "="*80)
    print(f"Evaluating: {model_name}")
    print("="*80)

    # Load Model
    print(f"Loading model from: {model_checkpoint}")
    checkpoint = torch.load(model_checkpoint, map_location=device)

    config = FullFeatureConfig()
    config.T_fixed = 300

    # Get config from checkpoint or use default values
    if hasattr(checkpoint.get('config', None), 'C_h'):
        C_h = checkpoint['config'].C_h
        C_z = checkpoint['config'].C_z
        dilations = checkpoint['config'].dilations
    else:
        # Use default values if config doesn't have these attributes
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

    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # Extract Features from Training Data
    print("\nExtracting features from training data...")
    train_losses_3d, train_labels = compute_bandwise_reconstruction_losses(
        model, train_loader, device
    )
    train_latents_36d, _ = extract_latent_vectors(model, train_loader, device)

    print(f"Training features: losses={train_losses_3d.shape}, latents={train_latents_36d.shape}")

    # Extract Features from Test Data
    print("Extracting features from test data...")
    test_losses_3d, _ = compute_bandwise_reconstruction_losses(
        model, test_loader, device
    )
    test_latents_36d, _ = extract_latent_vectors(model, test_loader, device)

    print(f"Test features: losses={test_losses_3d.shape}, latents={test_latents_36d.shape}")

    # Fit Stage 1 Gaussian (3D on losses) - Real samples only
    print("\nFitting Stage 1 Gaussian (3D band-wise losses)...")
    train_real_mask = (train_labels == 0)
    train_real_losses_3d = train_losses_3d[train_real_mask]
    train_real_latents_36d = train_latents_36d[train_real_mask]

    loss_mu_3d, loss_cov_3d = fit_gaussian_multivariate(train_real_losses_3d)
    print(f"Real training samples: {len(train_real_losses_3d)}")
    print(f"3D Loss mean: {loss_mu_3d}")

    # Apply PCA and Fit Stage 2 Gaussian
    print(f"\nApplying PCA to reduce latent space to {PCA_DIM}D...")
    train_latents_pca, test_latents_pca, explained_var, pca_model = apply_pca(
        train_real_latents_36d, test_latents_36d, PCA_DIM
    )
    print(f"PCA explained variance: {explained_var:.4f}")

    print("Fitting Stage 2 Gaussian (PCA-reduced latent space)...")
    latent_mu_pca, latent_cov_pca = fit_gaussian_multivariate(train_latents_pca)

    # Apply Two-Stage Filtering
    print("\nApplying two-stage filtering on test data...")
    results = two_stage_filtering_pca10d(
        test_losses_3d, test_latents_pca, test_labels,
        loss_mu_3d, loss_cov_3d,
        latent_mu_pca, latent_cov_pca,
        CONFIDENCE_STAGE1, CONFIDENCE_STAGE2
    )

    # Add PCA explained variance to results
    results['explained_variance'] = explained_var

    # Print Results
    print("\n" + "-"*80)
    print(f"Results for {model_name}")
    print("-"*80)
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"F1-Score:    {results['f1']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {results['tn']:4d}    {results['fp']:4d}")
    print(f"       Fake   {results['fn']:4d}    {results['tp']:4d}")
    print("-"*80)

    return results


# ========================================
# Visualization
# ========================================

def plot_comparison(results_dict, output_dir):
    """Plot comparison of two models"""

    # Prepare data
    model_names = list(results_dict.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Plot 1: Metrics Comparison Bar Chart
    ax1 = axes[0, 0]
    x = np.arange(len(metrics))
    width = 0.35

    values_model1 = [
        results_dict[model_names[0]]['accuracy'],
        results_dict[model_names[0]]['precision'],
        results_dict[model_names[0]]['recall'],
        results_dict[model_names[0]]['f1'],
        results_dict[model_names[0]]['specificity']
    ]

    values_model2 = [
        results_dict[model_names[1]]['accuracy'],
        results_dict[model_names[1]]['precision'],
        results_dict[model_names[1]]['recall'],
        results_dict[model_names[1]]['f1'],
        results_dict[model_names[1]]['specificity']
    ]

    bars1 = ax1.bar(x - width/2, values_model1, width, label=model_names[0], alpha=0.8)
    bars2 = ax1.bar(x + width/2, values_model2, width, label=model_names[1], alpha=0.8)

    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Confusion Matrix for Model 1
    ax2 = axes[0, 1]
    cm1 = results_dict[model_names[0]]['confusion_matrix']
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                annot_kws={'size': 14})
    ax2.set_title(f'Confusion Matrix: {model_names[0]}\nAcc={results_dict[model_names[0]]["accuracy"]:.4f}, F1={results_dict[model_names[0]]["f1"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)

    # Plot 3: Confusion Matrix for Model 2
    ax3 = axes[1, 0]
    cm2 = results_dict[model_names[1]]['confusion_matrix']
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax3,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                annot_kws={'size': 14})
    ax3.set_title(f'Confusion Matrix: {model_names[1]}\nAcc={results_dict[model_names[1]]["accuracy"]:.4f}, F1={results_dict[model_names[1]]["f1"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=11)
    ax3.set_xlabel('Predicted Label', fontsize=11)

    # Plot 4: Filtering Statistics Comparison
    ax4 = axes[1, 1]
    categories = ['Stage 1\nPass Rate', 'Stage 2\nPass Rate', 'Classified\nas Real']

    stats_model1 = [
        results_dict[model_names[0]]['stage1_pass_rate'],
        results_dict[model_names[0]]['stage2_pass_rate'],
        np.sum(results_dict[model_names[0]]['predictions'] == 0) / len(results_dict[model_names[0]]['predictions'])
    ]

    stats_model2 = [
        results_dict[model_names[1]]['stage1_pass_rate'],
        results_dict[model_names[1]]['stage2_pass_rate'],
        np.sum(results_dict[model_names[1]]['predictions'] == 0) / len(results_dict[model_names[1]]['predictions'])
    ]

    x2 = np.arange(len(categories))
    bars3 = ax4.bar(x2 - width/2, stats_model1, width, label=model_names[0], alpha=0.8)
    bars4 = ax4.bar(x2 + width/2, stats_model2, width, label=model_names[1], alpha=0.8)

    ax4.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax4.set_title('Filtering Statistics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x2)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.0)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'two_models_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {os.path.join(output_dir, 'two_models_comparison.png')}")


# ========================================
# Main
# ========================================

def main():
    print("="*80)
    print("Evaluating Two Models on Final Test Set")
    print("="*80)
    print("Method: Two-Stage Gaussian Filtering with PCA_10D")
    print(f"Configuration: Stage1_Conf={CONFIDENCE_STAGE1}, Stage2_Conf={CONFIDENCE_STAGE2}")
    print("="*80)
    print("\nModels to evaluate:")
    for name, path in MODELS.items():
        print(f"  - {name}: {path}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
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

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Get test labels (will be same for both models)
    test_labels = []
    for _, _, _, labels in test_loader:
        test_labels.extend(labels.numpy())
    test_labels = np.array(test_labels)

    print(f"  Real samples: {np.sum(test_labels == 0)}")
    print(f"  Fake samples: {np.sum(test_labels == 1)}")

    # Evaluate both models
    results_dict = {}

    for model_name, model_path in MODELS.items():
        results = evaluate_model(
            model_name, model_path,
            train_loader, test_loader, test_labels,
            device
        )
        results_dict[model_name] = results

    # ========================================
    # Summary and Comparison
    # ========================================
    print("\n\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    print("\n{:<25} {:<20} {:<20}".format("Metric", "method1_margin", "method2_discriminator"))
    print("-"*80)
    print("{:<25} {:<20.4f} {:<20.4f}".format("Accuracy",
        results_dict['method1_margin']['accuracy'],
        results_dict['method2_discriminator']['accuracy']))
    print("{:<25} {:<20.4f} {:<20.4f}".format("Precision",
        results_dict['method1_margin']['precision'],
        results_dict['method2_discriminator']['precision']))
    print("{:<25} {:<20.4f} {:<20.4f}".format("Recall",
        results_dict['method1_margin']['recall'],
        results_dict['method2_discriminator']['recall']))
    print("{:<25} {:<20.4f} {:<20.4f}".format("F1-Score",
        results_dict['method1_margin']['f1'],
        results_dict['method2_discriminator']['f1']))
    print("{:<25} {:<20.4f} {:<20.4f}".format("Specificity",
        results_dict['method1_margin']['specificity'],
        results_dict['method2_discriminator']['specificity']))
    print("{:<25} {:<20.4f} {:<20.4f}".format("Explained Variance",
        results_dict['method1_margin']['explained_variance'],
        results_dict['method2_discriminator']['explained_variance']))
    print("-"*80)

    # Determine best model
    if results_dict['method1_margin']['f1'] > results_dict['method2_discriminator']['f1']:
        best_model = 'method1_margin'
    else:
        best_model = 'method2_discriminator'

    print(f"\nBest Model (by F1-Score): {best_model}")
    print("="*80)

    # Save Results
    print("\nSaving results...")

    # Save individual results
    for model_name, results in results_dict.items():
        metrics = {
            'model_name': model_name,
            'model_checkpoint': MODELS[model_name],
            'method': 'Two-Stage Gaussian Filtering (PCA_10D)',
            'pca_dim': PCA_DIM,
            'explained_variance': float(results['explained_variance']),
            'confidence_stage1': CONFIDENCE_STAGE1,
            'confidence_stage2': CONFIDENCE_STAGE2,
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'specificity': float(results['specificity']),
            'stage1_pass_rate': float(results['stage1_pass_rate']),
            'stage2_pass_rate': float(results['stage2_pass_rate']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'tn': results['tn'],
            'fp': results['fp'],
            'fn': results['fn'],
            'tp': results['tp']
        }

        output_file = os.path.join(OUTPUT_DIR, f'{model_name}_results.json')
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {model_name} results to: {output_file}")

    # Save comparison summary
    comparison = {
        'method': 'Two-Stage Gaussian Filtering (PCA_10D)',
        'configuration': {
            'pca_dim': PCA_DIM,
            'confidence_stage1': CONFIDENCE_STAGE1,
            'confidence_stage2': CONFIDENCE_STAGE2
        },
        'test_set_size': len(test_labels),
        'test_set_real': int(np.sum(test_labels == 0)),
        'test_set_fake': int(np.sum(test_labels == 1)),
        'best_model': best_model,
        'models': {}
    }

    for model_name, results in results_dict.items():
        comparison['models'][model_name] = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'specificity': float(results['specificity']),
            'explained_variance': float(results['explained_variance']),
            'confusion_matrix': results['confusion_matrix'].tolist()
        }

    comparison_file = os.path.join(OUTPUT_DIR, 'comparison_summary.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison summary to: {comparison_file}")

    # Plot comparison
    plot_comparison(results_dict, OUTPUT_DIR)

    print()
    print("="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
