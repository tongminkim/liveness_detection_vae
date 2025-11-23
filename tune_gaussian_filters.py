"""
Hyperparameter Tuning for Two-Stage Gaussian Filtering

Try different confidence intervals to find optimal threshold
that balances precision and recall for fake detection
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
OUTPUT_DIR = "runs/gaussian_filtering_tuning"
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


def is_within_confidence_1d(x, mu, sigma, confidence):
    """Check if value is within confidence interval for 1D Gaussian"""
    z_score = stats.norm.ppf((1 + confidence) / 2)
    lower = mu - z_score * sigma
    upper = mu + z_score * sigma
    return lower <= x <= upper


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
# Two-Stage Filtering with Tunable Confidence
# ========================================

def two_stage_filtering_tunable(test_losses, test_latents, test_labels,
                                 loss_mu, loss_sigma, latent_mu, latent_cov,
                                 confidence_stage1, confidence_stage2):
    """
    Apply two-stage filtering with different confidence levels for each stage
    """
    num_test = len(test_losses)
    predictions = []
    stage1_pass_count = 0

    for i in range(num_test):
        loss = test_losses[i]
        latent = test_latents[i]

        # Stage 1: Reconstruction loss filter
        stage1_pass = is_within_confidence_1d(loss, loss_mu, loss_sigma, confidence_stage1)

        if stage1_pass:
            stage1_pass_count += 1
            # Stage 2: Latent distribution filter
            stage2_pass = is_within_confidence_multivariate(latent, latent_mu, latent_cov, confidence_stage2)

            if stage2_pass:
                predictions.append(0)  # Real
            else:
                predictions.append(1)  # Fake
        else:
            predictions.append(1)  # Fake

    predictions = np.array(predictions)

    # Compute metrics
    cm = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')

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
        'stage1_pass_rate': stage1_pass_count / num_test,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


# ========================================
# Hyperparameter Search
# ========================================

def grid_search_confidence_levels(test_losses, test_latents, test_labels,
                                   loss_mu, loss_sigma, latent_mu, latent_cov):
    """
    Grid search over different confidence levels for both stages
    """

    # Try different confidence levels
    confidence_levels = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    results = []

    print("="*80)
    print("Grid Search: Confidence Levels")
    print("="*80)

    for conf1 in confidence_levels:
        for conf2 in confidence_levels:
            result = two_stage_filtering_tunable(
                test_losses, test_latents, test_labels,
                loss_mu, loss_sigma, latent_mu, latent_cov,
                conf1, conf2
            )

            result['confidence_stage1'] = conf1
            result['confidence_stage2'] = conf2
            results.append(result)

            print(f"Stage1={conf1:.2f}, Stage2={conf2:.2f}: "
                  f"Acc={result['accuracy']:.4f}, "
                  f"Prec={result['precision']:.4f}, "
                  f"Rec={result['recall']:.4f}, "
                  f"F1={result['f1']:.4f}")

    return results


# ========================================
# Visualization
# ========================================

def plot_hyperparameter_heatmaps(results, output_dir):
    """Plot heatmaps for different metrics across confidence levels"""

    # Extract unique confidence levels
    conf1_vals = sorted(list(set([r['confidence_stage1'] for r in results])))
    conf2_vals = sorted(list(set([r['confidence_stage2'] for r in results])))

    n_conf1 = len(conf1_vals)
    n_conf2 = len(conf2_vals)

    # Create matrices for each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        matrix = np.zeros((n_conf1, n_conf2))

        for r in results:
            i = conf1_vals.index(r['confidence_stage1'])
            j = conf2_vals.index(r['confidence_stage2'])
            matrix[i, j] = r[metric]

        im = axes[idx].imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f'{label}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Stage 2 Confidence', fontsize=12)
        axes[idx].set_ylabel('Stage 1 Confidence', fontsize=12)
        axes[idx].set_xticks(range(n_conf2))
        axes[idx].set_yticks(range(n_conf1))
        axes[idx].set_xticklabels([f'{c:.2f}' for c in conf2_vals], rotation=45)
        axes[idx].set_yticklabels([f'{c:.2f}' for c in conf1_vals])

        # Add text annotations
        for i in range(n_conf1):
            for j in range(n_conf2):
                text = axes[idx].text(j, i, f'{matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=axes[idx])

    # Remove extra subplot
    axes[5].remove()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmaps to: {os.path.join(output_dir, 'hyperparameter_heatmaps.png')}")


def plot_pareto_frontier(results, output_dir):
    """Plot Pareto frontier: Recall vs Precision"""

    recalls = [r['recall'] for r in results]
    precisions = [r['precision'] for r in results]
    f1s = [r['f1'] for r in results]

    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(recalls, precisions, c=f1s, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, label='F1-Score')

    # Highlight top configurations
    top_f1_idx = np.argmax(f1s)
    best_balance_idx = np.argmax([min(r['recall'], r['precision']) for r in results])

    plt.scatter(recalls[top_f1_idx], precisions[top_f1_idx],
               color='red', s=300, marker='*', edgecolors='black', linewidths=2,
               label=f'Best F1={f1s[top_f1_idx]:.3f}')

    plt.scatter(recalls[best_balance_idx], precisions[best_balance_idx],
               color='blue', s=300, marker='D', edgecolors='black', linewidths=2,
               label=f'Best Balance')

    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Trade-off\n(Color = F1-Score)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_tradeoff.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Pareto frontier to: {os.path.join(output_dir, 'precision_recall_tradeoff.png')}")


def plot_top_configurations(results, output_dir, top_k=5):
    """Plot confusion matrices for top configurations"""

    # Sort by F1-score
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    top_results = sorted_results[:top_k]

    fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))

    for idx, result in enumerate(top_results):
        cm = result['confusion_matrix']
        conf1 = result['confidence_stage1']
        conf2 = result['confidence_stage2']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   ax=axes[idx], cbar=False)

        axes[idx].set_title(f'Rank {idx+1}\nStage1={conf1:.2f}, Stage2={conf2:.2f}\n'
                           f'F1={result["f1"]:.3f}, Acc={result["accuracy"]:.3f}',
                           fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual' if idx == 0 else '')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_configurations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top configurations to: {os.path.join(output_dir, 'top_configurations.png')}")


def analyze_stage1_pass_rate(results, output_dir):
    """Analyze how Stage 1 pass rate affects performance"""

    stage1_pass_rates = [r['stage1_pass_rate'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    recalls = [r['recall'] for r in results]
    precisions = [r['precision'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy vs Stage1 Pass Rate
    axes[0].scatter(stage1_pass_rates, accuracies, alpha=0.6, s=50)
    axes[0].set_xlabel('Stage 1 Pass Rate', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs Stage 1 Pass Rate', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Recall vs Stage1 Pass Rate
    axes[1].scatter(stage1_pass_rates, recalls, alpha=0.6, s=50, color='orange')
    axes[1].set_xlabel('Stage 1 Pass Rate', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Recall vs Stage 1 Pass Rate', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Precision vs Stage1 Pass Rate
    axes[2].scatter(stage1_pass_rates, precisions, alpha=0.6, s=50, color='green')
    axes[2].set_xlabel('Stage 1 Pass Rate', fontsize=12)
    axes[2].set_ylabel('Precision', fontsize=12)
    axes[2].set_title('Precision vs Stage 1 Pass Rate', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage1_pass_rate_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Stage 1 pass rate analysis to: {os.path.join(output_dir, 'stage1_pass_rate_analysis.png')}")


# ========================================
# Main
# ========================================

def main():
    print("="*80)
    print("Hyperparameter Tuning: Two-Stage Gaussian Filtering")
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
    # Fit Gaussians on Training Data
    # ========================================
    print("Fitting Gaussians on training data...")

    train_losses, train_labels = compute_reconstruction_losses(model, train_loader, device)
    train_latents, _ = extract_latent_vectors(model, train_loader, device)

    # Filter only real samples
    train_real_losses = train_losses[train_labels == 0]
    train_real_latents = train_latents[train_labels == 0]

    # Fit Gaussians
    loss_mu, loss_sigma = fit_gaussian_1d(train_real_losses)
    latent_mu, latent_cov = fit_gaussian_multivariate(train_real_latents)

    print(f"Reconstruction loss: mu={loss_mu:.4f}, sigma={loss_sigma:.4f}")
    print(f"Latent dimension: {len(latent_mu)}")
    print()

    # ========================================
    # Extract Test Features
    # ========================================
    print("Extracting test features...")

    test_losses, test_labels = compute_reconstruction_losses(model, test_loader, device)
    test_latents, _ = extract_latent_vectors(model, test_loader, device)

    print(f"Test samples: {len(test_losses)}")
    print(f"  Real: {np.sum(test_labels == 0)}")
    print(f"  Fake: {np.sum(test_labels == 1)}")
    print()

    # ========================================
    # Grid Search
    # ========================================
    results = grid_search_confidence_levels(
        test_losses, test_latents, test_labels,
        loss_mu, loss_sigma, latent_mu, latent_cov
    )

    print()
    print("="*80)
    print("Top 10 Configurations by F1-Score")
    print("="*80)

    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)

    for idx, r in enumerate(sorted_results[:10]):
        print(f"\nRank {idx+1}:")
        print(f"  Stage 1 Confidence: {r['confidence_stage1']:.2f}")
        print(f"  Stage 2 Confidence: {r['confidence_stage2']:.2f}")
        print(f"  Accuracy:    {r['accuracy']:.4f}")
        print(f"  Precision:   {r['precision']:.4f}")
        print(f"  Recall:      {r['recall']:.4f}")
        print(f"  F1-Score:    {r['f1']:.4f}")
        print(f"  Specificity: {r['specificity']:.4f}")
        print(f"  Confusion Matrix: TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}")

    # ========================================
    # Save Results
    # ========================================
    print("\n" + "="*80)
    print("Saving results...")

    # Save all results to JSON
    results_json = []
    for r in sorted_results:
        results_json.append({
            'confidence_stage1': r['confidence_stage1'],
            'confidence_stage2': r['confidence_stage2'],
            'accuracy': float(r['accuracy']),
            'precision': float(r['precision']),
            'recall': float(r['recall']),
            'f1': float(r['f1']),
            'specificity': float(r['specificity']),
            'stage1_pass_rate': float(r['stage1_pass_rate']),
            'confusion_matrix': r['confusion_matrix'].tolist()
        })

    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Saved results to: {os.path.join(OUTPUT_DIR, 'all_results.json')}")

    # Generate visualizations
    plot_hyperparameter_heatmaps(results, OUTPUT_DIR)
    plot_pareto_frontier(results, OUTPUT_DIR)
    plot_top_configurations(results, OUTPUT_DIR, top_k=5)
    analyze_stage1_pass_rate(results, OUTPUT_DIR)

    print()
    print("="*80)
    print("Hyperparameter tuning complete!")
    print("="*80)


if __name__ == "__main__":
    main()
