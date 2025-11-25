"""
Advanced Hyperparameter Tuning for Two-Stage Gaussian Filtering

Stage 1: 3D Gaussian on band-wise reconstruction losses (lf, bp, hf)
Stage 2: Multiple approaches
  - Full 36D latent space
  - PCA-reduced latent space (multiple dimensions)
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
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"
MODEL_CHECKPOINT = "runs/stage2_method1_margin/stage2_method1_best.pt"
OUTPUT_DIR = "runs/gaussian_filtering_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def two_stage_filtering_3d(test_losses_3d, test_latents, test_labels,
                            loss_mu_3d, loss_cov_3d,
                            latent_mu, latent_cov,
                            confidence_stage1, confidence_stage2,
                            method_name="Full36D"):
    """
    Two-stage filtering with 3D Gaussian on band-wise losses

    Args:
        method_name: "Full36D" or "PCA_XX"
    """
    num_test = len(test_losses_3d)
    predictions = []
    stage1_pass_count = 0

    for i in range(num_test):
        loss_vec = test_losses_3d[i]  # (3,)
        latent = test_latents[i]

        # Stage 1: 3D Gaussian on band-wise reconstruction losses
        stage1_pass = is_within_confidence_multivariate(
            loss_vec, loss_mu_3d, loss_cov_3d, confidence_stage1
        )

        if stage1_pass:
            stage1_pass_count += 1
            # Stage 2: Latent distribution filter
            stage2_pass = is_within_confidence_multivariate(
                latent, latent_mu, latent_cov, confidence_stage2
            )

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
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary'
    )

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'method': method_name,
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

def comprehensive_grid_search(test_losses_3d, test_latents_36d, test_labels,
                               loss_mu_3d, loss_cov_3d,
                               train_latents_36d, train_labels):
    """
    Comprehensive grid search over:
    1. Confidence levels
    2. Latent representation methods (Full 36D, PCA with various dimensions)
    """

    # Confidence levels to try
    confidence_levels = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # PCA dimensions to try
    pca_dims = [5, 10, 15, 20, 25, 30]

    # Filter real samples for Gaussian fitting
    train_real_mask = (train_labels == 0)
    train_real_latents_36d = train_latents_36d[train_real_mask]

    results = []

    print("="*80)
    print("Comprehensive Grid Search")
    print("="*80)

    # ========================================
    # Method 1: Full 36D Latent
    # ========================================
    print("\n[1/7] Testing Full 36D Latent Space...")

    latent_mu_36d, latent_cov_36d = fit_gaussian_multivariate(train_real_latents_36d)

    for conf1 in confidence_levels:
        for conf2 in confidence_levels:
            result = two_stage_filtering_3d(
                test_losses_3d, test_latents_36d, test_labels,
                loss_mu_3d, loss_cov_3d,
                latent_mu_36d, latent_cov_36d,
                conf1, conf2,
                method_name="Full36D"
            )
            result['confidence_stage1'] = conf1
            result['confidence_stage2'] = conf2
            result['latent_dim'] = 36
            result['explained_variance'] = 1.0

            results.append(result)

            if conf1 == 0.70 and conf2 in [0.50, 0.70, 0.90]:
                print(f"  Conf1={conf1:.2f}, Conf2={conf2:.2f}: "
                      f"Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}")

    # ========================================
    # Method 2: PCA-reduced Latent
    # ========================================
    for idx, n_comp in enumerate(pca_dims):
        print(f"\n[{idx+2}/7] Testing PCA {n_comp}D Latent Space...")

        # Apply PCA
        train_pca, test_pca, expl_var, pca_model = apply_pca(
            train_real_latents_36d, test_latents_36d, n_comp
        )

        print(f"  Explained variance: {expl_var:.4f}")

        # Fit Gaussian on PCA-reduced space
        latent_mu_pca, latent_cov_pca = fit_gaussian_multivariate(train_pca)

        for conf1 in confidence_levels:
            for conf2 in confidence_levels:
                result = two_stage_filtering_3d(
                    test_losses_3d, test_pca, test_labels,
                    loss_mu_3d, loss_cov_3d,
                    latent_mu_pca, latent_cov_pca,
                    conf1, conf2,
                    method_name=f"PCA_{n_comp}D"
                )
                result['confidence_stage1'] = conf1
                result['confidence_stage2'] = conf2
                result['latent_dim'] = n_comp
                result['explained_variance'] = float(expl_var)

                results.append(result)

                if conf1 == 0.70 and conf2 in [0.50, 0.70, 0.90]:
                    print(f"  Conf1={conf1:.2f}, Conf2={conf2:.2f}: "
                          f"Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}")

    return results


# ========================================
# Visualization
# ========================================

def plot_method_comparison(results, output_dir):
    """Compare different methods (Full36D vs PCA variants)"""

    methods = sorted(list(set([r['method'] for r in results])))

    # For each method, find best F1 score
    method_best = {}
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        best = max(method_results, key=lambda x: x['f1'])
        method_best[method] = best

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        # Extract data
        method_names = []
        values = []
        colors = []

        for method in methods:
            method_names.append(method)
            values.append(method_best[method][metric])

            if method == 'Full36D':
                colors.append('steelblue')
            else:
                colors.append('coral')

        # Bar plot
        bars = ax.bar(range(len(method_names)), values, color=colors)
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'Best {title} by Method', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved method comparison to: {os.path.join(output_dir, 'method_comparison.png')}")


def plot_pca_analysis(results, output_dir):
    """Analyze PCA performance vs dimensions"""

    pca_results = [r for r in results if 'PCA' in r['method']]

    # Group by PCA dimension and find best for each
    pca_dims = sorted(list(set([r['latent_dim'] for r in pca_results])))

    best_by_dim = {}
    for dim in pca_dims:
        dim_results = [r for r in pca_results if r['latent_dim'] == dim]
        best = max(dim_results, key=lambda x: x['f1'])
        best_by_dim[dim] = best

    # Also include Full36D for comparison
    full_results = [r for r in results if r['method'] == 'Full36D']
    best_full = max(full_results, key=lambda x: x['f1'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: F1-Score vs PCA Dimension
    dims = list(pca_dims) + [36]
    f1_scores = [best_by_dim[d]['f1'] for d in pca_dims] + [best_full['f1']]
    explained_vars = [best_by_dim[d]['explained_variance'] for d in pca_dims] + [1.0]

    axes[0].plot(dims, f1_scores, 'o-', linewidth=2, markersize=10, color='darkblue')
    axes[0].axhline(y=best_full['f1'], color='red', linestyle='--',
                   label=f'Full 36D: {best_full["f1"]:.4f}')
    axes[0].set_xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Best F1-Score', fontsize=12, fontweight='bold')
    axes[0].set_title('F1-Score vs Latent Dimension', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Annotate points
    for dim, f1 in zip(dims, f1_scores):
        axes[0].annotate(f'{f1:.3f}', (dim, f1), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Explained Variance vs PCA Dimension
    axes[1].plot(pca_dims, [best_by_dim[d]['explained_variance'] for d in pca_dims],
                'o-', linewidth=2, markersize=10, color='darkgreen')
    axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95% Variance')
    axes[1].set_xlabel('PCA Dimension', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    axes[1].set_title('Explained Variance vs PCA Dimension', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 1.05)

    # Annotate points
    for dim, var in zip(pca_dims, [best_by_dim[d]['explained_variance'] for d in pca_dims]):
        axes[1].annotate(f'{var:.3f}', (dim, var), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA analysis to: {os.path.join(output_dir, 'pca_analysis.png')}")


def plot_top_configs_per_method(results, output_dir, top_k=3):
    """Plot confusion matrices for top configs of each method"""

    methods = ['Full36D', 'PCA_5D', 'PCA_10D', 'PCA_15D', 'PCA_20D', 'PCA_25D', 'PCA_30D']

    fig, axes = plt.subplots(len(methods), top_k, figsize=(4*top_k, 4*len(methods)))

    for row_idx, method in enumerate(methods):
        method_results = [r for r in results if r['method'] == method]
        sorted_results = sorted(method_results, key=lambda x: x['f1'], reverse=True)[:top_k]

        for col_idx, result in enumerate(sorted_results):
            ax = axes[row_idx, col_idx]
            cm = result['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Real', 'Fake'],
                       yticklabels=['Real', 'Fake'],
                       ax=ax, cbar=False)

            title = f'{method} (Rank {col_idx+1})\n'
            title += f'C1={result["confidence_stage1"]:.2f}, C2={result["confidence_stage2"]:.2f}\n'
            title += f'F1={result["f1"]:.3f}, Acc={result["accuracy"]:.3f}'

            ax.set_title(title, fontsize=9, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel('Actual', fontsize=10)
            else:
                ax.set_ylabel('')

            if row_idx == len(methods) - 1:
                ax.set_xlabel('Predicted', fontsize=10)
            else:
                ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_configs_per_method.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top configs to: {os.path.join(output_dir, 'top_configs_per_method.png')}")


# ========================================
# Main
# ========================================

def main():
    print("="*80)
    print("Advanced Hyperparameter Tuning")
    print("Stage 1: 3D Gaussian (Band-wise Reconstruction Loss)")
    print("Stage 2: Full 36D vs PCA-reduced Latent Space")
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
    # Extract Features
    # ========================================
    print("Extracting features from training data...")

    # Band-wise reconstruction losses (3D)
    train_losses_3d, train_labels = compute_bandwise_reconstruction_losses(
        model, train_loader, device
    )

    # Full 36D latent vectors
    train_latents_36d, _ = extract_latent_vectors(model, train_loader, device)

    print(f"Training features extracted:")
    print(f"  3D losses shape: {train_losses_3d.shape}")
    print(f"  36D latents shape: {train_latents_36d.shape}")
    print()

    print("Extracting features from test data...")

    test_losses_3d, test_labels = compute_bandwise_reconstruction_losses(
        model, test_loader, device
    )

    test_latents_36d, _ = extract_latent_vectors(model, test_loader, device)

    print(f"Test features extracted:")
    print(f"  3D losses shape: {test_losses_3d.shape}")
    print(f"  36D latents shape: {test_latents_36d.shape}")
    print()

    # ========================================
    # Fit Stage 1 Gaussian (3D on losses)
    # ========================================
    print("Fitting Stage 1 Gaussian (3D band-wise losses)...")

    train_real_mask = (train_labels == 0)
    train_real_losses_3d = train_losses_3d[train_real_mask]

    loss_mu_3d, loss_cov_3d = fit_gaussian_multivariate(train_real_losses_3d)

    print(f"3D Loss mean: {loss_mu_3d}")
    print(f"3D Loss covariance shape: {loss_cov_3d.shape}")
    print()

    # ========================================
    # Comprehensive Grid Search
    # ========================================
    results = comprehensive_grid_search(
        test_losses_3d, test_latents_36d, test_labels,
        loss_mu_3d, loss_cov_3d,
        train_latents_36d, train_labels
    )

    # ========================================
    # Analyze Results
    # ========================================
    print("\n" + "="*80)
    print("Top 10 Overall Configurations by F1-Score")
    print("="*80)

    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)

    for idx, r in enumerate(sorted_results[:10]):
        print(f"\nRank {idx+1}:")
        print(f"  Method: {r['method']}")
        print(f"  Latent Dim: {r['latent_dim']}")
        print(f"  Explained Variance: {r['explained_variance']:.4f}")
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

    results_json = []
    for r in sorted_results:
        results_json.append({
            'method': r['method'],
            'latent_dim': r['latent_dim'],
            'explained_variance': float(r['explained_variance']),
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
    plot_method_comparison(results, OUTPUT_DIR)
    plot_pca_analysis(results, OUTPUT_DIR)
    plot_top_configs_per_method(results, OUTPUT_DIR, top_k=3)

    print()
    print("="*80)
    print("Advanced hyperparameter tuning complete!")
    print("="*80)


if __name__ == "__main__":
    main()
