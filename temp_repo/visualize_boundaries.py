import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy import stats
import json
import os
from pathlib import Path

from config_bandvae import Config
from model_bandvae import BandSplitVAE
from dataset_stage2 import Stage2Dataset

SPLIT_JSON = "data_split.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best config
with open("visualizations/hyperparameter_tuning/best_config.json", "r") as f:
    best_config = json.load(f)

CONF_STAGE1 = best_config["conf_stage1"]  # 0.9
CONF_STAGE2 = best_config["conf_stage2"]  # 0.7

def mahalanobis_distance(x, mean, cov_inv):
    """Compute Mahalanobis distance"""
    diff = x - mean
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=-1))

def fit_gaussian(features):
    """Fit single Gaussian and return mean, covariance, inverse covariance"""
    mean = np.mean(features, axis=0)
    cov = np.cov(features.T)
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    return mean, cov, cov_inv

def fit_gmm(features, n_components=3):
    """Fit GMM and return model"""
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(features)
    return gmm

def gmm_min_mahalanobis(x, gmm):
    """Compute minimum Mahalanobis distance across all GMM components"""
    distances = []
    for k in range(gmm.n_components):
        mean = gmm.means_[k]
        cov = gmm.covariances_[k]
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        diff = x - mean
        dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=-1))
        distances.append(dist)
    return np.min(distances, axis=0)

def get_chi2_threshold(confidence, dim):
    """Get chi-squared threshold for given confidence and dimensions"""
    return np.sqrt(stats.chi2.ppf(confidence, dim))

def plot_decision_boundary(ax, X_2d, mean_2d, cov_2d, threshold, title):
    """Plot decision boundary as contour"""
    # Create grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Compute Mahalanobis distance for grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    cov_inv_2d = np.linalg.inv(cov_2d + 1e-6 * np.eye(2))
    distances = mahalanobis_distance(grid_points, mean_2d, cov_inv_2d)
    distances = distances.reshape(xx.shape)

    # Plot contour at threshold
    ax.contour(xx, yy, distances, levels=[threshold], colors='red', linewidths=2, linestyles='dashed')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

def plot_gmm_decision_boundary(ax, X_2d, gmm_2d, threshold, title):
    """Plot GMM decision boundary as contour"""
    # Create grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Compute minimum Mahalanobis distance for grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    distances = gmm_min_mahalanobis(grid_points, gmm_2d)
    distances = distances.reshape(xx.shape)

    # Plot contour at threshold
    ax.contour(xx, yy, distances, levels=[threshold], colors='red', linewidths=2, linestyles='dashed')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

def extract_features(model, dataloader, device):
    """Extract Stage 1 (reconstruction loss) and Stage 2 (latent) features"""
    model.eval()
    losses_lf, losses_bp, losses_hf = [], [], []
    latents = []
    labels_list = []

    with torch.no_grad():
        for x_lf, x_bp, x_hf, batch_labels in dataloader:
            x_lf = x_lf.to(device)
            x_bp = x_bp.to(device)
            x_hf = x_hf.to(device)

            # Forward pass
            recons, mus, _, _ = model(x_lf, x_bp, x_hf)

            # Compute reconstruction losses per sample
            for i in range(x_lf.size(0)):
                losses_lf.append(
                    torch.nn.functional.mse_loss(recons["lf"][i], x_lf[i]).item()
                )
                losses_bp.append(
                    torch.nn.functional.mse_loss(recons["bp"][i], x_bp[i]).item()
                )
                losses_hf.append(
                    torch.nn.functional.mse_loss(recons["hf"][i], x_hf[i]).item()
                )
                labels_list.append(batch_labels[i].item())

            # Extract latent vectors
            z_lf = mus["lf"].mean(dim=2)  # (B, C_z)
            z_bp = mus["bp"].mean(dim=2)  # (B, C_z)
            z_hf = mus["hf"].mean(dim=2)  # (B, C_z)
            latents.append(torch.cat([z_lf, z_bp, z_hf], dim=1).cpu().numpy())

    # Stage 1: Reconstruction losses (N, 3)
    stage1_features = np.stack([losses_lf, losses_bp, losses_hf], axis=1)

    # Stage 2: Latent vectors (N, 36)
    stage2_features = np.vstack(latents)

    # Labels
    labels = np.array(labels_list)

    return stage1_features, stage2_features, labels

def visualize_model(model_path, output_dir):
    """Visualize decision boundaries for a single model"""
    model_name = Path(model_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    # Load config and model
    config = Config(mode="full")
    checkpoint = torch.load(model_path, map_location=device)

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=checkpoint["config"].C_h,
        C_z=checkpoint["config"].C_z,
        dilations=checkpoint["config"].dilations,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test dataset
    test_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name="final_test",
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # Extract features
    print("Extracting features...")
    stage1_features, stage2_features, labels = extract_features(model, test_loader, device)

    # Filter real samples for training distributions
    real_mask = (labels == 0)
    stage1_real = stage1_features[real_mask]
    stage2_real = stage2_features[real_mask]

    print(f"Total samples: {len(labels)}, Real: {real_mask.sum()}, Fake: {(~real_mask).sum()}")

    # === STAGE 1: GMM on Reconstruction Loss ===
    print("\nStage 1: Fitting GMM on reconstruction loss...")
    gmm_stage1 = fit_gmm(stage1_real, n_components=3)
    threshold_stage1 = get_chi2_threshold(CONF_STAGE1, dim=3)

    # Predict Stage 1
    stage1_distances = gmm_min_mahalanobis(stage1_features, gmm_stage1)
    stage1_pred = (stage1_distances > threshold_stage1).astype(int)  # 0: Real, 1: Fake
    stage1_correct = (stage1_pred == labels)
    stage1_acc = stage1_correct.mean()

    print(f"Stage 1 Accuracy: {stage1_acc:.4f}")

    # t-SNE for Stage 1
    print("Running t-SNE for Stage 1...")
    tsne_stage1 = TSNE(n_components=2, random_state=42, perplexity=30)
    stage1_tsne = tsne_stage1.fit_transform(stage1_features)

    # Fit GMM on t-SNE space for boundary visualization
    gmm_stage1_2d = fit_gmm(stage1_tsne[real_mask], n_components=3)

    # Plot Stage 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stage 1 - Ground Truth
    ax = axes[0]
    for label, color, name in [(0, 'blue', 'Real'), (1, 'red', 'Fake')]:
        mask = labels == label
        ax.scatter(stage1_tsne[mask, 0], stage1_tsne[mask, 1],
                  c=color, label=name, alpha=0.6, s=20)
    plot_gmm_decision_boundary(ax, stage1_tsne, gmm_stage1_2d, threshold_stage1,
                               f'{model_name} - Stage 1 (GMM)\nGround Truth')
    ax.legend()

    # Stage 1 - Correctness
    ax = axes[1]
    for correct, color, name in [(True, 'green', 'Correct'), (False, 'orange', 'Wrong')]:
        mask = stage1_correct == correct
        ax.scatter(stage1_tsne[mask, 0], stage1_tsne[mask, 1],
                  c=color, label=name, alpha=0.6, s=20)
    plot_gmm_decision_boundary(ax, stage1_tsne, gmm_stage1_2d, threshold_stage1,
                               f'{model_name} - Stage 1 (GMM)\nPrediction Correctness (Acc: {stage1_acc:.2%})')
    ax.legend()

    plt.tight_layout()
    stage1_path = output_dir / f"{model_name}_stage1_gmm.png"
    plt.savefig(stage1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {stage1_path}")

    # === STAGE 2: Gaussian on Latent Space with PCA ===
    for pca_dim in [10, 20, 30]:
        print(f"\nStage 2: PCA {pca_dim}D + Gaussian...")

        # Apply PCA
        pca = PCA(n_components=pca_dim, random_state=42)
        stage2_pca = pca.fit_transform(stage2_features)
        stage2_pca_real = stage2_pca[real_mask]

        # Fit Gaussian
        mean_stage2, cov_stage2, cov_inv_stage2 = fit_gaussian(stage2_pca_real)
        threshold_stage2 = get_chi2_threshold(CONF_STAGE2, dim=pca_dim)

        # Predict Stage 2
        stage2_distances = mahalanobis_distance(stage2_pca, mean_stage2, cov_inv_stage2)
        stage2_pred = (stage2_distances > threshold_stage2).astype(int)
        stage2_correct = (stage2_pred == labels)
        stage2_acc = stage2_correct.mean()

        print(f"Stage 2 (PCA {pca_dim}) Accuracy: {stage2_acc:.4f}")

        # t-SNE for Stage 2
        print(f"Running t-SNE for Stage 2 (PCA {pca_dim})...")
        tsne_stage2 = TSNE(n_components=2, random_state=42, perplexity=30)
        stage2_tsne = tsne_stage2.fit_transform(stage2_pca)

        # Project Gaussian to t-SNE space (approximate)
        mean_2d = stage2_tsne[real_mask].mean(axis=0)
        cov_2d = np.cov(stage2_tsne[real_mask].T)

        # Plot Stage 2
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Stage 2 - Ground Truth
        ax = axes[0]
        for label, color, name in [(0, 'blue', 'Real'), (1, 'red', 'Fake')]:
            mask = labels == label
            ax.scatter(stage2_tsne[mask, 0], stage2_tsne[mask, 1],
                      c=color, label=name, alpha=0.6, s=20)
        plot_decision_boundary(ax, stage2_tsne, mean_2d, cov_2d, threshold_stage2,
                              f'{model_name} - Stage 2 (Gaussian, PCA {pca_dim})\nGround Truth')
        ax.legend()

        # Stage 2 - Correctness
        ax = axes[1]
        for correct, color, name in [(True, 'green', 'Correct'), (False, 'orange', 'Wrong')]:
            mask = stage2_correct == correct
            ax.scatter(stage2_tsne[mask, 0], stage2_tsne[mask, 1],
                      c=color, label=name, alpha=0.6, s=20)
        plot_decision_boundary(ax, stage2_tsne, mean_2d, cov_2d, threshold_stage2,
                              f'{model_name} - Stage 2 (Gaussian, PCA {pca_dim})\nPrediction Correctness (Acc: {stage2_acc:.2%})')
        ax.legend()

        plt.tight_layout()
        stage2_path = output_dir / f"{model_name}_stage2_gaussian_pca{pca_dim}.png"
        plt.savefig(stage2_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {stage2_path}")

def main():
    output_dir = Path("visualizations/decision_boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        "runs/stage1_pretrained.pt",
        "runs/method1_margin_best.pt",
        "runs/method2_discriminator_best.pt",
    ]

    for model_path in models:
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping...")
            continue
        visualize_model(model_path, output_dir)

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
