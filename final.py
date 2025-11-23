"""
Evaluate Final Method: Two-Stage Gaussian Filtering with PCA_10D

Stage 1: 3D Gaussian on band-wise reconstruction losses (lf, bp, hf)
Stage 2: PCA-reduced 10D latent space

Final Configuration:
- PCA dimensions: 10
- Stage 1 confidence: 0.95
- Stage 2 confidence: 0.70
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from config_bandvae import FullFeatureConfig
from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE, band_split_vae_loss

# ========================================
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"
MODEL_CHECKPOINT = "runs/stage2_method1_margin_fixed/stage2_method1_best.pt"
OUTPUT_DIR = "runs/final_evaluation_pca10d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Final hyperparameters
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
            loss_lf = nn.functional.mse_loss(recons["lf"][i], x_lf[i])
            all_losses_lf.append(loss_lf.item())

            # BP band reconstruction loss
            loss_bp = nn.functional.mse_loss(recons["bp"][i], x_bp[i])
            all_losses_bp.append(loss_bp.item())

            # HF band reconstruction loss
            loss_hf = nn.functional.mse_loss(recons["hf"][i], x_hf[i])
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
        z_lf = mus["lf"].mean(dim=2)  # (B, C_z)
        z_bp = mus["bp"].mean(dim=2)  # (B, C_z)
        z_hf = mus["hf"].mean(dim=2)  # (B, C_z)

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


def two_stage_filtering_pca10d(
    test_losses_3d,
    test_latents_pca,
    test_labels,
    loss_mu_3d,
    loss_cov_3d,
    latent_mu_pca,
    latent_cov_pca,
    confidence_stage1,
    confidence_stage2,
):
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
        test_labels, predictions, average="binary"
    )

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "predictions": predictions,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "stage1_pass_count": stage1_pass_count,
        "stage2_pass_count": stage2_pass_count,
        "stage1_pass_rate": stage1_pass_count / num_test,
        "stage2_pass_rate": stage2_pass_count / num_test,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ========================================
# Visualization
# ========================================


def plot_confusion_matrix(cm, accuracy, f1, output_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))

    # Normalize by row (true labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        cbar_kws={"label": "Count"},
        annot_kws={"size": 16},
    )

    title = "Confusion Matrix\n"
    title += f"Two-Stage Gaussian Filtering (PCA_10D)\n"
    title += f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}"

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")

    # Add percentages as text
    for i in range(2):
        for j in range(2):
            percentage = cm_normalized[i, j] * 100
            plt.text(
                j + 0.5,
                i + 0.75,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def plot_detailed_metrics(results, output_path):
    """Plot detailed metrics comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Bar chart of all metrics
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
    values = [
        results["accuracy"],
        results["precision"],
        results["recall"],
        results["f1"],
        results["specificity"],
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = axes[0].bar(
        metrics, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )

    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Performance Metrics\n(PCA_10D, Conf1=0.95, Conf2=0.70)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Plot 2: Filtering statistics
    stage1_pass = results["stage1_pass_rate"]
    stage2_pass = results["stage2_pass_rate"]
    classified_real = np.sum(results["predictions"] == 0) / len(results["predictions"])
    classified_fake = np.sum(results["predictions"] == 1) / len(results["predictions"])

    categories = [
        "Stage 1\nPass Rate",
        "Stage 2\nPass Rate",
        "Classified\nas Real",
        "Classified\nas Fake",
    ]
    stats_values = [stage1_pass, stage2_pass, classified_real, classified_fake]
    stats_colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]

    bars2 = axes[1].bar(
        categories,
        stats_values,
        color=stats_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Rate", fontsize=12, fontweight="bold")
    axes[1].set_title("Filtering Statistics", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars2, stats_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved detailed metrics to: {output_path}")


# ========================================
# Main
# ========================================


def main():
    print("=" * 80)
    print("Final Evaluation: Two-Stage Gaussian Filtering with PCA_10D")
    print("=" * 80)
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Configuration:")
    print(f"  PCA dimensions: {PCA_DIM}")
    print(f"  Stage 1 confidence: {CONFIDENCE_STAGE1}")
    print(f"  Stage 2 confidence: {CONFIDENCE_STAGE2}")
    print("=" * 80)
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
        split_name="stage2_train",
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=2.0,
        fc_high=8.0,
        filter_order=4,
        random_crop=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    test_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name="final_test",
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=2.0,
        fc_high=8.0,
        filter_order=4,
        random_crop=False,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
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
        C_h=checkpoint["config"].C_h,
        C_z=checkpoint["config"].C_z,
        dilations=checkpoint["config"].dilations,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print()

    # ========================================
    # Extract Features
    # ========================================
    print("Extracting features from training data...")

    train_losses_3d, train_labels = compute_bandwise_reconstruction_losses(
        model, train_loader, device
    )
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

    train_real_mask = train_labels == 0
    train_real_losses_3d = train_losses_3d[train_real_mask]
    train_real_latents_36d = train_latents_36d[train_real_mask]

    loss_mu_3d, loss_cov_3d = fit_gaussian_multivariate(train_real_losses_3d)

    print(f"Number of real training samples: {len(train_real_losses_3d)}")
    print(f"3D Loss mean: {loss_mu_3d}")
    print(f"3D Loss covariance shape: {loss_cov_3d.shape}")
    print()

    # ========================================
    # Apply PCA and Fit Stage 2 Gaussian
    # ========================================
    print(f"Applying PCA to reduce latent space to {PCA_DIM}D...")

    train_latents_pca, test_latents_pca, explained_var, pca_model = apply_pca(
        train_real_latents_36d, test_latents_36d, PCA_DIM
    )

    print(f"PCA explained variance: {explained_var:.4f}")
    print(
        f"PCA latent shapes: train={train_latents_pca.shape}, test={test_latents_pca.shape}"
    )
    print()

    print("Fitting Stage 2 Gaussian (PCA-reduced latent space)...")
    latent_mu_pca, latent_cov_pca = fit_gaussian_multivariate(train_latents_pca)
    print(f"PCA latent mean (first 5): {latent_mu_pca[:5]}")
    print()

    # ========================================
    # Apply Two-Stage Filtering
    # ========================================
    print("Applying two-stage filtering on test data...")

    results = two_stage_filtering_pca10d(
        test_losses_3d,
        test_latents_pca,
        test_labels,
        loss_mu_3d,
        loss_cov_3d,
        latent_mu_pca,
        latent_cov_pca,
        CONFIDENCE_STAGE1,
        CONFIDENCE_STAGE2,
    )

    # ========================================
    # Print Results
    # ========================================
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {MODEL_CHECKPOINT}")
    print(f"Method: Two-Stage Gaussian Filtering (PCA_10D)")
    print(
        f"Configuration: Stage1_Conf={CONFIDENCE_STAGE1}, Stage2_Conf={CONFIDENCE_STAGE2}"
    )
    print("=" * 80)
    print()
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"F1-Score:    {results['f1']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    print()
    print("Filtering Statistics:")
    print(
        f"  Stage 1 pass rate: {results['stage1_pass_rate']:.2%} ({results['stage1_pass_count']}/{len(test_labels)})"
    )
    print(
        f"  Stage 2 pass rate: {results['stage2_pass_rate']:.2%} ({results['stage2_pass_count']}/{len(test_labels)})"
    )
    print(
        f"  Classified as Real: {np.sum(results['predictions'] == 0)}/{len(test_labels)} ({100 * np.sum(results['predictions'] == 0) / len(test_labels):.2f}%)"
    )
    print(
        f"  Classified as Fake: {np.sum(results['predictions'] == 1)}/{len(test_labels)} ({100 * np.sum(results['predictions'] == 1) / len(test_labels):.2f}%)"
    )
    print()
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {results['tn']:4d}    {results['fp']:4d}")
    print(f"       Fake   {results['fn']:4d}    {results['tp']:4d}")
    print()
    print("=" * 80)

    # ========================================
    # Save Results
    # ========================================
    print("\nSaving results...")

    # Save metrics as JSON
    metrics = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "method": "PCA_10D",
        "pca_dim": PCA_DIM,
        "explained_variance": float(explained_var),
        "confidence_stage1": CONFIDENCE_STAGE1,
        "confidence_stage2": CONFIDENCE_STAGE2,
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1": float(results["f1"]),
        "specificity": float(results["specificity"]),
        "stage1_pass_rate": float(results["stage1_pass_rate"]),
        "stage2_pass_rate": float(results["stage2_pass_rate"]),
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")

    # Plot confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"],
        results["accuracy"],
        results["f1"],
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )

    # Plot detailed metrics
    plot_detailed_metrics(results, os.path.join(OUTPUT_DIR, "detailed_metrics.png"))

    print()
    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
