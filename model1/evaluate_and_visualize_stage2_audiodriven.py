"""
Stage 2 Methods 평가 및 시각화 (Audio-Driven Only)
- Method 1 (Margin Loss) vs Method 2 (Discriminator)
- Final test set에서 audio-driven fake만 사용
- 5가지 시각화:
  1. Reconstruction Loss Distribution (Histogram)
  2. ROC Curve
  3. 2D Latent Space (t-SNE: LF vs HF)
  4. 3D Latent Space (t-SNE: LF, BP, HF)
  5. Score Distribution by Fake Type
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Subset

from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE
from train_stage2_method2_discriminator import LatentDiscriminator
from config_bandvae import FullFeatureConfig


# ========================================
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"

# Model checkpoints
METHOD1_CHECKPOINT = "runs/stage2_method1_margin_fixed/stage2_method1_best.pt"
METHOD2_CHECKPOINT = "runs/stage2_method2_discriminator/stage2_method2_best.pt"

# Output directory
OUTPUT_DIR = "runs/stage2_evaluation_audiodriven"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
config = FullFeatureConfig()
config.T_fixed = 300
config.fc_low = 2.0
config.fc_high = 8.0
config.filter_order = 4
config.C_h = 48
config.C_z = 12
config.dilations = [1, 2, 4]
config.batch_size = 32
config.num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Helper Functions
# ========================================

def filter_audiodriven_indices(dataset):
    """
    Filter indices to keep only audio-driven fake samples and all real samples
    """
    keep_indices = []

    for i in range(len(dataset)):
        file_path = dataset.files[i]
        label = dataset.labels[i]

        # Keep all real samples
        if label == 0:
            keep_indices.append(i)
        # Keep only audio-driven fake samples
        elif 'audio-driven' in file_path:
            keep_indices.append(i)

    return keep_indices


@torch.no_grad()
def compute_scores_method1(model, loader, device):
    """
    Method 1: Reconstruction loss as anomaly score
    - Real samples: low reconstruction loss
    - Fake samples: high reconstruction loss (ideally)
    """
    model.eval()

    all_scores = []
    all_labels = []
    all_lf_losses = []
    all_bp_losses = []
    all_hf_losses = []
    all_latent_lf = []
    all_latent_bp = []
    all_latent_hf = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, _, _ = model(x_lf, x_bp, x_hf)

        # Per-sample reconstruction loss
        loss_lf = torch.mean(torch.abs(recons['lf'] - x_lf), dim=[1, 2])
        loss_bp = torch.mean(torch.abs(recons['bp'] - x_bp), dim=[1, 2])
        loss_hf = torch.mean(torch.abs(recons['hf'] - x_hf), dim=[1, 2])
        total_loss = loss_lf + loss_bp + loss_hf

        # Collect
        all_scores.append(total_loss.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_lf_losses.append(loss_lf.cpu().numpy())
        all_bp_losses.append(loss_bp.cpu().numpy())
        all_hf_losses.append(loss_hf.cpu().numpy())

        # Latent vectors (use mean pooling over time)
        z_lf = mus['lf'].mean(dim=2) if len(mus['lf'].shape) == 3 else mus['lf']
        z_bp = mus['bp'].mean(dim=2) if len(mus['bp'].shape) == 3 else mus['bp']
        z_hf = mus['hf'].mean(dim=2) if len(mus['hf'].shape) == 3 else mus['hf']

        all_latent_lf.append(z_lf.cpu().numpy())
        all_latent_bp.append(z_bp.cpu().numpy())
        all_latent_hf.append(z_hf.cpu().numpy())

    return {
        'scores': np.concatenate(all_scores),
        'labels': np.concatenate(all_labels),
        'lf_losses': np.concatenate(all_lf_losses),
        'bp_losses': np.concatenate(all_bp_losses),
        'hf_losses': np.concatenate(all_hf_losses),
        'latent_lf': np.concatenate(all_latent_lf),
        'latent_bp': np.concatenate(all_latent_bp),
        'latent_hf': np.concatenate(all_latent_hf)
    }


@torch.no_grad()
def compute_scores_method2(model, discriminator, loader, device):
    """
    Method 2: Discriminator output as anomaly score
    - Real samples: low score (close to 0)
    - Fake samples: high score (close to 1)
    """
    model.eval()
    discriminator.eval()

    all_scores = []
    all_labels = []
    all_lf_losses = []
    all_bp_losses = []
    all_hf_losses = []
    all_latent_lf = []
    all_latent_bp = []
    all_latent_hf = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, _, _ = model(x_lf, x_bp, x_hf)

        # Discriminator score
        z_lf = mus['lf']
        z_bp = mus['bp']
        z_hf = mus['hf']

        logits = discriminator(z_lf, z_bp, z_hf).squeeze(1)
        scores = torch.sigmoid(logits)  # Convert to probability

        # Per-sample reconstruction loss (for additional analysis)
        loss_lf = torch.mean(torch.abs(recons['lf'] - x_lf), dim=[1, 2])
        loss_bp = torch.mean(torch.abs(recons['bp'] - x_bp), dim=[1, 2])
        loss_hf = torch.mean(torch.abs(recons['hf'] - x_hf), dim=[1, 2])

        # Collect
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_lf_losses.append(loss_lf.cpu().numpy())
        all_bp_losses.append(loss_bp.cpu().numpy())
        all_hf_losses.append(loss_hf.cpu().numpy())

        # Latent vectors (use mean pooling over time)
        z_lf_pooled = z_lf.mean(dim=2) if len(z_lf.shape) == 3 else z_lf
        z_bp_pooled = z_bp.mean(dim=2) if len(z_bp.shape) == 3 else z_bp
        z_hf_pooled = z_hf.mean(dim=2) if len(z_hf.shape) == 3 else z_hf

        all_latent_lf.append(z_lf_pooled.cpu().numpy())
        all_latent_bp.append(z_bp_pooled.cpu().numpy())
        all_latent_hf.append(z_hf_pooled.cpu().numpy())

    return {
        'scores': np.concatenate(all_scores),
        'labels': np.concatenate(all_labels),
        'lf_losses': np.concatenate(all_lf_losses),
        'bp_losses': np.concatenate(all_bp_losses),
        'hf_losses': np.concatenate(all_hf_losses),
        'latent_lf': np.concatenate(all_latent_lf),
        'latent_bp': np.concatenate(all_latent_bp),
        'latent_hf': np.concatenate(all_latent_hf)
    }


# ========================================
# Visualization Functions
# ========================================

def plot_1_reconstruction_loss_distribution(results_m1, results_m2, output_path):
    """
    Plot 1: Reconstruction Loss Distribution (Histogram)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Method 1
    ax = axes[0]
    real_mask_m1 = (results_m1['labels'] == 0)
    fake_mask_m1 = (results_m1['labels'] == 1)

    ax.hist(results_m1['scores'][real_mask_m1], bins=50, alpha=0.6, label='Real', color='blue', density=True)
    ax.hist(results_m1['scores'][fake_mask_m1], bins=50, alpha=0.6, label='Fake (Audio-Driven)', color='red', density=True)
    ax.set_xlabel('Reconstruction Loss')
    ax.set_ylabel('Density')
    ax.set_title('Method 1 (Margin Loss): Reconstruction Loss Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Method 2
    ax = axes[1]
    real_mask_m2 = (results_m2['labels'] == 0)
    fake_mask_m2 = (results_m2['labels'] == 1)

    ax.hist(results_m2['scores'][real_mask_m2], bins=50, alpha=0.6, label='Real', color='blue', density=True)
    ax.hist(results_m2['scores'][fake_mask_m2], bins=50, alpha=0.6, label='Fake (Audio-Driven)', color='red', density=True)
    ax.set_xlabel('Discriminator Score')
    ax.set_ylabel('Density')
    ax.set_title('Method 2 (Discriminator): Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_2_roc_curve(results_m1, results_m2, output_path):
    """
    Plot 2: ROC Curve
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Method 1: ROC curve
    fpr_m1, tpr_m1, _ = roc_curve(results_m1['labels'], results_m1['scores'])
    auc_m1 = auc(fpr_m1, tpr_m1)

    # Method 2: ROC curve
    fpr_m2, tpr_m2, _ = roc_curve(results_m2['labels'], results_m2['scores'])
    auc_m2 = auc(fpr_m2, tpr_m2)

    ax.plot(fpr_m1, tpr_m1, label=f'Method 1 (Margin Loss) - AUC={auc_m1:.4f}', linewidth=2)
    ax.plot(fpr_m2, tpr_m2, label=f'Method 2 (Discriminator) - AUC={auc_m2:.4f}', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison (Audio-Driven Only)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return auc_m1, auc_m2


def plot_3_latent_2d_tsne(results_m1, results_m2, output_path):
    """
    Plot 3: 2D Latent Space (t-SNE: LF vs HF)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Method 1
    ax = axes[0]
    latent_combined_m1 = np.concatenate([
        results_m1['latent_lf'],
        results_m1['latent_hf']
    ], axis=1)

    print("Computing t-SNE for Method 1 (2D)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d_m1 = tsne.fit_transform(latent_combined_m1)

    real_mask_m1 = (results_m1['labels'] == 0)
    fake_mask_m1 = (results_m1['labels'] == 1)

    ax.scatter(latent_2d_m1[real_mask_m1, 0], latent_2d_m1[real_mask_m1, 1],
               c='blue', alpha=0.5, s=10, label='Real')
    ax.scatter(latent_2d_m1[fake_mask_m1, 0], latent_2d_m1[fake_mask_m1, 1],
               c='red', alpha=0.5, s=10, label='Fake (Audio-Driven)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Method 1 (Margin Loss): 2D Latent Space (LF + HF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Method 2
    ax = axes[1]
    latent_combined_m2 = np.concatenate([
        results_m2['latent_lf'],
        results_m2['latent_hf']
    ], axis=1)

    print("Computing t-SNE for Method 2 (2D)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d_m2 = tsne.fit_transform(latent_combined_m2)

    real_mask_m2 = (results_m2['labels'] == 0)
    fake_mask_m2 = (results_m2['labels'] == 1)

    ax.scatter(latent_2d_m2[real_mask_m2, 0], latent_2d_m2[real_mask_m2, 1],
               c='blue', alpha=0.5, s=10, label='Real')
    ax.scatter(latent_2d_m2[fake_mask_m2, 0], latent_2d_m2[fake_mask_m2, 1],
               c='red', alpha=0.5, s=10, label='Fake (Audio-Driven)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Method 2 (Discriminator): 2D Latent Space (LF + HF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_4_latent_3d_tsne(results_m1, results_m2, output_path):
    """
    Plot 4: 3D Latent Space (t-SNE: LF, BP, HF)
    """
    fig = plt.figure(figsize=(16, 7))

    # Method 1
    ax = fig.add_subplot(121, projection='3d')
    latent_combined_m1 = np.concatenate([
        results_m1['latent_lf'],
        results_m1['latent_bp'],
        results_m1['latent_hf']
    ], axis=1)

    print("Computing t-SNE for Method 1 (3D)...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    latent_3d_m1 = tsne.fit_transform(latent_combined_m1)

    real_mask_m1 = (results_m1['labels'] == 0)
    fake_mask_m1 = (results_m1['labels'] == 1)

    ax.scatter(latent_3d_m1[real_mask_m1, 0], latent_3d_m1[real_mask_m1, 1], latent_3d_m1[real_mask_m1, 2],
               c='blue', alpha=0.5, s=10, label='Real')
    ax.scatter(latent_3d_m1[fake_mask_m1, 0], latent_3d_m1[fake_mask_m1, 1], latent_3d_m1[fake_mask_m1, 2],
               c='red', alpha=0.5, s=10, label='Fake (Audio-Driven)')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_zlabel('t-SNE Dim 3')
    ax.set_title('Method 1 (Margin Loss): 3D Latent Space')
    ax.legend()

    # Method 2
    ax = fig.add_subplot(122, projection='3d')
    latent_combined_m2 = np.concatenate([
        results_m2['latent_lf'],
        results_m2['latent_bp'],
        results_m2['latent_hf']
    ], axis=1)

    print("Computing t-SNE for Method 2 (3D)...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    latent_3d_m2 = tsne.fit_transform(latent_combined_m2)

    real_mask_m2 = (results_m2['labels'] == 0)
    fake_mask_m2 = (results_m2['labels'] == 1)

    ax.scatter(latent_3d_m2[real_mask_m2, 0], latent_3d_m2[real_mask_m2, 1], latent_3d_m2[real_mask_m2, 2],
               c='blue', alpha=0.5, s=10, label='Real')
    ax.scatter(latent_3d_m2[fake_mask_m2, 0], latent_3d_m2[fake_mask_m2, 1], latent_3d_m2[fake_mask_m2, 2],
               c='red', alpha=0.5, s=10, label='Fake (Audio-Driven)')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_zlabel('t-SNE Dim 3')
    ax.set_title('Method 2 (Discriminator): 3D Latent Space')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_5_score_comparison(results_m1, results_m2, output_path):
    """
    Plot 5: Score Comparison (Audio-Driven only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    real_mask_m1 = (results_m1['labels'] == 0)
    fake_mask_m1 = (results_m1['labels'] == 1)
    real_mask_m2 = (results_m2['labels'] == 0)
    fake_mask_m2 = (results_m2['labels'] == 1)

    # Method 1
    ax = axes[0]
    ax.hist(results_m1['scores'][real_mask_m1], bins=30, alpha=0.6,
            label=f'Real (n={real_mask_m1.sum()})', color='blue', density=True)
    ax.hist(results_m1['scores'][fake_mask_m1], bins=30, alpha=0.6,
            label=f'Audio-Driven (n={fake_mask_m1.sum()})', color='red', density=True)
    ax.set_xlabel('Reconstruction Loss')
    ax.set_ylabel('Density')
    ax.set_title('Method 1 (Margin Loss): Audio-Driven Fake')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Method 2
    ax = axes[1]
    ax.hist(results_m2['scores'][real_mask_m2], bins=30, alpha=0.6,
            label=f'Real (n={real_mask_m2.sum()})', color='blue', density=True)
    ax.hist(results_m2['scores'][fake_mask_m2], bins=30, alpha=0.6,
            label=f'Audio-Driven (n={fake_mask_m2.sum()})', color='red', density=True)
    ax.set_xlabel('Discriminator Score')
    ax.set_ylabel('Density')
    ax.set_title('Method 2 (Discriminator): Audio-Driven Fake')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ========================================
# Main Evaluation
# ========================================

def main():
    print("="*80)
    print("Stage 2 Methods Evaluation (Audio-Driven Only)")
    print("="*80)
    print()

    # Load data split
    print("Loading data split...")
    with open(SPLIT_JSON, 'r') as f:
        data_split = json.load(f)

    print(f"Final test set: {len(data_split['final_test']['real'])} real, {len(data_split['final_test']['fake'])} fake")
    print()

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name='final_test',
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order,
        random_crop=False
    )

    # Filter to keep only audio-driven fakes
    print("Filtering to keep only audio-driven fake samples...")
    keep_indices = filter_audiodriven_indices(test_dataset)
    filtered_dataset = Subset(test_dataset, keep_indices)

    # Count samples
    n_real = sum(1 for i in keep_indices if test_dataset.labels[i] == 0)
    n_fake = sum(1 for i in keep_indices if test_dataset.labels[i] == 1)
    print(f"Filtered dataset: {n_real} real, {n_fake} fake (audio-driven only)")
    print()

    test_loader = DataLoader(
        filtered_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # ========================================
    # Method 1: Margin Loss
    # ========================================
    print("="*80)
    print("Evaluating Method 1 (Margin Loss)...")
    print("="*80)

    checkpoint_m1 = torch.load(METHOD1_CHECKPOINT, map_location=device, weights_only=False)
    model_m1 = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)
    model_m1.load_state_dict(checkpoint_m1['model_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint_m1['epoch']}")
    print(f"  Best separation: {checkpoint_m1['val_separation']:.4f}")
    print()

    print("Computing scores for Method 1...")
    results_m1 = compute_scores_method1(model_m1, test_loader, device)
    print(f"  Real samples: {(results_m1['labels'] == 0).sum()}")
    print(f"  Fake samples: {(results_m1['labels'] == 1).sum()}")
    print(f"  Mean score (Real): {results_m1['scores'][results_m1['labels'] == 0].mean():.4f}")
    print(f"  Mean score (Fake): {results_m1['scores'][results_m1['labels'] == 1].mean():.4f}")
    print()

    # ========================================
    # Method 2: Discriminator
    # ========================================
    print("="*80)
    print("Evaluating Method 2 (Discriminator)...")
    print("="*80)

    checkpoint_m2 = torch.load(METHOD2_CHECKPOINT, map_location=device, weights_only=False)
    model_m2 = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)
    model_m2.load_state_dict(checkpoint_m2['model_state_dict'])

    discriminator_m2 = LatentDiscriminator(
        C_z=config.C_z,
        num_bands=3,
        hidden_dim=128
    ).to(device)
    discriminator_m2.load_state_dict(checkpoint_m2['discriminator_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint_m2['epoch']}")
    print(f"  Best accuracy: {checkpoint_m2['val_accuracy']:.4f}")
    print()

    print("Computing scores for Method 2...")
    results_m2 = compute_scores_method2(model_m2, discriminator_m2, test_loader, device)
    print(f"  Real samples: {(results_m2['labels'] == 0).sum()}")
    print(f"  Fake samples: {(results_m2['labels'] == 1).sum()}")
    print(f"  Mean score (Real): {results_m2['scores'][results_m2['labels'] == 0].mean():.4f}")
    print(f"  Mean score (Fake): {results_m2['scores'][results_m2['labels'] == 1].mean():.4f}")
    print()

    # ========================================
    # Visualizations
    # ========================================
    print("="*80)
    print("Generating visualizations...")
    print("="*80)
    print()

    plot_1_reconstruction_loss_distribution(
        results_m1, results_m2,
        os.path.join(OUTPUT_DIR, "1_score_distribution.png")
    )

    auc_m1, auc_m2 = plot_2_roc_curve(
        results_m1, results_m2,
        os.path.join(OUTPUT_DIR, "2_roc_curve.png")
    )

    plot_3_latent_2d_tsne(
        results_m1, results_m2,
        os.path.join(OUTPUT_DIR, "3_latent_2d_tsne.png")
    )

    plot_4_latent_3d_tsne(
        results_m1, results_m2,
        os.path.join(OUTPUT_DIR, "4_latent_3d_tsne.png")
    )

    plot_5_score_comparison(
        results_m1, results_m2,
        os.path.join(OUTPUT_DIR, "5_score_comparison.png")
    )

    # ========================================
    # Summary
    # ========================================
    print()
    print("="*80)
    print("Evaluation Summary (Audio-Driven Only)")
    print("="*80)
    print()
    print(f"Dataset: {n_real} Real, {n_fake} Fake (audio-driven)")
    print()
    print(f"Method 1 (Margin Loss):")
    print(f"  - AUC: {auc_m1:.4f}")
    print(f"  - Mean score (Real): {results_m1['scores'][results_m1['labels'] == 0].mean():.4f}")
    print(f"  - Mean score (Fake): {results_m1['scores'][results_m1['labels'] == 1].mean():.4f}")
    print(f"  - Separation: {results_m1['scores'][results_m1['labels'] == 1].mean() - results_m1['scores'][results_m1['labels'] == 0].mean():.4f}")
    print()
    print(f"Method 2 (Discriminator):")
    print(f"  - AUC: {auc_m2:.4f}")
    print(f"  - Mean score (Real): {results_m2['scores'][results_m2['labels'] == 0].mean():.4f}")
    print(f"  - Mean score (Fake): {results_m2['scores'][results_m2['labels'] == 1].mean():.4f}")
    print(f"  - Separation: {results_m2['scores'][results_m2['labels'] == 1].mean() - results_m2['scores'][results_m2['labels'] == 0].mean():.4f}")
    print()
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
