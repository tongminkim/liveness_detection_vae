"""
Evaluate and Compare Stage 2 Methods
- Method 1: Margin Loss
- Method 2: Discriminator

Evaluate on final_test split and generate comparison metrics
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from train_stage2_method2_discriminator import LatentDiscriminator
from config_bandvae import FullFeatureConfig


# ========================================
# Configuration
# ========================================

SPLIT_JSON = "/home/elicer/liveness_detection/model1/data_split.json"

# Model checkpoints
METHOD1_CHECKPOINT = "runs/stage2_method1_margin/stage2_method1_best.pt"
METHOD2_CHECKPOINT = "runs/stage2_method2_discriminator/stage2_method2_best.pt"

# Output
OUTPUT_DIR = "runs/stage2_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Evaluation Functions
# ========================================

@torch.no_grad()
def evaluate_method1_margin(model, loader, device, margin):
    """Evaluate Method 1 (Margin Loss) - Use reconstruction loss as anomaly score"""
    model.eval()

    all_labels = []
    all_losses = []

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

            all_labels.append(labels[i].item())
            all_losses.append(loss_dict['total'])

    all_labels = np.array(all_labels)
    all_losses = np.array(all_losses)

    # Anomaly score = reconstruction loss (higher for fake)
    anomaly_scores = all_losses

    # Compute metrics
    auc = roc_auc_score(all_labels, anomaly_scores)

    # Find best threshold
    thresholds = np.linspace(all_losses.min(), all_losses.max(), 100)
    best_acc = 0.0
    best_threshold = 0.0

    for threshold in thresholds:
        preds = (anomaly_scores > threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # Final predictions with best threshold
    preds = (anomaly_scores > best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary')

    # Separation
    real_losses = all_losses[all_labels == 0]
    fake_losses = all_losses[all_labels == 1]
    separation = fake_losses.mean() - real_losses.mean()

    return {
        'auc': auc,
        'accuracy': best_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': best_threshold,
        'real_loss_mean': real_losses.mean(),
        'real_loss_std': real_losses.std(),
        'fake_loss_mean': fake_losses.mean(),
        'fake_loss_std': fake_losses.std(),
        'separation': separation,
        'labels': all_labels,
        'scores': anomaly_scores
    }


@torch.no_grad()
def evaluate_method2_discriminator(model, discriminator, loader, device):
    """Evaluate Method 2 (Discriminator) - Use discriminator output as anomaly score"""
    model.eval()
    discriminator.eval()

    all_labels = []
    all_logits = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        # Discriminator prediction
        z_lf = mus['lf']
        z_bp = mus['bp']
        z_hf = mus['hf']

        logits = discriminator(z_lf, z_bp, z_hf).squeeze(1)  # (B,)

        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(logits.cpu().numpy())

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # Anomaly score = sigmoid(logits)
    anomaly_scores = 1 / (1 + np.exp(-all_logits))  # Sigmoid

    # Compute metrics
    auc = roc_auc_score(all_labels, anomaly_scores)

    # Find best threshold
    thresholds = np.linspace(0, 1, 100)
    best_acc = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (anomaly_scores > threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # Final predictions with best threshold
    preds = (anomaly_scores > best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary')

    # Separation
    real_scores = anomaly_scores[all_labels == 0]
    fake_scores = anomaly_scores[all_labels == 1]
    separation = fake_scores.mean() - real_scores.mean()

    return {
        'auc': auc,
        'accuracy': best_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': best_threshold,
        'real_score_mean': real_scores.mean(),
        'real_score_std': real_scores.std(),
        'fake_score_mean': fake_scores.mean(),
        'fake_score_std': fake_scores.std(),
        'separation': separation,
        'labels': all_labels,
        'scores': anomaly_scores
    }


@torch.no_grad()
def extract_latent_representations(model, loader, device):
    """Extract latent representations for t-SNE visualization"""
    model.eval()

    all_latents = []
    all_labels = []

    for x_lf, x_bp, x_hf, labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        # Concatenate latent vectors
        z_lf = mus['lf']  # (B, C_z)
        z_bp = mus['bp']  # (B, C_z)
        z_hf = mus['hf']  # (B, C_z)
        z_concat = torch.cat([z_lf, z_bp, z_hf], dim=1)  # (B, C_z * 3)

        all_latents.append(z_concat.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_latents = np.vstack(all_latents)
    all_labels = np.array(all_labels)

    return all_latents, all_labels


def plot_tsne_comparison(latents_m1, labels_m1, latents_m2, labels_m2, output_path):
    """Plot t-SNE visualization comparing two methods"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Method 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d_m1 = tsne.fit_transform(latents_m1)

    real_mask = (labels_m1 == 0)
    fake_mask = (labels_m1 == 1)

    axes[0].scatter(latents_2d_m1[real_mask, 0], latents_2d_m1[real_mask, 1],
                   c='blue', label='Real', alpha=0.6, s=30)
    axes[0].scatter(latents_2d_m1[fake_mask, 0], latents_2d_m1[fake_mask, 1],
                   c='red', label='Fake', alpha=0.6, s=30)
    axes[0].set_title('Method 1: Margin Loss', fontsize=14)
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Method 2
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d_m2 = tsne.fit_transform(latents_m2)

    real_mask = (labels_m2 == 0)
    fake_mask = (labels_m2 == 1)

    axes[1].scatter(latents_2d_m2[real_mask, 0], latents_2d_m2[real_mask, 1],
                   c='blue', label='Real', alpha=0.6, s=30)
    axes[1].scatter(latents_2d_m2[fake_mask, 0], latents_2d_m2[fake_mask, 1],
                   c='red', label='Fake', alpha=0.6, s=30)
    axes[1].set_title('Method 2: Discriminator', fontsize=14)
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE comparison to: {output_path}")


def plot_score_distributions(results_m1, results_m2, output_path):
    """Plot anomaly score distributions for both methods"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Method 1
    labels_m1 = results_m1['labels']
    scores_m1 = results_m1['scores']

    real_scores = scores_m1[labels_m1 == 0]
    fake_scores = scores_m1[labels_m1 == 1]

    axes[0].hist(real_scores, bins=50, alpha=0.6, label='Real', color='blue')
    axes[0].hist(fake_scores, bins=50, alpha=0.6, label='Fake', color='red')
    axes[0].axvline(results_m1['threshold'], color='black', linestyle='--',
                   label=f"Threshold={results_m1['threshold']:.4f}")
    axes[0].set_title(f"Method 1: Margin Loss (AUC={results_m1['auc']:.4f})", fontsize=14)
    axes[0].set_xlabel('Reconstruction Loss')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Method 2
    labels_m2 = results_m2['labels']
    scores_m2 = results_m2['scores']

    real_scores = scores_m2[labels_m2 == 0]
    fake_scores = scores_m2[labels_m2 == 1]

    axes[1].hist(real_scores, bins=50, alpha=0.6, label='Real', color='blue')
    axes[1].hist(fake_scores, bins=50, alpha=0.6, label='Fake', color='red')
    axes[1].axvline(results_m2['threshold'], color='black', linestyle='--',
                   label=f"Threshold={results_m2['threshold']:.4f}")
    axes[1].set_title(f"Method 2: Discriminator (AUC={results_m2['auc']:.4f})", fontsize=14)
    axes[1].set_xlabel('Anomaly Score (Sigmoid)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved score distributions to: {output_path}")


# ========================================
# Main Evaluation
# ========================================

def main():
    print("="*80)
    print("Stage 2 Methods Comparison")
    print("="*80)
    print(f"Method 1 checkpoint: {METHOD1_CHECKPOINT}")
    print(f"Method 2 checkpoint: {METHOD2_CHECKPOINT}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    print()

    # Load test dataset
    print("Loading test dataset...")
    config = FullFeatureConfig()
    config.T_fixed = 300

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
    print()

    # ========================================
    # Evaluate Method 1
    # ========================================
    print("Evaluating Method 1 (Margin Loss)...")
    checkpoint_m1 = torch.load(METHOD1_CHECKPOINT, map_location=device)

    model_m1 = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=checkpoint_m1['config'].C_h,
        C_z=checkpoint_m1['config'].C_z,
        dilations=checkpoint_m1['config'].dilations
    ).to(device)
    model_m1.load_state_dict(checkpoint_m1['model_state_dict'])

    results_m1 = evaluate_method1_margin(model_m1, test_loader, device, checkpoint_m1['margin'])
    latents_m1, labels_m1 = extract_latent_representations(model_m1, test_loader, device)

    print(f"Method 1 Results:")
    print(f"  AUC: {results_m1['auc']:.4f}")
    print(f"  Accuracy: {results_m1['accuracy']:.4f}")
    print(f"  Precision: {results_m1['precision']:.4f}")
    print(f"  Recall: {results_m1['recall']:.4f}")
    print(f"  F1-score: {results_m1['f1']:.4f}")
    print(f"  Separation: {results_m1['separation']:.4f}")
    print()

    # ========================================
    # Evaluate Method 2
    # ========================================
    print("Evaluating Method 2 (Discriminator)...")
    checkpoint_m2 = torch.load(METHOD2_CHECKPOINT, map_location=device)

    model_m2 = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=checkpoint_m2['config'].C_h,
        C_z=checkpoint_m2['config'].C_z,
        dilations=checkpoint_m2['config'].dilations
    ).to(device)
    model_m2.load_state_dict(checkpoint_m2['model_state_dict'])

    discriminator_m2 = LatentDiscriminator(
        C_z=checkpoint_m2['config'].C_z,
        num_bands=3,
        hidden_dim=128
    ).to(device)
    discriminator_m2.load_state_dict(checkpoint_m2['discriminator_state_dict'])

    results_m2 = evaluate_method2_discriminator(model_m2, discriminator_m2, test_loader, device)
    latents_m2, labels_m2 = extract_latent_representations(model_m2, test_loader, device)

    print(f"Method 2 Results:")
    print(f"  AUC: {results_m2['auc']:.4f}")
    print(f"  Accuracy: {results_m2['accuracy']:.4f}")
    print(f"  Precision: {results_m2['precision']:.4f}")
    print(f"  Recall: {results_m2['recall']:.4f}")
    print(f"  F1-score: {results_m2['f1']:.4f}")
    print(f"  Separation: {results_m2['separation']:.4f}")
    print()

    # ========================================
    # Save Results
    # ========================================
    print("Saving results...")

    # Save metrics
    comparison = {
        'method1_margin': {
            'auc': float(results_m1['auc']),
            'accuracy': float(results_m1['accuracy']),
            'precision': float(results_m1['precision']),
            'recall': float(results_m1['recall']),
            'f1': float(results_m1['f1']),
            'separation': float(results_m1['separation']),
            'threshold': float(results_m1['threshold']),
            'real_loss_mean': float(results_m1['real_loss_mean']),
            'fake_loss_mean': float(results_m1['fake_loss_mean'])
        },
        'method2_discriminator': {
            'auc': float(results_m2['auc']),
            'accuracy': float(results_m2['accuracy']),
            'precision': float(results_m2['precision']),
            'recall': float(results_m2['recall']),
            'f1': float(results_m2['f1']),
            'separation': float(results_m2['separation']),
            'threshold': float(results_m2['threshold']),
            'real_score_mean': float(results_m2['real_score_mean']),
            'fake_score_mean': float(results_m2['fake_score_mean'])
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'comparison_metrics.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"Saved metrics to: {os.path.join(OUTPUT_DIR, 'comparison_metrics.json')}")

    # Plot visualizations
    plot_tsne_comparison(
        latents_m1, labels_m1, latents_m2, labels_m2,
        os.path.join(OUTPUT_DIR, 'tsne_comparison.png')
    )

    plot_score_distributions(
        results_m1, results_m2,
        os.path.join(OUTPUT_DIR, 'score_distributions.png')
    )

    print()
    print("="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
