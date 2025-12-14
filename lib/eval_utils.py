from __future__ import annotations

import json

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from config_bandvae import get_config
from matplotlib import pyplot as plt
from model import BandSplitVAE
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@torch.no_grad()
def compute_bandwise_reconstruction_losses(model, loader, device):
    model.eval()
    losses_lf, losses_bp, losses_hf, labels = [], [], [], []
    for x_lf, x_bp, x_hf, batch_labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        recons, _, _, _ = model(x_lf, x_bp, x_hf)
        for i in range(x_lf.size(0)):
            losses_lf.append(F.mse_loss(recons["lf"][i], x_lf[i]).item())
            losses_bp.append(F.mse_loss(recons["bp"][i], x_bp[i]).item())
            losses_hf.append(F.mse_loss(recons["hf"][i], x_hf[i]).item())
            labels.append(batch_labels[i].item())
    return np.stack([losses_lf, losses_bp, losses_hf], axis=1), np.array(labels)


@torch.no_grad()
def extract_latent_vectors(model, loader, device):
    model.eval()
    latents, labels = [], []
    for x_lf, x_bp, x_hf, batch_labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)
        _, mus, _, _ = model(x_lf, x_bp, x_hf)
        z_lf = mus["lf"].mean(dim=2)
        z_bp = mus["bp"].mean(dim=2)
        z_hf = mus["hf"].mean(dim=2)
        z_concat = torch.cat([z_lf, z_bp, z_hf], dim=1)
        latents.append(z_concat.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
    return np.vstack(latents), np.array(labels)


def fit_gaussian_multivariate(data):
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mu, cov


def is_within_confidence_multivariate(x, mu, cov, confidence):
    diff = x - mu
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(mu)))
    mahal_dist_sq = diff @ cov_inv @ diff
    dim = len(mu)
    threshold = stats.chi2.ppf(confidence, dim)
    return mahal_dist_sq <= threshold


def apply_pca(train_data, test_data, n_components):
    pca = PCA(n_components=n_components)
    train_reduced = pca.fit_transform(train_data)
    test_reduced = pca.transform(test_data)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    return train_reduced, test_reduced, explained_variance, pca


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
    stage1_pass = [
        is_within_confidence_multivariate(
            test_losses_3d[i], loss_mu_3d, loss_cov_3d, confidence_stage1
        )
        for i in range(len(test_losses_3d))
    ]
    stage1_pass = np.array(stage1_pass)

    stage2_pass = [
        is_within_confidence_multivariate(
            test_latents_pca[i], latent_mu_pca, latent_cov_pca, confidence_stage2
        )
        for i in range(len(test_latents_pca))
    ]
    stage2_pass = np.array(stage2_pass)

    predictions = np.ones_like(test_labels)
    predictions[stage1_pass] = 0
    predictions[stage1_pass & stage2_pass] = 0

    tn, fp, fn, tp = confusion_matrix(test_labels, predictions, labels=[0, 1]).ravel()
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="binary", pos_label=1
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "predictions": predictions,
        "stage1_pass": stage1_pass,
        "stage2_pass": stage2_pass,
        "stage1_pass_rate": stage1_pass.mean(),
        "stage2_pass_rate": stage2_pass.mean(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=[0, 1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_confusion_matrix(cm, accuracy, f1, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix\nAccuracy={accuracy:.3f}, F1={f1:.3f}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_detailed_metrics(results, save_path):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "stage1_pass_rate",
        "stage2_pass_rate",
    ]
    values = [results[m] for m in metrics]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=metrics, y=values)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_evaluation(
    checkpoint_path,
    train_loader,
    test_loader,
    device,
    output_dir,
    pca_dim=10,
    confidence_stage1=0.95,
    confidence_stage2=0.70,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    config_eval = get_config("full")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BandSplitVAE(
        C_in_per_band=config_eval.C_in_per_band,
        C_h=checkpoint["config"].C_h,
        C_z=checkpoint["config"].C_z,
        dilations=checkpoint["config"].dilations,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_losses_3d, train_labels = compute_bandwise_reconstruction_losses(
        model, train_loader, device
    )
    train_latents_36d, _ = extract_latent_vectors(model, train_loader, device)
    test_losses_3d, test_labels = compute_bandwise_reconstruction_losses(
        model, test_loader, device
    )
    test_latents_36d, _ = extract_latent_vectors(model, test_loader, device)

    train_real_mask = train_labels == 0
    train_real_losses_3d = train_losses_3d[train_real_mask]
    train_real_latents_36d = train_latents_36d[train_real_mask]

    loss_mu_3d, loss_cov_3d = fit_gaussian_multivariate(train_real_losses_3d)
    train_latents_pca, test_latents_pca, explained_var, _ = apply_pca(
        train_real_latents_36d, test_latents_36d, pca_dim
    )
    latent_mu_pca, latent_cov_pca = fit_gaussian_multivariate(train_latents_pca)

    results = two_stage_filtering_pca10d(
        test_losses_3d,
        test_latents_pca,
        test_labels,
        loss_mu_3d,
        loss_cov_3d,
        latent_mu_pca,
        latent_cov_pca,
        confidence_stage1,
        confidence_stage2,
    )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "pca_dim": pca_dim,
        "explained_variance": float(explained_var),
        "confidence_stage1": confidence_stage1,
        "confidence_stage2": confidence_stage2,
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1": float(results["f1"]),
        "specificity": float(results["specificity"]),
        "stage1_pass_rate": float(results["stage1_pass_rate"]),
        "stage2_pass_rate": float(results["stage2_pass_rate"]),
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        results["confusion_matrix"],
        results["accuracy"],
        results["f1"],
        output_dir / "confusion_matrix.png",
    )
    plot_detailed_metrics(results, output_dir / "detailed_metrics.png")
    return results
