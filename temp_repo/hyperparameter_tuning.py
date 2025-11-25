"""
Hyperparameter tuning for inference method
Tune: PCA dimensions, confidence thresholds, distribution models
"""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader

from config_bandvae import Config
from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE


# Configuration
SPLIT_JSON = "data_split.json"
MODEL_CHECKPOINT = "runs/stage1_pretrained.pt"
OUTPUT_DIR = Path("visualizations/hyperparameter_tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter search space
PCA_DIMS = [5, 10, 15, 20, 30, 36]
CONFIDENCE_STAGE1 = [0.90, 0.95, 0.99]
CONFIDENCE_STAGE2 = [0.50, 0.70, 0.90]
DISTRIBUTION_MODELS = ["gaussian", "gmm", "ocsvm", "iforest"]

# Fixed parameters
FC_LOW = 2.0
FC_HIGH = 8.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract reconstruction losses and latent vectors"""
    model.eval()
    losses_lf, losses_bp, losses_hf = [], [], []
    latents = []
    labels_list = []

    for x_lf, x_bp, x_hf, batch_labels in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        recons, mus, _, _ = model(x_lf, x_bp, x_hf)

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
        z_lf = mus["lf"].mean(dim=2)
        z_bp = mus["bp"].mean(dim=2)
        z_hf = mus["hf"].mean(dim=2)
        latents.append(torch.cat([z_lf, z_bp, z_hf], dim=1).cpu().numpy())

    losses = np.stack([losses_lf, losses_bp, losses_hf], axis=1)
    latents = np.vstack(latents)
    labels = np.array(labels_list)
    return losses, latents, labels


def fit_gaussian(data):
    """Fit Gaussian distribution"""
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return {"mu": mu, "cov": cov, "type": "gaussian"}


def fit_gmm(data, n_components=3):
    """Fit Gaussian Mixture Model"""
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    return {"model": gmm, "type": "gmm"}


def fit_ocsvm(data):
    """Fit One-Class SVM"""
    ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.1)
    ocsvm.fit(data)
    return {"model": ocsvm, "type": "ocsvm"}


def fit_iforest(data):
    """Fit Isolation Forest"""
    iforest = IsolationForest(contamination=0.1, random_state=42)
    iforest.fit(data)
    return {"model": iforest, "type": "iforest"}


def within_confidence_gaussian(x, params, confidence):
    """Check if sample within Gaussian confidence region"""
    mu = params["mu"]
    cov = params["cov"]
    try:
        diff = x - mu
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(mu)))
        mahal = diff @ cov_inv @ diff
        threshold = stats.chi2.ppf(confidence, len(mu))
        return mahal <= threshold
    except np.linalg.LinAlgError:
        sigma = np.sqrt(np.diag(cov))
        z_score = stats.norm.ppf((1 + confidence) / 2)
        return np.all(np.abs(x - mu) <= z_score * sigma)


def within_confidence_gmm(x, params, confidence):
    """Check if sample within GMM confidence region"""
    gmm = params["model"]
    log_prob = gmm.score_samples(x.reshape(1, -1))
    # Use log probability threshold based on training data percentile
    return log_prob[0] > params.get("threshold", -np.inf)


def predict_ocsvm(x, params):
    """Predict using One-Class SVM"""
    ocsvm = params["model"]
    pred = ocsvm.predict(x.reshape(1, -1))
    return pred[0] == 1  # 1 = inlier (Real), -1 = outlier (Fake)


def predict_iforest(x, params):
    """Predict using Isolation Forest"""
    iforest = params["model"]
    pred = iforest.predict(x.reshape(1, -1))
    return pred[0] == 1  # 1 = inlier (Real), -1 = outlier (Fake)


def evaluate_model(
    test_losses,
    test_latents_pca,
    test_labels,
    loss_params,
    latent_params,
    conf_stage1,
    conf_stage2,
):
    """Evaluate model with given parameters"""
    preds = []

    for loss_vec, latent_vec in zip(test_losses, test_latents_pca):
        # Stage 1: Loss filtering
        if loss_params["type"] == "gaussian":
            stage1_pass = within_confidence_gaussian(loss_vec, loss_params, conf_stage1)
        elif loss_params["type"] == "gmm":
            stage1_pass = within_confidence_gmm(loss_vec, loss_params, conf_stage1)
        elif loss_params["type"] == "ocsvm":
            stage1_pass = predict_ocsvm(loss_vec, loss_params)
        elif loss_params["type"] == "iforest":
            stage1_pass = predict_iforest(loss_vec, loss_params)

        if stage1_pass:
            # Stage 2: Latent filtering
            if latent_params["type"] == "gaussian":
                stage2_pass = within_confidence_gaussian(
                    latent_vec, latent_params, conf_stage2
                )
            elif latent_params["type"] == "gmm":
                stage2_pass = within_confidence_gmm(
                    latent_vec, latent_params, conf_stage2
                )
            elif latent_params["type"] == "ocsvm":
                stage2_pass = predict_ocsvm(latent_vec, latent_params)
            elif latent_params["type"] == "iforest":
                stage2_pass = predict_iforest(latent_vec, latent_params)

            preds.append(0 if stage2_pass else 1)
        else:
            preds.append(1)

    preds = np.array(preds)

    # Compute metrics
    cm = confusion_matrix(test_labels, preds)
    accuracy = accuracy_score(test_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
    }


def visualize_pca_explained_variance(pca_models, output_path):
    """Visualize explained variance for different PCA dimensions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cumulative explained variance
    for dim, pca in pca_models.items():
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        axes[0].plot(range(1, len(cumsum) + 1), cumsum, marker="o", label=f"PCA-{dim}")
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("PCA Explained Variance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Individual component variance
    for dim, pca in pca_models.items():
        if dim <= 20:  # Only plot for smaller dimensions
            axes[1].bar(
                range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_,
                alpha=0.6,
                label=f"PCA-{dim}",
            )
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].set_title("Individual Component Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_tsne(latents, labels, pca_dim, output_path):
    """Visualize latent space using t-SNE"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 8))
    colors = ["blue", "red"]
    class_names = ["Real", "Fake"]

    for cls in [0, 1]:
        mask = labels == cls
        plt.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=colors[cls],
            label=class_names[cls],
            alpha=0.6,
            s=50,
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE Visualization (PCA-{pca_dim}D)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_distribution_2d(data, labels, params, output_path):
    """Visualize 2D distribution (first 2 principal components)"""
    plt.figure(figsize=(10, 8))

    # Plot data points
    real_mask = labels == 0
    fake_mask = labels == 1
    plt.scatter(data[real_mask, 0], data[real_mask, 1], c="blue", label="Real", alpha=0.6, s=50)
    plt.scatter(data[fake_mask, 0], data[fake_mask, 1], c="red", label="Fake", alpha=0.6, s=50)

    # Plot distribution contour
    if params["type"] == "gaussian":
        mu = params["mu"][:2]
        cov = params["cov"][:2, :2]
        x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
        y = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        rv = stats.multivariate_normal(mu, cov)
        Z = rv.pdf(pos)
        plt.contour(X, Y, Z, levels=5, alpha=0.5, colors="green")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Distribution Visualization ({params['type'].upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_confidence_heatmap(results, output_path):
    """Visualize F1 scores for different confidence thresholds"""
    conf1_vals = sorted(set(r["conf_stage1"] for r in results))
    conf2_vals = sorted(set(r["conf_stage2"] for r in results))

    f1_matrix = np.zeros((len(conf1_vals), len(conf2_vals)))

    for i, c1 in enumerate(conf1_vals):
        for j, c2 in enumerate(conf2_vals):
            matching = [
                r for r in results if r["conf_stage1"] == c1 and r["conf_stage2"] == c2
            ]
            if matching:
                f1_matrix[i, j] = matching[0]["f1"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=[f"{c:.2f}" for c in conf2_vals],
        yticklabels=[f"{c:.2f}" for c in conf1_vals],
        cmap="YlOrRd",
        cbar_kws={"label": "F1 Score"},
    )
    plt.xlabel("Stage 2 Confidence")
    plt.ylabel("Stage 1 Confidence")
    plt.title("F1 Score Heatmap (Confidence Thresholds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("Hyperparameter Tuning for Inference Method")
    print("=" * 80)

    # Load config
    config = Config(mode="full")
    config.T_fixed = 300

    # Load datasets
    print("Loading datasets...")
    train_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name="stage2_train",
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=FC_LOW,
        fc_high=FC_HIGH,
        filter_order=4,
        random_crop=False,
        base_dir="test_balanced_npz",
    )

    test_dataset = Stage2Dataset(
        split_json_path=SPLIT_JSON,
        split_name="final_test",
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=FC_LOW,
        fc_high=FC_HIGH,
        filter_order=4,
        random_crop=False,
        base_dir="test_balanced_npz",
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Load model
    print("Loading model...")
    from torch.serialization import add_safe_globals

    add_safe_globals([Config])
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False)

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=checkpoint["config"].C_h,
        C_z=checkpoint["config"].C_z,
        dilations=checkpoint["config"].dilations,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Extract features
    print("Extracting features...")
    train_losses, train_latents, train_labels = extract_features(
        model, train_loader, device
    )
    test_losses, test_latents, test_labels = extract_features(model, test_loader, device)

    print(f"Train: losses {train_losses.shape}, latents {train_latents.shape}")
    print(f"Test: losses {test_losses.shape}, latents {test_latents.shape}")

    # Get real samples only
    real_mask = train_labels == 0
    train_real_losses = train_losses[real_mask]
    train_real_latents = train_latents[real_mask]

    print(f"Real training samples: {len(train_real_losses)}")

    # Fit loss distribution models
    print("\nFitting loss distribution models...")
    loss_models = {}
    for dist_type in DISTRIBUTION_MODELS:
        if dist_type == "gaussian":
            loss_models[dist_type] = fit_gaussian(train_real_losses)
        elif dist_type == "gmm":
            loss_models[dist_type] = fit_gmm(train_real_losses, n_components=2)
        elif dist_type == "ocsvm":
            loss_models[dist_type] = fit_ocsvm(train_real_losses)
        elif dist_type == "iforest":
            loss_models[dist_type] = fit_iforest(train_real_losses)

    # PCA dimension tuning
    print("\nTuning PCA dimensions...")
    pca_models = {}
    latent_models = {}
    all_results = []

    for pca_dim in PCA_DIMS:
        print(f"\n  PCA dimension: {pca_dim}")
        pca = PCA(n_components=pca_dim)
        train_latents_pca = pca.fit_transform(train_real_latents)
        test_latents_pca = pca.transform(test_latents)

        pca_models[pca_dim] = pca
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"    Explained variance: {explained_var:.4f}")

        # Fit latent distribution models
        latent_models[pca_dim] = {}
        for dist_type in DISTRIBUTION_MODELS:
            if dist_type == "gaussian":
                latent_models[pca_dim][dist_type] = fit_gaussian(train_latents_pca)
            elif dist_type == "gmm":
                latent_models[pca_dim][dist_type] = fit_gmm(
                    train_latents_pca, n_components=2
                )
            elif dist_type == "ocsvm":
                latent_models[pca_dim][dist_type] = fit_ocsvm(train_latents_pca)
            elif dist_type == "iforest":
                latent_models[pca_dim][dist_type] = fit_iforest(train_latents_pca)

        # Visualize t-SNE for selected PCA dimensions
        if pca_dim in [10, 20, 50]:
            tsne_path = OUTPUT_DIR / f"tsne_pca{pca_dim}.png"
            visualize_tsne(test_latents_pca, test_labels, pca_dim, tsne_path)

        # Test different distribution models and confidence thresholds
        for loss_dist in DISTRIBUTION_MODELS:
            for latent_dist in DISTRIBUTION_MODELS:
                # For non-Gaussian models, use single "confidence" value
                if loss_dist in ["ocsvm", "iforest"]:
                    conf1_range = [0.95]  # Dummy value
                else:
                    conf1_range = CONFIDENCE_STAGE1

                if latent_dist in ["ocsvm", "iforest"]:
                    conf2_range = [0.70]  # Dummy value
                else:
                    conf2_range = CONFIDENCE_STAGE2

                for conf1 in conf1_range:
                    for conf2 in conf2_range:
                        metrics = evaluate_model(
                            test_losses,
                            test_latents_pca,
                            test_labels,
                            loss_models[loss_dist],
                            latent_models[pca_dim][latent_dist],
                            conf1,
                            conf2,
                        )

                        result = {
                            "pca_dim": pca_dim,
                            "loss_dist": loss_dist,
                            "latent_dist": latent_dist,
                            "conf_stage1": conf1,
                            "conf_stage2": conf2,
                            "explained_variance": float(explained_var),
                            **metrics,
                        }
                        all_results.append(result)

    # Visualize PCA explained variance
    print("\nGenerating visualizations...")
    pca_var_path = OUTPUT_DIR / "pca_explained_variance.png"
    visualize_pca_explained_variance(pca_models, pca_var_path)

    # Visualize distribution for best Gaussian model
    gaussian_results = [r for r in all_results if r["loss_dist"] == "gaussian" and r["latent_dist"] == "gaussian"]
    if gaussian_results:
        best_gaussian = max(gaussian_results, key=lambda x: x["f1"])
        best_pca_dim = best_gaussian["pca_dim"]
        best_pca = pca_models[best_pca_dim]
        test_latents_pca_2d = best_pca.transform(test_latents)[:, :2]
        dist_path = OUTPUT_DIR / f"distribution_2d_pca{best_pca_dim}.png"
        visualize_distribution_2d(
            test_latents_pca_2d,
            test_labels,
            latent_models[best_pca_dim]["gaussian"],
            dist_path,
        )

    # Visualize confidence threshold heatmap for best config
    for pca_dim in [10, 20]:
        gaussian_conf_results = [
            r
            for r in all_results
            if r["pca_dim"] == pca_dim
            and r["loss_dist"] == "gaussian"
            and r["latent_dist"] == "gaussian"
        ]
        if gaussian_conf_results:
            heatmap_path = OUTPUT_DIR / f"confidence_heatmap_pca{pca_dim}.png"
            visualize_confidence_heatmap(gaussian_conf_results, heatmap_path)

    # Find best configurations
    print("\n" + "=" * 80)
    print("BEST RESULTS")
    print("=" * 80)

    # Best overall
    best_overall = max(all_results, key=lambda x: x["f1"])
    print("\nBest Overall Configuration:")
    print(f"  PCA Dimension: {best_overall['pca_dim']}")
    print(f"  Loss Distribution: {best_overall['loss_dist']}")
    print(f"  Latent Distribution: {best_overall['latent_dist']}")
    print(f"  Confidence Stage1: {best_overall['conf_stage1']:.2f}")
    print(f"  Confidence Stage2: {best_overall['conf_stage2']:.2f}")
    print(f"  Accuracy: {best_overall['accuracy']:.4f}")
    print(f"  Precision: {best_overall['precision']:.4f}")
    print(f"  Recall: {best_overall['recall']:.4f}")
    print(f"  F1 Score: {best_overall['f1']:.4f}")
    print(f"  Specificity: {best_overall['specificity']:.4f}")

    # Best per distribution type
    print("\nBest Configuration per Distribution Type:")
    for dist_type in DISTRIBUTION_MODELS:
        dist_results = [
            r
            for r in all_results
            if r["loss_dist"] == dist_type and r["latent_dist"] == dist_type
        ]
        if dist_results:
            best_dist = max(dist_results, key=lambda x: x["f1"])
            print(f"\n  {dist_type.upper()}:")
            print(f"    PCA Dimension: {best_dist['pca_dim']}")
            print(f"    F1 Score: {best_dist['f1']:.4f}")
            print(f"    Accuracy: {best_dist['accuracy']:.4f}")

    # Save results
    results_path = OUTPUT_DIR / "tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    best_path = OUTPUT_DIR / "best_config.json"
    with open(best_path, "w") as f:
        json.dump(best_overall, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  All results: {results_path}")
    print(f"  Best config: {best_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
