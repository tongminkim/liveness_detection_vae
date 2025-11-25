"""
Evaluate pretrained model with correct method (no StandardScaler)
"""
import json
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from config_bandvae import Config
from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE

# Configuration
SPLIT_JSON = "data_split.json"
MODEL_CHECKPOINT = "runs/stage1_pretrained.pt"
OUTPUT_FILE = "pretrained_correct_eval.json"

PCA_DIM = 10
CONFIDENCE_STAGE1 = 0.95
CONFIDENCE_STAGE2 = 0.70

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            losses_lf.append(
                torch.nn.functional.mse_loss(recons["lf"][i], x_lf[i]).item()
            )
            losses_bp.append(
                torch.nn.functional.mse_loss(recons["bp"][i], x_bp[i]).item()
            )
            losses_hf.append(
                torch.nn.functional.mse_loss(recons["hf"][i], x_hf[i]).item()
            )
            labels.append(batch_labels[i].item())

    losses = np.stack([losses_lf, losses_bp, losses_hf], axis=1)
    labels = np.array(labels)
    return losses, labels


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
        latents.append(torch.cat([z_lf, z_bp, z_hf], dim=1).cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())

    lat_all = np.vstack(latents)
    labels = np.array(labels)
    return lat_all, labels


def fit_gaussian(data):
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mu, cov


def within_confidence(x, mu, cov, confidence):
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


def main():
    print("=" * 80)
    print("Evaluating Pretrained Model (Correct Method - No StandardScaler)")
    print("=" * 80)

    # Load config
    config = Config(mode="full")
    config.T_fixed = 300

    # Build datasets
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
        fc_low=2.0,
        fc_high=8.0,
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
    print("Extracting train features...")
    train_losses, train_labels = compute_bandwise_reconstruction_losses(model, train_loader, device)
    train_latents, _ = extract_latent_vectors(model, train_loader, device)

    print("Extracting test features...")
    test_losses, test_labels = compute_bandwise_reconstruction_losses(model, test_loader, device)
    test_latents, _ = extract_latent_vectors(model, test_loader, device)

    print(f"Train losses shape: {train_losses.shape}, latents: {train_latents.shape}")
    print(f"Test losses shape: {test_losses.shape}, latents: {test_latents.shape}")

    # Fit Gaussians on real samples only
    real_mask = train_labels == 0
    train_real_losses = train_losses[real_mask]
    train_real_latents = train_latents[real_mask]

    print(f"Real training samples: {len(train_real_losses)}")

    # Stage 1: Fit Gaussian on 3D losses
    loss_mu, loss_cov = fit_gaussian(train_real_losses)
    print(f"Loss mu: {loss_mu}")

    # Stage 2: Apply PCA (NO StandardScaler!)
    print(f"Applying PCA to {PCA_DIM}D (NO StandardScaler)...")
    pca = PCA(n_components=PCA_DIM)
    train_latents_pca = pca.fit_transform(train_real_latents)  # No scaler!
    test_latents_pca = pca.transform(test_latents)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA explained variance: {explained_var:.4f}")

    # Fit Gaussian on PCA latents
    latent_mu, latent_cov = fit_gaussian(train_latents_pca)

    # Two-stage filtering
    print("Applying two-stage filtering...")
    preds = []
    stage1_pass = 0
    stage2_pass = 0

    for loss_vec, latent_vec in zip(test_losses, test_latents_pca):
        if within_confidence(loss_vec, loss_mu, loss_cov, CONFIDENCE_STAGE1):
            stage1_pass += 1
            if within_confidence(latent_vec, latent_mu, latent_cov, CONFIDENCE_STAGE2):
                stage2_pass += 1
                preds.append(0)  # Real
            else:
                preds.append(1)  # Fake
        else:
            preds.append(1)  # Fake

    preds = np.array(preds)

    # Compute metrics
    cm = confusion_matrix(test_labels, preds)
    accuracy = accuracy_score(test_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average="binary")
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Results
    results = {
        "model": MODEL_CHECKPOINT,
        "method": "Two-Stage Gaussian (NO StandardScaler)",
        "fc_low": 2.0,
        "fc_high": 8.0,
        "pca_dim": PCA_DIM,
        "confidence_stage1": CONFIDENCE_STAGE1,
        "confidence_stage2": CONFIDENCE_STAGE2,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "stage1_pass_rate": float(stage1_pass / len(test_labels)),
        "stage2_pass_rate": float(stage2_pass / len(test_labels)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "explained_variance": float(explained_var),
    }

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (Correct Method - No StandardScaler)")
    print("=" * 80)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print()
    print("Confusion Matrix:")
    print(cm)
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print()
    print(f"Stage 1 pass rate: {stage1_pass}/{len(test_labels)} = {stage1_pass/len(test_labels):.2%}")
    print(f"Stage 2 pass rate: {stage2_pass}/{len(test_labels)} = {stage2_pass/len(test_labels):.2%}")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
