import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config_bandvae import Config
from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE


def fit_gaussian(data):
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mu, cov


def within_confidence(x, mu, cov, confidence):
    diff = x - mu
    try:
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(mu)))
        dist_sq = diff @ cov_inv @ diff
        threshold = stats.chi2.ppf(confidence, len(mu))
        return dist_sq <= threshold
    except np.linalg.LinAlgError:
        sigma = np.sqrt(np.diag(cov))
        z_score = stats.norm.ppf((1 + confidence) / 2)
        return np.all(np.abs(diff) <= z_score * sigma)


class ModelVisualizer:
    def __init__(self, split_json, base_dir, output_root, device=None):
        self.split_json = Path(split_json)
        self.base_dir = Path(base_dir)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = Config(mode="full")
        sns.set_theme(style="whitegrid")

    def run(self, checkpoints, pca_dim=10, conf1=0.95, conf2=0.70):
        summary = []
        for ckpt_path in checkpoints:
            metrics = self._process_checkpoint(Path(ckpt_path), pca_dim, conf1, conf2)
            summary.append(metrics)
        self._write_summary(summary)

    def _build_loaders(self, T_fixed):
        common_kwargs = dict(
            split_json_path=self.split_json,
            T_fixed=T_fixed,
            fps=self.config.fps,
            use_acceleration=self.config.use_acceleration,
            use_angle=self.config.use_angle,
            use_angle_rate=self.config.use_angle_rate,
            fc_low=self.config.fc_low,
            fc_high=self.config.fc_high,
            filter_order=self.config.filter_order,
            random_crop=False,
            base_dir=self.base_dir,
        )

        train_ds = Stage2Dataset(split_name="stage2_train", **common_kwargs)
        test_ds = Stage2Dataset(split_name="final_test", **common_kwargs)

        loader_args = dict(batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        return (
            DataLoader(train_ds, **loader_args),
            DataLoader(test_ds, **loader_args),
        )

    def _load_model(self, ckpt_path):
        torch.serialization.add_safe_globals([Config])
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", self.config)

        model = BandSplitVAE(
            C_in_per_band=self.config.C_in_per_band,
            C_h=getattr(cfg, "C_h", self.config.C_h),
            C_z=getattr(cfg, "C_z", self.config.C_z),
            dilations=getattr(cfg, "dilations", self.config.dilations),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, getattr(cfg, "T_fixed", self.config.T_fixed)

    @torch.no_grad()
    def _collect_features(self, model, loader):
        model.eval()
        all_losses = []
        all_latents = []
        all_labels = []
        per_band = {"lf": [], "bp": [], "hf": []}
        per_band_latents = {"lf": [], "bp": [], "hf": []}

        for x_lf, x_bp, x_hf, labels in loader:
            x_lf = x_lf.to(self.device)
            x_bp = x_bp.to(self.device)
            x_hf = x_hf.to(self.device)

            recons, mus, logvars, _ = model(x_lf, x_bp, x_hf)

            mse_lf = nn.functional.mse_loss(recons["lf"], x_lf, reduction="none").mean(
                dim=[1, 2]
            )
            mse_bp = nn.functional.mse_loss(recons["bp"], x_bp, reduction="none").mean(
                dim=[1, 2]
            )
            mse_hf = nn.functional.mse_loss(recons["hf"], x_hf, reduction="none").mean(
                dim=[1, 2]
            )

            losses = torch.stack([mse_lf, mse_bp, mse_hf], dim=1)
            all_losses.append(losses.cpu().numpy())

            z_lf = mus["lf"].mean(dim=2)
            z_bp = mus["bp"].mean(dim=2)
            z_hf = mus["hf"].mean(dim=2)
            z_concat = torch.cat([z_lf, z_bp, z_hf], dim=1)
            all_latents.append(z_concat.cpu().numpy())
            per_band_latents["lf"].append(z_lf.cpu().numpy())
            per_band_latents["bp"].append(z_bp.cpu().numpy())
            per_band_latents["hf"].append(z_hf.cpu().numpy())

            per_band["lf"].append(mse_lf.cpu().numpy())
            per_band["bp"].append(mse_bp.cpu().numpy())
            per_band["hf"].append(mse_hf.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        losses_arr = np.vstack(all_losses)
        latents_arr = np.vstack(all_latents)
        labels_arr = np.array(all_labels)

        losses_arr = np.nan_to_num(losses_arr, copy=False, posinf=1e6, neginf=-1e6)
        latents_arr = np.nan_to_num(latents_arr, copy=False)
        per_band_latents = {
            k: np.nan_to_num(np.vstack(v), copy=False)
            for k, v in per_band_latents.items()
        }

        per_band = {
            k: np.nan_to_num(np.concatenate(v), copy=False) for k, v in per_band.items()
        }
        per_band["total"] = losses_arr.sum(axis=1)

        return losses_arr, latents_arr, labels_arr, per_band, per_band_latents

    def _two_stage_classify(
        self,
        train_losses,
        train_latents,
        train_labels,
        test_losses,
        test_latents,
        test_labels,
        pca_dim,
        conf1,
        conf2,
    ):
        real_mask = train_labels == 0
        real_losses = train_losses[real_mask].astype(np.float64)
        real_latents = train_latents[real_mask].astype(np.float64)
        test_losses = test_losses.astype(np.float64)
        test_latents = test_latents.astype(np.float64)

        loss_mu, loss_cov = fit_gaussian(real_losses)

        real_latents = np.clip(real_latents, -1e3, 1e3)
        test_latents = np.clip(test_latents, -1e3, 1e3)

        scaler = StandardScaler()
        real_latents = scaler.fit_transform(real_latents)
        test_latents = scaler.transform(test_latents)

        pca = PCA(n_components=pca_dim)
        train_lat_pca = pca.fit_transform(real_latents)
        test_lat_pca = pca.transform(test_latents)

        latent_mu, latent_cov = fit_gaussian(train_lat_pca)

        preds = []
        for loss_vec, latent_vec in zip(test_losses, test_lat_pca):
            pass_stage1 = within_confidence(loss_vec, loss_mu, loss_cov, conf1)
            if not pass_stage1:
                preds.append(1)
                continue

            pass_stage2 = within_confidence(latent_vec, latent_mu, latent_cov, conf2)
            preds.append(0 if pass_stage2 else 1)

        preds = np.array(preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average="binary"
        )
        acc = accuracy_score(test_labels, preds)
        return {
            "predictions": preds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
            "pca_model": pca,
            "train_lat_pca": train_lat_pca,
            "test_lat_pca": test_lat_pca,
        }

    def _plot_latent(self, latent, labels, title, path):
        plt.figure(figsize=(8, 6))
        palette = {0: "#1f77b4", 1: "#d62728"}
        sns.scatterplot(
            x=latent[:, 0],
            y=latent[:, 1],
            hue=labels,
            palette=palette,
            alpha=0.75,
            edgecolor="none",
        )
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Label", labels=["Real", "Fake"])
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _plot_latent_grid(self, embeddings, labels, title, output_path):
        """
        Grid plot for either PCA or t-SNE embeddings.
        `embeddings` is a dict: name -> (N,2) array.
        """
        names = ["lf", "bp", "hf", "merged"]
        palette = {0: "#1f77b4", 1: "#d62728"}
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for ax, name in zip(axes, names):
            emb = embeddings[name]
            sns.scatterplot(
                ax=ax,
                x=emb[:, 0],
                y=emb[:, 1],
                hue=labels,
                palette=palette,
                alpha=0.75,
                edgecolor="none",
            )
            ax.set_title(f"{name.upper()} {title}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            if ax.legend_:
                ax.legend_.remove()

        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(handles, ["Real", "Fake"], loc="upper center", ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_recon_losses(self, per_band, labels, path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        bands = [
            ("total", "Total (LF+BP+HF)"),
            ("lf", "LF"),
            ("bp", "BP"),
            ("hf", "HF"),
        ]

        for ax, (key, title) in zip(axes.flat, bands):
            real = per_band[key][labels == 0]
            fake = per_band[key][labels == 1]
            bins = 40
            ax.hist(
                real,
                bins=bins,
                alpha=0.6,
                color="#1f77b4",
                label="Real",
                density=True,
            )
            ax.hist(
                fake,
                bins=bins,
                alpha=0.6,
                color="#d62728",
                label="Fake",
                density=True,
            )
            ax.set_title(f"Reconstruction Loss - {title}")
            ax.set_xlabel("MSE")
            ax.set_ylabel("Density")
            ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _save_metrics(self, output_dir, metrics, checkpoint_name):
        metrics_path = output_dir / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": checkpoint_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        with metrics_path.open("w") as f:
            json.dump(payload, f, indent=2)
        return payload

    def _write_summary(self, rows):
        if not rows:
            return
        summary_path = self.output_root / "metrics_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["checkpoint", "accuracy", "precision", "recall", "f1"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved summary table to {summary_path}")

    def _process_checkpoint(self, ckpt_path, pca_dim, conf1, conf2):
        model, T_fixed = self._load_model(ckpt_path)
        train_loader, test_loader = self._build_loaders(T_fixed)
        output_dir = self.output_root / ckpt_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        (
            train_losses,
            train_latents,
            train_labels,
            _,
            _,
        ) = self._collect_features(model, train_loader)
        (
            test_losses,
            test_latents,
            test_labels,
            test_per_band,
            test_band_latents,
        ) = self._collect_features(model, test_loader)

        metrics = self._two_stage_classify(
            train_losses,
            train_latents,
            train_labels,
            test_losses,
            test_latents,
            test_labels,
            pca_dim,
            conf1,
            conf2,
        )
        self._save_metrics(output_dir, metrics, ckpt_path.name)

        # PCA grid (LF/BP/HF/Merged)
        pca_embeddings = {}
        for name, lat in {**test_band_latents, "merged": test_latents}.items():
            lat_std = StandardScaler().fit_transform(lat)
            pca_embeddings[name] = PCA(n_components=2).fit_transform(lat_std)
        self._plot_latent_grid(
            pca_embeddings, test_labels, "PCA", output_dir / "pca.png"
        )

        # t-SNE grid (LF/BP/HF/Merged)
        tsne_embeddings = {}
        for name, lat in {**test_band_latents, "merged": test_latents}.items():
            lat_std = StandardScaler().fit_transform(lat)
            tsne_embeddings[name] = TSNE(
                n_components=2, perplexity=30, learning_rate="auto", init="pca"
            ).fit_transform(lat_std)
        self._plot_latent_grid(
            tsne_embeddings, test_labels, "t-SNE", output_dir / "t-sne.png"
        )

        # Reconstruction loss distributions
        self._plot_recon_losses(
            test_per_band,
            test_labels,
            output_dir / "reconstruction_losses.png",
        )

        saved_metrics = self._save_metrics(output_dir, metrics, ckpt_path.name)

        print(
            f"[{ckpt_path.name}] acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"saved at {output_dir}"
        )
        return saved_metrics


def main():
    checkpoints = sorted(Path("runs").glob("*.pt"))
    if not checkpoints:
        raise SystemExit("No checkpoints found under runs/")

    visualizer = ModelVisualizer(
        split_json="data_split.json",
        base_dir="test_balanced_npz",
        output_root="visualizations",
    )
    visualizer.run(checkpoints)


if __name__ == "__main__":
    main()
