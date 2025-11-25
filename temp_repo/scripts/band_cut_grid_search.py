import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".matplotlib").resolve()))
(ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_bandvae import Config
from dataset_stage2 import Stage2Dataset
from model_bandvae import BandSplitVAE, band_split_vae_loss


def compute_bandwise_reconstruction_losses(model, loader, device):
    model.eval()
    losses_lf, losses_bp, losses_hf, labels = [], [], [], []

    with torch.no_grad():
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


def extract_latent_vectors(model, loader, device):
    model.eval()
    latents, labels = [], []
    per_band = {"lf": [], "bp": [], "hf": []}

    with torch.no_grad():
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
            per_band["lf"].append(z_lf.cpu().numpy())
            per_band["bp"].append(z_bp.cpu().numpy())
            per_band["hf"].append(z_hf.cpu().numpy())

    lat_all = np.vstack(latents)
    labels = np.array(labels)
    per_band = {k: np.vstack(v) for k, v in per_band.items()}
    return lat_all, labels, per_band


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


class BandCutGridSearch:
    def __init__(
        self,
        split_json="data_split.json",
        model_checkpoint="runs/method1_margin_best.pt",
        output_dir="visualizations/band_cut_grid",
        pca_dim=10,
        confidence_stage1=0.95,
        confidence_stage2=0.70,
        fc_low_values=None,
        fc_high_values=None,
        filter_order=4,
        batch_size=64,
        num_workers=0,
        max_pairs=None,
        epochs=10,
        lr=1e-4,
        train_real_only=False,
        pretrained_path="runs/stage1_pretrained.pt",
        use_pretrained=False,
        max_train_batches=None,
        device=None,
    ):
        self.split_json = (
            Path(split_json) if os.path.isabs(split_json) else ROOT / split_json
        )
        self.model_checkpoint = (
            Path(model_checkpoint)
            if os.path.isabs(model_checkpoint)
            else ROOT / model_checkpoint
        )
        self.output_dir = (
            Path(output_dir) if os.path.isabs(output_dir) else ROOT / output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pca_dim = pca_dim
        self.confidence_stage1 = confidence_stage1
        self.confidence_stage2 = confidence_stage2
        self.filter_order = filter_order
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_pairs = max_pairs
        self.epochs = epochs
        self.lr = lr
        self.train_real_only = train_real_only
        self.pretrained_path = (
            Path(pretrained_path)
            if os.path.isabs(pretrained_path)
            else ROOT / pretrained_path
        )
        self.use_pretrained = use_pretrained
        self.max_train_batches = max_train_batches
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fc_low_values = (
            fc_low_values
            if fc_low_values is not None
            else [1.0, 2.0, 3.0, 4.0]
        )
        self.fc_high_values = (
            fc_high_values
            if fc_high_values is not None
            else [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )

        self.config = Config(mode="full")
        self.config.T_fixed = 300
        self.dataset_base = ROOT / "test_balanced_npz"

    def _load_model(self):
        model = BandSplitVAE(
            C_in_per_band=self.config.C_in_per_band,
            C_h=self.config.C_h,
            C_z=self.config.C_z,
            dilations=self.config.dilations,
        ).to(self.device)

        if self.use_pretrained and self.pretrained_path.exists():
            from torch.serialization import add_safe_globals

            add_safe_globals([Config])
            checkpoint = torch.load(
                self.pretrained_path, map_location=self.device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"  Loaded pretrained weights from {self.pretrained_path}")
        else:
            print("  Starting from random init (no pretrained weights used)")

        model.train()
        return model

    def _train_for_pair(self, model, loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_batches = 0
            for batch_idx, (x_lf, x_bp, x_hf, labels) in enumerate(loader):
                if self.train_real_only:
                    mask = labels == 0
                    if not mask.any():
                        continue
                    x_lf, x_bp, x_hf = x_lf[mask], x_bp[mask], x_hf[mask]
                x_lf = x_lf.to(self.device, non_blocking=True)
                x_bp = x_bp.to(self.device, non_blocking=True)
                x_hf = x_hf.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)
                loss, _ = band_split_vae_loss(
                    recons,
                    mus,
                    logvars,
                    {"lf": x_lf, "bp": x_bp, "hf": x_hf},
                    x_hat_fused,
                    None,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1
                if (
                    self.max_train_batches is not None
                    and total_batches >= self.max_train_batches
                ):
                    break
            if total_batches > 0:
                avg_loss = total_loss / total_batches
                print(f"  Epoch {epoch+1}/{self.epochs} - loss: {avg_loss:.4f}")
        model.eval()
        return model

    def _build_loader(self, split_name, fc_low, fc_high, shuffle):
        dataset = Stage2Dataset(
            split_json_path=self.split_json,
            split_name=split_name,
            T_fixed=self.config.T_fixed,
            fps=self.config.fps,
            use_acceleration=self.config.use_acceleration,
            use_angle=self.config.use_angle,
            use_angle_rate=self.config.use_angle_rate,
            fc_low=fc_low,
            fc_high=fc_high,
            filter_order=self.filter_order,
            random_crop=False,
            base_dir=self.dataset_base,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _two_stage_filter(self, losses_3d, latents_pca, labels, loss_mu, loss_cov, latent_mu, latent_cov):
        preds = []
        stage1_pass = 0
        stage2_pass = 0

        for loss_vec, latent_vec in zip(losses_3d, latents_pca):
            if within_confidence(loss_vec, loss_mu, loss_cov, self.confidence_stage1):
                stage1_pass += 1
                if within_confidence(
                    latent_vec, latent_mu, latent_cov, self.confidence_stage2
                ):
                    stage2_pass += 1
                    preds.append(0)
                else:
                    preds.append(1)
            else:
                preds.append(1)

        preds = np.array(preds)
        cm = confusion_matrix(labels, preds)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "predictions": preds,
            "confusion_matrix": cm,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "stage1_pass_rate": stage1_pass / len(labels),
            "stage2_pass_rate": stage2_pass / len(labels),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    def _save_tsne(self, embeddings, labels, fc_low, fc_high, out_dir):
        paths = {}
        palette = {0: "#1f77b4", 1: "#d62728"}

        for name, latent in embeddings.items():
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate="auto",
                init="random",
                random_state=42,
            )
            emb = tsne.fit_transform(latent)

            plt.figure(figsize=(8, 6))
            for label_value in [0, 1]:
                mask = labels == label_value
                plt.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    s=18,
                    alpha=0.7,
                    label="Real" if label_value == 0 else "Fake",
                    color=palette[label_value],
                )

            title_band = "All Bands" if name == "all" else name.upper()
            plt.title(
                f"t-SNE ({title_band}) fc_low={fc_low}Hz fc_high={fc_high}Hz",
                fontsize=12,
            )
            plt.legend()
            plt.tight_layout()
            out_path = out_dir / f"tsne_{name}_fc{fc_low:.1f}_{fc_high:.1f}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            paths[name] = str(out_path)

        return paths

    def _compute_pca2(self, latent):
        # Apply PCA directly without StandardScaler to preserve scale information
        pca2 = PCA(n_components=2)
        return pca2.fit_transform(latent)

    def _save_pca_grid(self, embeddings, labels, fc_low, fc_high, out_dir):
        names = list(embeddings.keys())
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        palette = {0: "#1f77b4", 1: "#d62728"}

        for ax, name in zip(axes, names):
            latent = embeddings[name]
            try:
                emb2 = self._compute_pca2(latent)
            except Exception:
                emb2 = np.zeros((latent.shape[0], 2))
            for label_value in [0, 1]:
                mask = labels == label_value
                ax.scatter(
                    emb2[mask, 0],
                    emb2[mask, 1],
                    s=18,
                    alpha=0.7,
                    label="Real" if label_value == 0 else "Fake",
                    color=palette[label_value],
                )
            title_band = "All Bands" if name == "all" else name.upper()
            ax.set_title(f"PCA2 ({title_band})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="upper center", ncol=2)
        fig.suptitle(f"PCA projections fc_low={fc_low}Hz fc_high={fc_high}Hz", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f"pca_grid_fc{fc_low:.1f}_{fc_high:.1f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def _evaluate_pair(self, fc_low, fc_high):
        model = self._load_model()

        train_loader = self._build_loader("stage2_train", fc_low, fc_high, shuffle=True)
        test_loader = self._build_loader("final_test", fc_low, fc_high, shuffle=False)

        model = self._train_for_pair(model, train_loader)

        train_losses, train_labels = compute_bandwise_reconstruction_losses(
            model, train_loader, self.device
        )
        train_latents, _, train_per_band = extract_latent_vectors(
            model, train_loader, self.device
        )

        test_losses, test_labels = compute_bandwise_reconstruction_losses(
            model, test_loader, self.device
        )
        test_latents, _, test_per_band = extract_latent_vectors(
            model, test_loader, self.device
        )

        # Clean numeric issues before PCA/Gaussian
        train_losses = np.clip(
            np.nan_to_num(train_losses, copy=False, posinf=1e6, neginf=-1e6),
            -1e4,
            1e4,
        )
        test_losses = np.clip(
            np.nan_to_num(test_losses, copy=False, posinf=1e6, neginf=-1e6),
            -1e4,
            1e4,
        )
        train_latents = np.clip(
            np.nan_to_num(train_latents, copy=False, posinf=1e2, neginf=-1e2),
            -1e2,
            1e2,
        )
        test_latents = np.clip(
            np.nan_to_num(test_latents, copy=False, posinf=1e2, neginf=-1e2),
            -1e2,
            1e2,
        )
        train_per_band = {
            k: np.clip(
                np.nan_to_num(v, copy=False, posinf=1e2, neginf=-1e2), -1e2, 1e2
            )
            for k, v in train_per_band.items()
        }
        test_per_band = {
            k: np.clip(
                np.nan_to_num(v, copy=False, posinf=1e2, neginf=-1e2), -1e2, 1e2
            )
            for k, v in test_per_band.items()
        }

        real_mask = train_labels == 0
        loss_mu, loss_cov = fit_gaussian(train_losses[real_mask])

        # Apply PCA (NO StandardScaler - preserves scale information)
        pca = PCA(n_components=self.pca_dim)
        train_latents_pca = pca.fit_transform(train_latents[real_mask])
        test_latents_pca = pca.transform(test_latents)
        latent_mu, latent_cov = fit_gaussian(train_latents_pca)

        results = self._two_stage_filter(
            test_losses,
            test_latents_pca,
            test_labels,
            loss_mu,
            loss_cov,
            latent_mu,
            latent_cov,
        )
        # Ensure numpy arrays for downstream serialization
        results["confusion_matrix"] = np.asarray(results["confusion_matrix"])
        results["predictions"] = np.asarray(results["predictions"])

        pair_dir = self.output_dir / f"fc_{fc_low:.1f}_{fc_high:.1f}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Save trained model checkpoint
        model_save_path = pair_dir / "model_best.pt"
        torch.save({
            'fc_low': fc_low,
            'fc_high': fc_high,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'accuracy': results['accuracy'],
            'f1': results['f1'],
        }, model_save_path)
        print(f"  Saved model to {model_save_path}")

        tsne_paths = self._save_tsne(
            {
                "all": test_latents_pca,
                "lf": test_per_band["lf"],
                "bp": test_per_band["bp"],
                "hf": test_per_band["hf"],
            },
            test_labels,
            fc_low,
            fc_high,
            pair_dir,
        )
        pca_grid_path = self._save_pca_grid(
            {
                "all": test_latents,
                "lf": test_per_band["lf"],
                "bp": test_per_band["bp"],
                "hf": test_per_band["hf"],
            },
            test_labels,
            fc_low,
            fc_high,
            pair_dir,
        )

        results.update(
            {
                "fc_low": fc_low,
                "fc_high": fc_high,
                "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                "tsne_path_all": tsne_paths.get("all"),
                "tsne_path_lf": tsne_paths.get("lf"),
                "tsne_path_bp": tsne_paths.get("bp"),
                "tsne_path_hf": tsne_paths.get("hf"),
                "pca_grid_path": str(pca_grid_path),
                "confusion_matrix": results["confusion_matrix"].tolist(),
                "tn": int(results["tn"]),
                "fp": int(results["fp"]),
                "fn": int(results["fn"]),
                "tp": int(results["tp"]),
            }
        )
        # Prepare metrics for JSON (convert arrays to lists)
        metrics_save = dict(results)
        preds_arr = np.asarray(results["predictions"]).ravel()
        metrics_save["predictions"] = [int(x) for x in preds_arr.tolist()]
        cm_arr = np.asarray(results["confusion_matrix"])
        metrics_save["confusion_matrix"] = cm_arr.astype(int).tolist()

        with open(pair_dir / "metrics.json", "w") as f:
            json.dump(metrics_save, f, indent=2)
        return results

    def _save_lookup(self, results):
        csv_path = self.output_dir / "lookup_table.csv"
        json_path = self.output_dir / "lookup_table.json"

        fieldnames = [
            "fc_low",
            "fc_high",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "stage1_pass_rate",
            "stage2_pass_rate",
            "explained_variance",
            "tsne_path_all",
            "tsne_path_lf",
            "tsne_path_bp",
            "tsne_path_hf",
            "pca_grid_path",
            "tn",
            "fp",
            "fn",
            "tp",
        ]

        sanitized = []
        for row in results:
            row_copy = dict(row)
            row_copy["predictions"] = [int(x) for x in np.asarray(row.get("predictions", [])).ravel().tolist()]
            if isinstance(row_copy.get("confusion_matrix"), np.ndarray):
                row_copy["confusion_matrix"] = row_copy["confusion_matrix"].astype(int).tolist()
            sanitized.append(row_copy)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sanitized:
                writer.writerow({k: row.get(k) for k in fieldnames})

        with open(json_path, "w") as f:
            json.dump(sanitized, f, indent=2)

        return csv_path, json_path

    def _plot_heatmap(self, results, metric="accuracy"):
        grid = {}
        for row in results:
            grid.setdefault(row["fc_low"], {})[row["fc_high"]] = row[metric]

        fc_low_vals = sorted(grid.keys())
        fc_high_vals = sorted({h for v in grid.values() for h in v.keys()})
        matrix = np.full((len(fc_low_vals), len(fc_high_vals)), np.nan)

        for i, fc_low in enumerate(fc_low_vals):
            for j, fc_high in enumerate(fc_high_vals):
                if fc_high in grid.get(fc_low, {}):
                    matrix[i, j] = grid[fc_low][fc_high]

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            xticklabels=[f"{v:.1f}" for v in fc_high_vals],
            yticklabels=[f"{v:.1f}" for v in fc_low_vals],
            cmap="viridis",
        )
        plt.xlabel("fc_high (Hz)")
        plt.ylabel("fc_low (Hz)")
        plt.title(f"{metric.title()} Heatmap")
        plt.tight_layout()
        heatmap_path = self.output_dir / f"{metric}_heatmap.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        return heatmap_path

    def run(self):
        results = []
        evaluated = 0
        for fc_low in self.fc_low_values:
            for fc_high in self.fc_high_values:
                if fc_high <= fc_low:
                    continue
                print(f"Running fc_low={fc_low}Hz, fc_high={fc_high}Hz...")
                results.append(self._evaluate_pair(fc_low, fc_high))
                evaluated += 1
                if self.max_pairs is not None and evaluated >= self.max_pairs:
                    break
            if self.max_pairs is not None and evaluated >= self.max_pairs:
                break

        csv_path, json_path = self._save_lookup(results)
        heatmap_path = self._plot_heatmap(results, metric="accuracy")

        print("\nLookup table saved:")
        print(f"- CSV:  {csv_path}")
        print(f"- JSON: {json_path}")
        print(f"- Heatmap: {heatmap_path}")
        print("t-SNE plots stored per pair under:", self.output_dir)
        return results


def main():
    parser = argparse.ArgumentParser(description="Grid search over band cutoffs.")
    parser.add_argument(
        "--fc-lows",
        type=str,
        default=None,
        help="Comma-separated fc_low values (Hz). Example: 1,2,3",
    )
    parser.add_argument(
        "--fc-highs",
        type=str,
        default=None,
        help="Comma-separated fc_high values (Hz). Example: 6,8,10",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on number of pairs to evaluate (for quick sanity checks).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Dataloader batch size."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers (use 0 in restricted environments).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/band_cut_grid",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per cutoff pair (default: 10).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning per pair.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Use both real and fake samples during fine-tuning (default: real only).",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="runs/stage1_pretrained.pt",
        help="Path to baseline checkpoint used for each pair.",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Load weights from --pretrained-path instead of random init.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on batches per epoch (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g., cuda, cuda:0, mps, cpu). Defaults to cuda if available.",
    )
    args = parser.parse_args()

    def _parse_float_list(val):
        if val is None:
            return None
        return [float(x) for x in val.split(",") if x.strip()]

    search = BandCutGridSearch(
        fc_low_values=_parse_float_list(args.fc_lows),
        fc_high_values=_parse_float_list(args.fc_highs),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_pairs=args.max_pairs,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        train_real_only=not args.train_all,
        pretrained_path=args.pretrained_path,
        use_pretrained=args.use_pretrained,
        max_train_batches=args.max_train_batches,
        device=args.device,
    )
    search.run()


if __name__ == "__main__":
    main()
