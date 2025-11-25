"""
기존 학습된 trial의 평가 및 시각화 수행
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset_bandvae import LipLivenessBandDataset
from model_bandvae import BandSplitVAE
from config_bandvae import FullFeatureConfig


# Paths
TEST_LIVE_DIR = "/home/elicer/liveness_detection/model1/processed_live_test"
TEST_FAKE_DIR = "/home/elicer/liveness_detection/model1/processed_fake_test"


@torch.no_grad()
def validate(model, loader, config, device):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    n_batches = len(loader)

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        from model_bandvae import band_split_vae_loss
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        loss, loss_dict = band_split_vae_loss(
            recons, mus, logvars, targets,
            x_hat_fused, None, betas=betas, alpha_fusion=0.0
        )

        total_loss += loss_dict['total']

    return {'total': total_loss / n_batches}


@torch.no_grad()
def compute_reconstruction_losses(model, loader, device):
    """Compute per-sample reconstruction losses"""
    model.eval()

    total_losses = []
    lf_losses = []
    bp_losses = []
    hf_losses = []

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        recons, _, _, _ = model(x_lf, x_bp, x_hf)

        loss_lf = torch.mean(torch.abs(recons['lf'] - x_lf), dim=[1, 2])
        loss_bp = torch.mean(torch.abs(recons['bp'] - x_bp), dim=[1, 2])
        loss_hf = torch.mean(torch.abs(recons['hf'] - x_hf), dim=[1, 2])
        loss_total = loss_lf + loss_bp + loss_hf

        total_losses.append(loss_total.cpu().numpy())
        lf_losses.append(loss_lf.cpu().numpy())
        bp_losses.append(loss_bp.cpu().numpy())
        hf_losses.append(loss_hf.cpu().numpy())

    return {
        'total': np.concatenate(total_losses),
        'lf': np.concatenate(lf_losses),
        'bp': np.concatenate(bp_losses),
        'hf': np.concatenate(hf_losses)
    }


@torch.no_grad()
def extract_latent_vectors(model, loader, device):
    """Extract latent vectors"""
    model.eval()

    latents_lf = []
    latents_bp = []
    latents_hf = []

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        _, mus, _, _ = model(x_lf, x_bp, x_hf)

        # Ensure 2D: (batch, C_z)
        mu_lf = mus['lf'].cpu().numpy()
        mu_bp = mus['bp'].cpu().numpy()
        mu_hf = mus['hf'].cpu().numpy()

        if mu_lf.ndim > 2:
            mu_lf = mu_lf.squeeze()
        if mu_bp.ndim > 2:
            mu_bp = mu_bp.squeeze()
        if mu_hf.ndim > 2:
            mu_hf = mu_hf.squeeze()

        latents_lf.append(mu_lf)
        latents_bp.append(mu_bp)
        latents_hf.append(mu_hf)

    return {
        'lf': np.concatenate(latents_lf, axis=0),
        'bp': np.concatenate(latents_bp, axis=0),
        'hf': np.concatenate(latents_hf, axis=0)
    }


def plot_reconstruction_loss_overall(live_losses, fake_losses, save_path):
    """Histogram of total reconstruction losses"""
    plt.figure(figsize=(10, 6))

    bins = np.linspace(
        min(live_losses['total'].min(), fake_losses['total'].min()),
        max(live_losses['total'].max(), fake_losses['total'].max()),
        50
    )

    plt.hist(live_losses['total'], bins=bins, alpha=0.6, label='Live', color='blue')
    plt.hist(fake_losses['total'], bins=bins, alpha=0.6, label='Fake', color='red')

    plt.xlabel('Total Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Loss Distribution (Live vs Fake)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_reconstruction_loss_2d(live_losses, fake_losses, save_path):
    """2D scatter: HF vs LF"""
    plt.figure(figsize=(10, 8))

    plt.scatter(live_losses['lf'], live_losses['hf'],
                alpha=0.5, s=20, label='Live', color='blue')
    plt.scatter(fake_losses['lf'], fake_losses['hf'],
                alpha=0.5, s=20, label='Fake', color='red')

    plt.xlabel('LF Reconstruction Loss')
    plt.ylabel('HF Reconstruction Loss')
    plt.title('2D Reconstruction Loss (HF vs LF)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_reconstruction_loss_3d(live_losses, fake_losses, save_path):
    """3D scatter: HF, BP, LF"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(live_losses['lf'], live_losses['bp'], live_losses['hf'],
               alpha=0.5, s=20, label='Live', color='blue')
    ax.scatter(fake_losses['lf'], fake_losses['bp'], fake_losses['hf'],
               alpha=0.5, s=20, label='Fake', color='red')

    ax.set_xlabel('LF Reconstruction Loss')
    ax.set_ylabel('BP Reconstruction Loss')
    ax.set_zlabel('HF Reconstruction Loss')
    ax.set_title('3D Reconstruction Loss (LF, BP, HF)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_latent_pca_pc1(live_latents, fake_latents, save_path):
    """PCA PC1 histogram"""
    # Ensure 2D and concatenate
    live_lf = live_latents['lf'].reshape(len(live_latents['lf']), -1)
    live_bp = live_latents['bp'].reshape(len(live_latents['bp']), -1)
    live_hf = live_latents['hf'].reshape(len(live_latents['hf']), -1)
    live_concat = np.concatenate([live_lf, live_bp, live_hf], axis=1)

    fake_lf = fake_latents['lf'].reshape(len(fake_latents['lf']), -1)
    fake_bp = fake_latents['bp'].reshape(len(fake_latents['bp']), -1)
    fake_hf = fake_latents['hf'].reshape(len(fake_latents['hf']), -1)
    fake_concat = np.concatenate([fake_lf, fake_bp, fake_hf], axis=1)

    all_data = np.concatenate([live_concat, fake_concat], axis=0)
    pca = PCA(n_components=1)
    pca.fit(all_data)

    live_pc1 = pca.transform(live_concat)[:, 0]
    fake_pc1 = pca.transform(fake_concat)[:, 0]

    plt.figure(figsize=(10, 6))

    bins = np.linspace(
        min(live_pc1.min(), fake_pc1.min()),
        max(live_pc1.max(), fake_pc1.max()),
        50
    )

    plt.hist(live_pc1, bins=bins, alpha=0.6, label='Live', color='blue')
    plt.hist(fake_pc1, bins=bins, alpha=0.6, label='Fake', color='red')

    plt.xlabel('PC1')
    plt.ylabel('Frequency')
    plt.title(f'Latent Space PCA (PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_latent_tsne_2d(live_latents, fake_latents, save_path):
    """t-SNE 2D visualization"""
    # Ensure 2D and concatenate
    live_lf = live_latents['lf'].reshape(len(live_latents['lf']), -1)
    live_bp = live_latents['bp'].reshape(len(live_latents['bp']), -1)
    live_hf = live_latents['hf'].reshape(len(live_latents['hf']), -1)
    live_concat = np.concatenate([live_lf, live_bp, live_hf], axis=1)

    fake_lf = fake_latents['lf'].reshape(len(fake_latents['lf']), -1)
    fake_bp = fake_latents['bp'].reshape(len(fake_latents['bp']), -1)
    fake_hf = fake_latents['hf'].reshape(len(fake_latents['hf']), -1)
    fake_concat = np.concatenate([fake_lf, fake_bp, fake_hf], axis=1)

    all_data = np.concatenate([live_concat, fake_concat], axis=0)
    labels = np.array([0] * len(live_concat) + [1] * len(fake_concat))

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data)//4))
    embedding = tsne.fit_transform(all_data)

    plt.figure(figsize=(10, 8))

    live_idx = labels == 0
    fake_idx = labels == 1

    plt.scatter(embedding[live_idx, 0], embedding[live_idx, 1],
                alpha=0.5, s=20, label='Live', color='blue')
    plt.scatter(embedding[fake_idx, 0], embedding[fake_idx, 1],
                alpha=0.5, s=20, label='Fake', color='red')

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Latent Space t-SNE (2D)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def main(trial_dir):
    """Main function"""

    print(f"Evaluating and visualizing: {trial_dir}")

    # Extract hyperparameters from directory name
    # Format: trial_000_fc_low_1.5_fc_high_7.0_C_h_96_C_z_12
    parts = os.path.basename(trial_dir).split('_')
    fc_low = float(parts[4])
    fc_high = float(parts[7])
    C_h = int(parts[10])
    C_z = int(parts[13])

    print(f"  fc_low={fc_low}, fc_high={fc_high}, C_h={C_h}, C_z={C_z}")

    # Create config
    config = FullFeatureConfig()
    config.fc_low = fc_low
    config.fc_high = fc_high
    config.C_h = C_h
    config.C_z = C_z

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    print("  Loading datasets...")
    test_live_dataset = LipLivenessBandDataset(
        data_dir=TEST_LIVE_DIR,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order
    )

    test_fake_dataset = LipLivenessBandDataset(
        data_dir=TEST_FAKE_DIR,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order
    )

    test_live_loader = DataLoader(test_live_dataset, batch_size=64, shuffle=False)
    test_fake_loader = DataLoader(test_fake_dataset, batch_size=64, shuffle=False)

    # Load model
    print("  Loading model...")
    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)

    checkpoint = torch.load(os.path.join(trial_dir, "best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    print("  Evaluating...")
    test_live_loss = validate(model, test_live_loader, config, device)
    test_fake_loss = validate(model, test_fake_loader, config, device)

    print(f"  Live loss: {test_live_loss['total']:.4f}")
    print(f"  Fake loss: {test_fake_loss['total']:.4f}")

    # Compute reconstruction losses
    print("  Computing reconstruction losses...")
    live_losses = compute_reconstruction_losses(model, test_live_loader, device)
    fake_losses = compute_reconstruction_losses(model, test_fake_loader, device)

    # Extract latent vectors
    print("  Extracting latent vectors...")
    live_latents = extract_latent_vectors(model, test_live_loader, device)
    fake_latents = extract_latent_vectors(model, test_fake_loader, device)

    # Create visualizations
    print("  Creating visualizations...")
    plot_reconstruction_loss_overall(live_losses, fake_losses,
        os.path.join(trial_dir, "reconstruction_loss_overall.png"))

    plot_reconstruction_loss_2d(live_losses, fake_losses,
        os.path.join(trial_dir, "reconstruction_loss_2d_hf_lf.png"))

    plot_reconstruction_loss_3d(live_losses, fake_losses,
        os.path.join(trial_dir, "reconstruction_loss_3d.png"))

    plot_latent_pca_pc1(live_latents, fake_latents,
        os.path.join(trial_dir, "latent_pca_pc1.png"))

    plot_latent_tsne_2d(live_latents, fake_latents,
        os.path.join(trial_dir, "latent_tsne_2d.png"))

    # Save metrics
    metrics = {
        'hyperparameters': {
            'fc_low': fc_low,
            'fc_high': fc_high,
            'C_h': C_h,
            'C_z': C_z
        },
        'test_live_loss': test_live_loss['total'],
        'test_fake_loss': test_fake_loss['total'],
        'separation': abs(test_live_loss['total'] - test_fake_loss['total'])
    }

    with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("  ✓ Saved metrics.json")
    print("\nDone!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        trial_dir = sys.argv[1]
    else:
        trial_dir = "runs/hyperparameter_tuning/trial_000_fc_low_1.5_fc_high_7.0_C_h_96_C_z_12"

    main(trial_dir)
