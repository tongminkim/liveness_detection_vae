"""
Test Balanced 데이터셋으로 Trial 평가 및 시각화
- fo와 fsgan 제외
- Real과 Fake를 같은 수로 맞춰서 평가
"""

import os
import sys
import json
import torch
import numpy as np
import random
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
TEST_REAL_BASE = "/home/elicer/liveness_detection/model1/test_balanced_npz/real"
TEST_FAKE_BASE = "/home/elicer/liveness_detection/model1/test_balanced_npz/fake"

# Fake directories to use (excluding fo and fsgan)
FAKE_DIRS = ["audio-driven", "dfl", "dffs"]


def collect_npz_files(base_dir, subdirs=None):
    """Collect all npz files from specified subdirectories"""
    import glob

    files = []
    if subdirs:
        for subdir in subdirs:
            pattern = os.path.join(base_dir, subdir, "**", "*.npz")
            files.extend(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(base_dir, "**", "*.npz")
        files.extend(glob.glob(pattern, recursive=True))

    return files


def create_temp_dataset_dir(files, output_dir):
    """Create temporary directory with symlinks to npz files"""
    os.makedirs(output_dir, exist_ok=True)

    # Create symlinks
    for i, src_file in enumerate(files):
        dst_file = os.path.join(output_dir, f"sample_{i:05d}.npz")
        if os.path.exists(dst_file):
            os.remove(dst_file)
        os.symlink(os.path.abspath(src_file), dst_file)

    return output_dir


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

    plt.hist(live_losses['total'], bins=bins, alpha=0.6, label='Real', color='blue')
    plt.hist(fake_losses['total'], bins=bins, alpha=0.6, label='Fake', color='red')

    plt.xlabel('Total Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Loss Distribution (Real vs Fake)')
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
                alpha=0.5, s=20, label='Real', color='blue')
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
               alpha=0.5, s=20, label='Real', color='blue')
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

    plt.hist(live_pc1, bins=bins, alpha=0.6, label='Real', color='blue')
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
                alpha=0.5, s=20, label='Real', color='blue')
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
    print(f"Using test_balanced_npz dataset (excluding fo and fsgan)")

    # Extract hyperparameters from directory name
    parts = os.path.basename(trial_dir).split('_')
    fc_low = float(parts[4])
    fc_high = float(parts[7])
    C_h = int(parts[10])
    C_z = int(parts[13])

    print(f"  fc_low={fc_low}, fc_high={fc_high}, C_h={C_h}, C_z={C_z}")

    # Create config
    config = FullFeatureConfig()
    config.T_fixed = 300  # Using T=300
    config.fc_low = fc_low
    config.fc_high = fc_high
    config.C_h = C_h
    config.C_z = C_z

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect files
    print("  Collecting test files...")
    fake_files = collect_npz_files(TEST_FAKE_BASE, FAKE_DIRS)
    real_files = collect_npz_files(TEST_REAL_BASE)

    print(f"  Found {len(fake_files)} fake samples (excluding fo, fsgan)")
    print(f"  Found {len(real_files)} real samples")

    # Limit to 100 samples each for faster processing
    random.seed(42)
    num_samples = 100
    if len(fake_files) > num_samples:
        fake_files = random.sample(fake_files, num_samples)
    if len(real_files) > num_samples:
        real_files = random.sample(real_files, num_samples)

    print(f"  Using {len(real_files)} real samples, {len(fake_files)} fake samples")

    # Create temporary dataset directories
    print("  Creating temporary dataset directories...")
    temp_real_dir = "/tmp/test_balanced_real"
    temp_fake_dir = "/tmp/test_balanced_fake"

    create_temp_dataset_dir(real_files, temp_real_dir)
    create_temp_dataset_dir(fake_files, temp_fake_dir)

    # Load datasets
    print("  Loading datasets...")
    test_real_dataset = LipLivenessBandDataset(
        data_dir=temp_real_dir,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order,
        random_crop=False  # Center crop for evaluation
    )

    test_fake_dataset = LipLivenessBandDataset(
        data_dir=temp_fake_dir,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order,
        random_crop=False  # Center crop for evaluation
    )

    test_real_loader = DataLoader(test_real_dataset, batch_size=64, shuffle=False)
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

    # Compute reconstruction losses
    print("  Computing reconstruction losses...")
    real_losses = compute_reconstruction_losses(model, test_real_loader, device)
    fake_losses = compute_reconstruction_losses(model, test_fake_loader, device)

    # Extract latent vectors
    print("  Extracting latent vectors...")
    real_latents = extract_latent_vectors(model, test_real_loader, device)
    fake_latents = extract_latent_vectors(model, test_fake_loader, device)

    # Create visualizations
    print("  Creating visualizations...")
    plot_reconstruction_loss_overall(real_losses, fake_losses,
        os.path.join(trial_dir, "test_balanced_reconstruction_loss_overall.png"))

    plot_reconstruction_loss_2d(real_losses, fake_losses,
        os.path.join(trial_dir, "test_balanced_reconstruction_loss_2d_hf_lf.png"))

    plot_reconstruction_loss_3d(real_losses, fake_losses,
        os.path.join(trial_dir, "test_balanced_reconstruction_loss_3d.png"))

    plot_latent_pca_pc1(real_latents, fake_latents,
        os.path.join(trial_dir, "test_balanced_latent_pca_pc1.png"))

    plot_latent_tsne_2d(real_latents, fake_latents,
        os.path.join(trial_dir, "test_balanced_latent_tsne_2d.png"))

    # Save metrics
    metrics = {
        'hyperparameters': {
            'fc_low': fc_low,
            'fc_high': fc_high,
            'C_h': C_h,
            'C_z': C_z,
            'T_fixed': 300
        },
        'test_real_loss_mean': float(real_losses['total'].mean()),
        'test_fake_loss_mean': float(fake_losses['total'].mean()),
        'separation': float(abs(real_losses['total'].mean() - fake_losses['total'].mean())),
        'num_real_samples': len(real_files),
        'num_fake_samples': len(fake_files)
    }

    with open(os.path.join(trial_dir, "test_balanced_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("  ✓ Saved test_balanced_metrics.json")
    print("\nDone!")

    # Cleanup temp directories
    import shutil
    shutil.rmtree(temp_real_dir, ignore_errors=True)
    shutil.rmtree(temp_fake_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        trial_dir = sys.argv[1]
    else:
        print("Usage: python visualize_test_balanced.py <trial_dir>")
        sys.exit(1)

    main(trial_dir)
