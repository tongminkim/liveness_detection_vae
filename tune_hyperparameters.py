"""
Hyperparameter Tuning for Band-Split VAE using Optuna

Strategy: Option 2 (Efficient with Pruning)
- n_trials: 15
- epochs_per_trial: 10
- pruning: MedianPruner
- target: 10 hours
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
from optuna.pruners import MedianPruner

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Local imports
from dataset_bandvae import LipLivenessBandDataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import FullFeatureConfig


# ============================================
# Global Configuration
# ============================================

# Paths
TRAIN_DIR = "/home/elicer/liveness_detection/model1/processed_live"
TEST_LIVE_DIR = "/home/elicer/liveness_detection/model1/processed_live_test"
TEST_FAKE_DIR = "/home/elicer/liveness_detection/model1/processed_fake_test"
OUTPUT_DIR = "/home/elicer/liveness_detection/model1/runs/hyperparameter_tuning"

# Tuning parameters
N_TRIALS = 15
EPOCHS_PER_TRIAL = 10
BATCH_SIZE = 64
NUM_WORKERS = 0  # Changed from 8 to 0 to avoid multiprocessing hang

# Pruning
USE_PRUNING = True
PRUNING_STARTUP_TRIALS = 5
PRUNING_WARMUP_STEPS = 3


# ============================================
# Logging Setup
# ============================================

def setup_logging(output_dir):
    """Setup global logging"""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "study.log")

    # Create logger
    logger = logging.getLogger("optuna_tuning")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger = setup_logging(OUTPUT_DIR)


# ============================================
# Training Functions
# ============================================

def train_epoch(model, loader, optimizer, config, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    recon_lf_loss = 0.0
    recon_bp_loss = 0.0
    recon_hf_loss = 0.0
    kl_lf_loss = 0.0
    kl_bp_loss = 0.0
    kl_hf_loss = 0.0

    n_batches = len(loader)

    for batch_idx, (x_lf, x_bp, x_hf) in enumerate(loader):
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        # Prepare targets
        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}

        # Compute loss
        betas = {
            'lf': config.beta_lf,
            'bp': config.beta_bp,
            'hf': config.beta_hf
        }

        loss, loss_dict = band_split_vae_loss(
            recons, mus, logvars,
            targets,
            x_hat_fused, None,
            betas=betas,
            alpha_fusion=0.0
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss_dict['total']
        recon_lf_loss += loss_dict['recon_lf']
        recon_bp_loss += loss_dict['recon_bp']
        recon_hf_loss += loss_dict['recon_hf']
        kl_lf_loss += loss_dict['kl_lf']
        kl_bp_loss += loss_dict['kl_bp']
        kl_hf_loss += loss_dict['kl_hf']

    return {
        'total': total_loss / n_batches,
        'recon_lf': recon_lf_loss / n_batches,
        'recon_bp': recon_bp_loss / n_batches,
        'recon_hf': recon_hf_loss / n_batches,
        'kl_lf': kl_lf_loss / n_batches,
        'kl_bp': kl_bp_loss / n_batches,
        'kl_hf': kl_hf_loss / n_batches
    }


@torch.no_grad()
def validate(model, loader, config, device):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    recon_lf_loss = 0.0
    recon_bp_loss = 0.0
    recon_hf_loss = 0.0
    kl_lf_loss = 0.0
    kl_bp_loss = 0.0
    kl_hf_loss = 0.0

    n_batches = len(loader)

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        # Prepare targets
        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}

        # Compute loss
        betas = {
            'lf': config.beta_lf,
            'bp': config.beta_bp,
            'hf': config.beta_hf
        }

        loss, loss_dict = band_split_vae_loss(
            recons, mus, logvars,
            targets,
            x_hat_fused, None,
            betas=betas,
            alpha_fusion=0.0
        )

        # Accumulate
        total_loss += loss_dict['total']
        recon_lf_loss += loss_dict['recon_lf']
        recon_bp_loss += loss_dict['recon_bp']
        recon_hf_loss += loss_dict['recon_hf']
        kl_lf_loss += loss_dict['kl_lf']
        kl_bp_loss += loss_dict['kl_bp']
        kl_hf_loss += loss_dict['kl_hf']

    return {
        'total': total_loss / n_batches,
        'recon_lf': recon_lf_loss / n_batches,
        'recon_bp': recon_bp_loss / n_batches,
        'recon_hf': recon_hf_loss / n_batches,
        'kl_lf': kl_lf_loss / n_batches,
        'kl_bp': kl_bp_loss / n_batches,
        'kl_hf': kl_hf_loss / n_batches
    }


# ============================================
# Evaluation Functions
# ============================================

@torch.no_grad()
def compute_reconstruction_losses(model, loader, device):
    """
    Compute per-sample reconstruction losses for visualization

    Returns:
        dict: {
            'total': np.array of shape (N,)
            'lf': np.array of shape (N,)
            'bp': np.array of shape (N,)
            'hf': np.array of shape (N,)
        }
    """
    model.eval()

    total_losses = []
    lf_losses = []
    bp_losses = []
    hf_losses = []

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, _, _, _ = model(x_lf, x_bp, x_hf)

        # Per-sample L1 loss
        loss_lf = torch.mean(torch.abs(recons['lf'] - x_lf), dim=[1, 2])  # (B,)
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
    """
    Extract latent vectors (mean) from encoder

    Returns:
        dict: {
            'lf': np.array of shape (N, C_z)
            'bp': np.array of shape (N, C_z)
            'hf': np.array of shape (N, C_z)
        }
    """
    model.eval()

    latents_lf = []
    latents_bp = []
    latents_hf = []

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        _, mus, _, _ = model(x_lf, x_bp, x_hf)

        # Extract mean vectors
        latents_lf.append(mus['lf'].cpu().numpy())  # (B, C_z)
        latents_bp.append(mus['bp'].cpu().numpy())
        latents_hf.append(mus['hf'].cpu().numpy())

    return {
        'lf': np.concatenate(latents_lf, axis=0),
        'bp': np.concatenate(latents_bp, axis=0),
        'hf': np.concatenate(latents_hf, axis=0)
    }


# ============================================
# Visualization Functions
# ============================================

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


def plot_latent_pca_pc1(live_latents, fake_latents, save_path):
    """PCA PC1 histogram"""
    # Concatenate all bands
    live_concat = np.concatenate([
        live_latents['lf'], live_latents['bp'], live_latents['hf']
    ], axis=1)  # (N, 3*C_z)

    fake_concat = np.concatenate([
        fake_latents['lf'], fake_latents['bp'], fake_latents['hf']
    ], axis=1)

    # PCA
    all_data = np.concatenate([live_concat, fake_concat], axis=0)
    pca = PCA(n_components=1)
    pca.fit(all_data)

    live_pc1 = pca.transform(live_concat)[:, 0]
    fake_pc1 = pca.transform(fake_concat)[:, 0]

    # Plot
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


def plot_latent_tsne_2d(live_latents, fake_latents, save_path):
    """t-SNE 2D visualization"""
    # Concatenate all bands
    live_concat = np.concatenate([
        live_latents['lf'], live_latents['bp'], live_latents['hf']
    ], axis=1)

    fake_concat = np.concatenate([
        fake_latents['lf'], fake_latents['bp'], fake_latents['hf']
    ], axis=1)

    # t-SNE
    all_data = np.concatenate([live_concat, fake_concat], axis=0)
    labels = np.array([0] * len(live_concat) + [1] * len(fake_concat))

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data)//4))
    embedding = tsne.fit_transform(all_data)

    # Plot
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


def plot_training_curves(train_losses, val_losses, save_path):
    """Training and validation curves"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    epochs = range(1, len(train_losses) + 1)

    # Total loss
    axes[0].plot(epochs, [x['total'] for x in train_losses], 'b-', label='Train')
    axes[0].plot(epochs, [x['total'] for x in val_losses], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Per-band reconstruction
    axes[1].plot(epochs, [x['recon_lf'] for x in train_losses], 'b-', label='LF')
    axes[1].plot(epochs, [x['recon_bp'] for x in train_losses], 'g-', label='BP')
    axes[1].plot(epochs, [x['recon_hf'] for x in train_losses], 'r-', label='HF')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Per-Band Reconstruction Loss (Train)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_all_visualizations(model, live_loader, fake_loader, device,
                                train_losses, val_losses, output_dir):
    """Create all 6 visualizations"""
    logger.info("  Generating visualizations...")

    # 1. Compute reconstruction losses
    live_losses = compute_reconstruction_losses(model, live_loader, device)
    fake_losses = compute_reconstruction_losses(model, fake_loader, device)

    # 2. Extract latent vectors
    live_latents = extract_latent_vectors(model, live_loader, device)
    fake_latents = extract_latent_vectors(model, fake_loader, device)

    # 3. Create visualizations
    plot_reconstruction_loss_overall(
        live_losses, fake_losses,
        os.path.join(output_dir, "reconstruction_loss_overall.png")
    )

    plot_reconstruction_loss_2d(
        live_losses, fake_losses,
        os.path.join(output_dir, "reconstruction_loss_2d_hf_lf.png")
    )

    plot_reconstruction_loss_3d(
        live_losses, fake_losses,
        os.path.join(output_dir, "reconstruction_loss_3d.png")
    )

    plot_latent_pca_pc1(
        live_latents, fake_latents,
        os.path.join(output_dir, "latent_pca_pc1.png")
    )

    plot_latent_tsne_2d(
        live_latents, fake_latents,
        os.path.join(output_dir, "latent_tsne_2d.png")
    )

    plot_training_curves(
        train_losses, val_losses,
        os.path.join(output_dir, "training_curves.png")
    )

    logger.info("  ✓ All visualizations created")


# ============================================
# Optuna Objective Function
# ============================================

def objective(trial):
    """Optuna objective function"""

    trial_start_time = time.time()

    # ========================================
    # 1. Suggest Hyperparameters
    # ========================================

    fc_low = trial.suggest_categorical('fc_low', [1.5, 2.0, 2.5])
    fc_high = trial.suggest_categorical('fc_high', [7.0, 8.0, 9.0])
    C_h = trial.suggest_categorical('C_h', [48, 64, 96])
    C_z = trial.suggest_categorical('C_z', [12, 16])

    # ========================================
    # 2. Create Trial Directory
    # ========================================

    trial_name = f"trial_{trial.number:03d}_fc_low_{fc_low}_fc_high_{fc_high}_C_h_{C_h}_C_z_{C_z}"
    trial_dir = os.path.join(OUTPUT_DIR, trial_name)
    os.makedirs(trial_dir, exist_ok=True)

    # Setup trial log file
    trial_log_file = os.path.join(trial_dir, "train.log")
    trial_logger = logging.getLogger(f"trial_{trial.number}")
    trial_logger.setLevel(logging.INFO)
    trial_fh = logging.FileHandler(trial_log_file)
    trial_fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    trial_logger.addHandler(trial_fh)

    logger.info("="*80)
    logger.info(f"Trial {trial.number}/{N_TRIALS-1} ({trial.number/N_TRIALS*100:.1f}%)")
    logger.info(f"Hyperparameters: fc_low={fc_low}, fc_high={fc_high}, C_h={C_h}, C_z={C_z}")
    logger.info(f"Directory: {trial_name}")
    logger.info("="*80)

    trial_logger.info(f"Trial {trial.number} started")
    trial_logger.info(f"Hyperparameters: fc_low={fc_low}, fc_high={fc_high}, C_h={C_h}, C_z={C_z}")

    # ========================================
    # 3. Create Config
    # ========================================

    config = FullFeatureConfig()
    config.fc_low = fc_low
    config.fc_high = fc_high
    config.C_h = C_h
    config.C_z = C_z
    config.epochs = EPOCHS_PER_TRIAL
    config.batch_size = BATCH_SIZE
    config.num_workers = NUM_WORKERS
    config.save_dir = trial_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================================
    # 4. Load Data
    # ========================================

    # Training set (with validation split)
    train_dataset = LipLivenessBandDataset(
        data_dir=TRAIN_DIR,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order
    )

    val_size = int(config.val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Test sets
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

    test_live_loader = DataLoader(
        test_live_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_fake_loader = DataLoader(
        test_fake_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    trial_logger.info(f"Train size: {train_size}, Val size: {val_size}")
    trial_logger.info(f"Test live: {len(test_live_dataset)}, Test fake: {len(test_fake_dataset)}")

    # ========================================
    # 5. Create Model
    # ========================================

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    n_params = sum(p.numel() for p in model.parameters())
    trial_logger.info(f"Model parameters: {n_params:,}")

    # ========================================
    # 6. Training Loop
    # ========================================

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, config, device)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start_time

        # Log
        msg = (
            f"Epoch {epoch}/{config.epochs} - "
            f"train_loss: {train_loss['total']:.4f}, "
            f"val_loss: {val_loss['total']:.4f} "
            f"[{epoch_time/60:.1f}min]"
        )
        logger.info(f"  {msg}")
        trial_logger.info(msg)

        # Save best model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'hyperparameters': {
                    'fc_low': fc_low,
                    'fc_high': fc_high,
                    'C_h': C_h,
                    'C_z': C_z
                }
            }, os.path.join(trial_dir, "best.pt"))

        # Pruning
        if USE_PRUNING:
            trial.report(val_loss['total'], epoch)

            if trial.should_prune():
                logger.info(f"  Trial {trial.number} pruned at epoch {epoch}")
                trial_logger.info(f"Trial pruned at epoch {epoch}")
                raise optuna.TrialPruned()

    # ========================================
    # 7. Final Evaluation on Test Set
    # ========================================

    logger.info("  Evaluating on test set...")

    # Load best model
    checkpoint = torch.load(os.path.join(trial_dir, "best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    test_live_loss = validate(model, test_live_loader, config, device)
    test_fake_loss = validate(model, test_fake_loader, config, device)

    logger.info(f"  Test Live Loss: {test_live_loss['total']:.4f}")
    logger.info(f"  Test Fake Loss: {test_fake_loss['total']:.4f}")
    logger.info(f"  Separation: {abs(test_live_loss['total'] - test_fake_loss['total']):.4f}")

    trial_logger.info(f"Test Live Loss: {test_live_loss['total']:.4f}")
    trial_logger.info(f"Test Fake Loss: {test_fake_loss['total']:.4f}")

    # ========================================
    # 8. Create Visualizations
    # ========================================

    create_all_visualizations(
        model, test_live_loader, test_fake_loader, device,
        train_losses, val_losses, trial_dir
    )

    # ========================================
    # 9. Save Metrics
    # ========================================

    metrics = {
        'trial_number': trial.number,
        'hyperparameters': {
            'fc_low': fc_low,
            'fc_high': fc_high,
            'C_h': C_h,
            'C_z': C_z
        },
        'best_val_loss': best_val_loss,
        'test_live_loss': test_live_loss['total'],
        'test_fake_loss': test_fake_loss['total'],
        'separation': abs(test_live_loss['total'] - test_fake_loss['total']),
        'model_parameters': n_params,
        'training_time_minutes': (time.time() - trial_start_time) / 60
    }

    with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ========================================
    # 10. Cleanup
    # ========================================

    trial_logger.removeHandler(trial_fh)
    trial_fh.close()

    trial_time = time.time() - trial_start_time
    logger.info(f"Trial {trial.number} completed - Best val_loss: {best_val_loss:.4f} [{trial_time/60:.1f}min]")
    logger.info("")

    return best_val_loss


# ============================================
# Main
# ============================================

def main():
    """Main function"""

    start_time = time.time()

    logger.info("="*80)
    logger.info("Hyperparameter Tuning for Band-Split VAE")
    logger.info("="*80)
    logger.info(f"Total trials: {N_TRIALS}, Epochs per trial: {EPOCHS_PER_TRIAL}")
    logger.info(f"Pruning: {USE_PRUNING} (startup={PRUNING_STARTUP_TRIALS}, warmup={PRUNING_WARMUP_STEPS})")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*80)
    logger.info("")

    # Create Optuna study
    if USE_PRUNING:
        pruner = MedianPruner(
            n_startup_trials=PRUNING_STARTUP_TRIALS,
            n_warmup_steps=PRUNING_WARMUP_STEPS
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        study_name='bandvae_tuning',
        storage=f'sqlite:///{OUTPUT_DIR}/optuna_study.db',
        load_if_exists=True
    )

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # ========================================
    # Results
    # ========================================

    logger.info("="*80)
    logger.info("All trials completed!")
    logger.info(f"Total time: {(time.time() - start_time)/3600:.1f} hours")
    logger.info("="*80)
    logger.info("")
    logger.info("Best trial:")
    logger.info(f"  Number: {study.best_trial.number}")
    logger.info(f"  Value (val_loss): {study.best_trial.value:.4f}")
    logger.info(f"  Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")
    logger.info("")

    # Save best hyperparameters
    best_params = {
        'trial_number': study.best_trial.number,
        'best_val_loss': study.best_trial.value,
        'hyperparameters': study.best_trial.params,
        'total_trials': N_TRIALS,
        'total_time_hours': (time.time() - start_time) / 3600
    }

    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), 'w') as f:
        json.dump(best_params, f, indent=2)

    logger.info(f"Results saved to {OUTPUT_DIR}/best_hyperparameters.json")
    logger.info("="*80)


if __name__ == "__main__":
    main()
