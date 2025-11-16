"""
Hyperparameter Tuning for Band-Split VAE with T_fixed=300
- Uses 20GBprocessed data (70K samples)
- T_fixed=300 with random crop
- Saves only best.pt
- Logs band-wise reconstruction losses
- Optimized for speed
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
from optuna.pruners import MedianPruner

from dataset_bandvae import LipLivenessBandDataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import FullFeatureConfig


# ========================================
# Configuration
# ========================================

# Data paths
TRAIN_DIR = "/home/elicer/liveness_detection/model1/20GBprocessed"

# Hyperparameter tuning settings
N_TRIALS = 15
EPOCHS_PER_TRIAL = 10
BATCH_SIZE = 128  # Increased for faster training
NUM_WORKERS = 8  # Increased for parallel data loading
USE_PRUNING = True
PRUNING_STARTUP_TRIALS = 5
PRUNING_WARMUP_STEPS = 3

# Output directory
OUTPUT_DIR = "runs/hyperparameter_tuning_T300"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ========================================
# Helper Functions
# ========================================

@torch.no_grad()
def validate(model, loader, config, device):
    """
    Validate the model and return band-wise losses
    """
    model.eval()

    total_loss = 0.0
    lf_rec_loss = 0.0
    bp_rec_loss = 0.0
    hf_rec_loss = 0.0
    n_batches = len(loader)

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        # Forward pass
        recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

        targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
        betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

        loss, loss_dict = band_split_vae_loss(
            recons, mus, logvars, targets,
            x_hat_fused, None, betas=betas, alpha_fusion=0.0
        )

        total_loss += loss_dict['total']
        lf_rec_loss += loss_dict['recon_lf']
        bp_rec_loss += loss_dict['recon_bp']
        hf_rec_loss += loss_dict['recon_hf']

    return {
        'total': total_loss / n_batches,
        'lf_rec': lf_rec_loss / n_batches,
        'bp_rec': bp_rec_loss / n_batches,
        'hf_rec': hf_rec_loss / n_batches
    }


def train_epoch(model, loader, optimizer, config, device, use_amp=True):
    """
    Train for one epoch with mixed precision
    """
    model.train()

    total_loss = 0.0
    lf_rec_loss = 0.0
    bp_rec_loss = 0.0
    hf_rec_loss = 0.0
    n_batches = len(loader)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(device)
        x_bp = x_bp.to(device)
        x_hf = x_hf.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

                targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
                betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

                loss, loss_dict = band_split_vae_loss(
                    recons, mus, logvars, targets,
                    x_hat_fused, None, betas=betas, alpha_fusion=0.0
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recons, mus, logvars, x_hat_fused = model(x_lf, x_bp, x_hf)

            targets = {'lf': x_lf, 'bp': x_bp, 'hf': x_hf}
            betas = {'lf': 1.0, 'bp': 1.0, 'hf': 1.0}

            loss, loss_dict = band_split_vae_loss(
                recons, mus, logvars, targets,
                x_hat_fused, None, betas=betas, alpha_fusion=0.0
            )

            loss.backward()
            optimizer.step()

        total_loss += loss_dict['total']
        lf_rec_loss += loss_dict['recon_lf']
        bp_rec_loss += loss_dict['recon_bp']
        hf_rec_loss += loss_dict['recon_hf']

    return {
        'total': total_loss / n_batches,
        'lf_rec': lf_rec_loss / n_batches,
        'bp_rec': bp_rec_loss / n_batches,
        'hf_rec': hf_rec_loss / n_batches
    }


# ========================================
# Optuna Objective Function
# ========================================

def objective(trial):
    """
    Optuna objective function for a single trial
    """

    # ========================================
    # 1. Sample Hyperparameters
    # ========================================

    fc_low = trial.suggest_categorical('fc_low', [1.5, 2.0, 2.5])
    fc_high = trial.suggest_categorical('fc_high', [7.0, 8.0, 9.0])
    C_h = trial.suggest_categorical('C_h', [48, 64, 96])
    C_z = trial.suggest_categorical('C_z', [12, 16])

    # ========================================
    # 2. Create Trial Directory
    # ========================================

    trial_dir = os.path.join(
        OUTPUT_DIR,
        f"trial_{trial.number:03d}_fc_low_{fc_low}_fc_high_{fc_high}_C_h_{C_h}_C_z_{C_z}"
    )
    os.makedirs(trial_dir, exist_ok=True)

    # Setup trial logger
    trial_logger = logging.getLogger(f"trial_{trial.number}")
    trial_logger.setLevel(logging.INFO)
    trial_handler = logging.FileHandler(os.path.join(trial_dir, "train.log"))
    trial_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    trial_logger.addHandler(trial_handler)

    logger.info("="*80)
    logger.info(f"Trial {trial.number}/{N_TRIALS-1} ({trial.number/(N_TRIALS-1)*100:.1f}%)")
    logger.info(f"Hyperparameters: fc_low={fc_low}, fc_high={fc_high}, C_h={C_h}, C_z={C_z}")
    logger.info(f"Directory: {os.path.basename(trial_dir)}")
    logger.info("="*80)

    trial_logger.info(f"Trial {trial.number} started")
    trial_logger.info(f"Hyperparameters: fc_low={fc_low}, fc_high={fc_high}, C_h={C_h}, C_z={C_z}")

    # ========================================
    # 3. Create Config
    # ========================================

    config = FullFeatureConfig()
    config.T_fixed = 300  # Changed to 300
    config.fc_low = fc_low
    config.fc_high = fc_high
    config.C_h = C_h
    config.C_z = C_z
    config.epochs = EPOCHS_PER_TRIAL
    config.batch_size = BATCH_SIZE
    config.num_workers = NUM_WORKERS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================================
    # 4. Load Datasets with T_fixed=300
    # ========================================

    # Training dataset with random crop
    full_dataset = LipLivenessBandDataset(
        data_dir=TRAIN_DIR,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order,
        random_crop=True  # Enable random crop for training
    )

    # Split into train/val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    trial_logger.info(f"Train size: {train_size}, Val size: {val_size}")

    # ========================================
    # 5. Create Model
    # ========================================

    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial_logger.info(f"Model parameters: {n_params:,}")

    # ========================================
    # 6. Training Loop
    # ========================================

    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Fixed learning rate

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config, device, use_amp=True)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, config, device)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start_time

        # Log with band-wise losses
        msg = (
            f"Epoch {epoch}/{config.epochs} - "
            f"train_loss: {train_loss['total']:.4f} "
            f"(lf:{train_loss['lf_rec']:.4f}, bp:{train_loss['bp_rec']:.4f}, hf:{train_loss['hf_rec']:.4f}), "
            f"val_loss: {val_loss['total']:.4f} "
            f"(lf:{val_loss['lf_rec']:.4f}, bp:{val_loss['bp_rec']:.4f}, hf:{val_loss['hf_rec']:.4f}) "
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
                    'C_z': C_z,
                    'T_fixed': 300
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
    # 7. Save Final Metrics
    # ========================================

    metrics = {
        'trial_number': trial.number,
        'hyperparameters': {
            'fc_low': fc_low,
            'fc_high': fc_high,
            'C_h': C_h,
            'C_z': C_z,
            'T_fixed': 300
        },
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }

    with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    trial_logger.info(f"Trial {trial.number} completed with best val loss: {best_val_loss:.4f}")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")

    # Clean up trial logger
    trial_logger.removeHandler(trial_handler)
    trial_handler.close()

    return best_val_loss


# ========================================
# Main Function
# ========================================

def main():
    """
    Main function to run hyperparameter tuning
    """

    print("="*80)
    print("Hyperparameter Tuning for Band-Split VAE (T_fixed=300)")
    print("="*80)
    print(f"Data directory: {TRAIN_DIR}")
    print(f"Total trials: {N_TRIALS}, Epochs per trial: {EPOCHS_PER_TRIAL}")
    print(f"Batch size: {BATCH_SIZE}, Num workers: {NUM_WORKERS}")
    print(f"Pruning: {USE_PRUNING} (startup={PRUNING_STARTUP_TRIALS}, warmup={PRUNING_WARMUP_STEPS})")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"T_fixed: 300 (with random crop)")
    print("="*80)
    print()

    logger.info("Hyperparameter Tuning for Band-Split VAE (T_fixed=300)")
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
        study_name="bandvae_tuning_T300",
        direction="minimize",
        pruner=pruner,
        storage=f"sqlite:///{os.path.join(OUTPUT_DIR, 'optuna_study.db')}",
        load_if_exists=True
    )

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # Print best trial
    print()
    print("="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val loss: {study.best_trial.value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print("="*80)

    logger.info("")
    logger.info("="*80)
    logger.info("Optimization Complete!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best val loss: {study.best_trial.value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_trial.params}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
