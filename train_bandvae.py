"""
Band-Split VAE 학습 스크립트

- 3-band split (LF/BP/HF)
- Live 데이터만으로 학습
- Per-band reconstruction + KL loss
- Weighted fusion
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset_bandvae import LipLivenessBandDataset
from model_bandvae import BandSplitVAE, band_split_vae_loss
from config_bandvae import get_config


# ============================================
# Training Functions
# ============================================

def train_epoch(model, loader, optimizer, config, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    recon_lf_loss = 0.0
    recon_bp_loss = 0.0
    recon_hf_loss = 0.0
    kl_lf_loss = 0.0
    kl_bp_loss = 0.0
    kl_hf_loss = 0.0
    fusion_loss = 0.0

    n_batches = len(loader)
    start_time = time.time()

    for batch_idx, (x_lf, x_bp, x_hf) in enumerate(loader):
        x_lf = x_lf.to(config.device)
        x_bp = x_bp.to(config.device)
        x_hf = x_hf.to(config.device)

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
            x_hat_fused, None,  # No full-band target for now
            betas=betas,
            alpha_fusion=0.0  # Disable fusion penalty during training
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

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - start_time
            avg_time = elapsed / (batch_idx + 1)
            eta = (n_batches - (batch_idx + 1)) * avg_time

            n = batch_idx + 1
            total_recon = (recon_lf_loss + recon_bp_loss + recon_hf_loss) / n
            total_kl = (kl_lf_loss + kl_bp_loss + kl_hf_loss) / n

            print(
                f"  [Epoch {epoch:02d} | {n:04d}/{n_batches}] "
                f"Total={total_loss / n:.4f} "
                f"(Recon={total_recon:.4f} KL={total_kl:.4f}) | "
                f"LF:[R={recon_lf_loss / n:.4f} K={kl_lf_loss / n:.4f}] "
                f"BP:[R={recon_bp_loss / n:.4f} K={kl_bp_loss / n:.4f}] "
                f"HF:[R={recon_hf_loss / n:.4f} K={kl_hf_loss / n:.4f}] | "
                f"ETA={eta / 60:.1f}m"
            )

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
def validate(model, loader, config):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    recon_lf_loss = 0.0
    recon_bp_loss = 0.0
    recon_hf_loss = 0.0
    kl_lf_loss = 0.0
    kl_bp_loss = 0.0
    kl_hf_loss = 0.0

    n_samples = 0

    for x_lf, x_bp, x_hf in loader:
        x_lf = x_lf.to(config.device)
        x_bp = x_bp.to(config.device)
        x_hf = x_hf.to(config.device)

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
        batch_size = x_lf.size(0)
        total_loss += loss_dict['total'] * batch_size
        recon_lf_loss += loss_dict['recon_lf'] * batch_size
        recon_bp_loss += loss_dict['recon_bp'] * batch_size
        recon_hf_loss += loss_dict['recon_hf'] * batch_size
        kl_lf_loss += loss_dict['kl_lf'] * batch_size
        kl_bp_loss += loss_dict['kl_bp'] * batch_size
        kl_hf_loss += loss_dict['kl_hf'] * batch_size
        n_samples += batch_size

    return {
        'total': total_loss / n_samples,
        'recon_lf': recon_lf_loss / n_samples,
        'recon_bp': recon_bp_loss / n_samples,
        'recon_hf': recon_hf_loss / n_samples,
        'kl_lf': kl_lf_loss / n_samples,
        'kl_bp': kl_bp_loss / n_samples,
        'kl_hf': kl_hf_loss / n_samples
    }


# ============================================
# Main Training Loop
# ============================================

def main(config):
    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Band-Split VAE Training for Lip Liveness Detection")
    print("="*60)
    print(f"\\nConfiguration:")
    print(f"  Feature mode: {config.feature_mode.upper()}")
    print(f"    - Features: {', '.join(config.get_feature_names())}")
    print(f"    - Channels per band: {config.C_in_per_band}")
    print(f"  Frequency bands: 3 (LF/BP/HF)")
    print(f"    - LF cutoff: {config.fc_low}Hz")
    print(f"    - HF cutoff: {config.fc_high}Hz")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Beta weights: LF={config.beta_lf}, BP={config.beta_bp}, HF={config.beta_hf}")
    print(f"  Save dir: {config.save_dir}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = LipLivenessBandDataset(
        data_dir=config.data_dir,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_procrustes=True,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate,
        fc_low=config.fc_low,
        fc_high=config.fc_high,
        filter_order=config.filter_order
    )

    # Split train/val
    train_size = int((1 - config.val_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Create model
    print("Creating model...")
    model = BandSplitVAE(
        C_in_per_band=config.C_in_per_band,
        C_h=config.C_h,
        C_z=config.C_z,
        dilations=config.dilations
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    best_val_loss = float('inf')
    best_path = save_dir / "best.pt"

    print("Starting training...")
    print("="*60)

    for epoch in range(1, config.epochs + 1):
        print(f"\\nEpoch {epoch}/{config.epochs}")
        print("-"*60)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, epoch)

        # Validate
        val_metrics = validate(model, val_loader, config)

        # Print summary
        train_recon = train_metrics['recon_lf'] + train_metrics['recon_bp'] + train_metrics['recon_hf']
        train_kl = train_metrics['kl_lf'] + train_metrics['kl_bp'] + train_metrics['kl_hf']
        val_recon = val_metrics['recon_lf'] + val_metrics['recon_bp'] + val_metrics['recon_hf']
        val_kl = val_metrics['kl_lf'] + val_metrics['kl_bp'] + val_metrics['kl_hf']

        print(f"\\n  Summary:")
        print(f"    Train - Total: {train_metrics['total']:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f}")
        print(f"            LF:[R={train_metrics['recon_lf']:.4f} K={train_metrics['kl_lf']:.4f}] "
              f"BP:[R={train_metrics['recon_bp']:.4f} K={train_metrics['kl_bp']:.4f}] "
              f"HF:[R={train_metrics['recon_hf']:.4f} K={train_metrics['kl_hf']:.4f}]")
        print(f"    Val   - Total: {val_metrics['total']:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f}")
        print(f"            LF:[R={val_metrics['recon_lf']:.4f} K={val_metrics['kl_lf']:.4f}] "
              f"BP:[R={val_metrics['recon_bp']:.4f} K={val_metrics['kl_bp']:.4f}] "
              f"HF:[R={val_metrics['recon_hf']:.4f} K={val_metrics['kl_hf']:.4f}]")

        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_path)
            print(f"    🌟 New best model saved! (Val loss: {val_metrics['total']:.4f})")

        # Save checkpoint every epoch
        ckpt_path = save_dir / f"checkpoint_epoch{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': config
        }, ckpt_path)
        print(f"    Checkpoint saved: {ckpt_path.name}")

    print("\\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {best_path}")
    print("="*60)


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Band-Split VAE")

    parser.add_argument(
        "--mode",
        type=str,
        default="simple",
        choices=["simple", "full"],
        help="Feature mode: 'simple' (position+velocity) or 'full' (all 5 features)"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Get config based on mode
    config = get_config(args.mode)

    # Override with command line args
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.device is not None:
        config.device = args.device

    # Run training
    main(config)
