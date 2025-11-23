"""
One-Class VAE 학습 스크립트

- Live 데이터만으로 학습
- Best model 저장
- 학습 진행 상황 로깅
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import LipLivenessDataset
from model import FeatureSpaceVAE, vae_loss
from config import get_config


# ============================================
# Training Functions
# ============================================

def train_epoch(model, loader, optimizer, config, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    n_batches = len(loader)

    start_time = time.time()

    for batch_idx, x in enumerate(loader):
        x = x.to(config.device)

        # Forward pass
        x_hat, mu, logvar = model(x)

        # Compute loss
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=config.beta)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        recon_loss += recon.item()
        kl_loss += kl.item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - start_time
            avg_time = elapsed / (batch_idx + 1)
            eta = (n_batches - (batch_idx + 1)) * avg_time

            print(
                f"  [Epoch {epoch:02d} | {batch_idx + 1:04d}/{n_batches}] "
                f"Loss={total_loss / (batch_idx + 1):.4f} | "
                f"Recon={recon_loss / (batch_idx + 1):.4f} | "
                f"KL={kl_loss / (batch_idx + 1):.4f} | "
                f"ETA={eta / 60:.1f}m"
            )

    return total_loss / n_batches, recon_loss / n_batches, kl_loss / n_batches


@torch.no_grad()
def validate(model, loader, config):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    n_samples = 0

    for x in loader:
        x = x.to(config.device)

        # Forward pass
        x_hat, mu, logvar = model(x)

        # Compute loss
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=config.beta)

        # Accumulate
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        recon_loss += recon.item() * batch_size
        kl_loss += kl.item() * batch_size
        n_samples += batch_size

    return (
        total_loss / n_samples,
        recon_loss / n_samples,
        kl_loss / n_samples
    )


# ============================================
# Main Training Loop
# ============================================

def main(config):
    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("One-Class VAE Training for Lip Liveness Detection")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Feature mode: {config.feature_mode.upper()}")
    print(f"    - Features: {', '.join(config.get_feature_names())}")
    print(f"    - Input channels: {config.C_in}")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Beta (KL weight): {config.beta}")
    print(f"  Save dir: {config.save_dir}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = LipLivenessDataset(
        data_dir=config.data_dir,
        T_fixed=config.T_fixed,
        fps=config.fps,
        use_procrustes=True,
        use_acceleration=config.use_acceleration,
        use_angle=config.use_angle,
        use_angle_rate=config.use_angle_rate
    )

    # Split train/val
    train_size = int((1 - config.val_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
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
    model = FeatureSpaceVAE(
        C_in=config.C_in,
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
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-"*60)

        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, config, epoch
        )

        # Validate
        val_loss, val_recon, val_kl = validate(model, val_loader, config)

        # Print summary
        print(f"\n  Summary:")
        print(f"    Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"    Val Loss:   {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_path)
            print(f"    🌟 New best model saved! (Val loss: {val_loss:.4f})")

        # Save checkpoint every epoch
        ckpt_path = save_dir / f"checkpoint_epoch{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, ckpt_path)
        print(f"    Checkpoint saved: {ckpt_path.name}")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {best_path}")
    print("="*60)


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train One-Class VAE")

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
    parser.add_argument("--beta", type=float, default=None)
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
    if args.beta is not None:
        config.beta = args.beta
    if args.device is not None:
        config.device = args.device

    # Run training
    main(config)
