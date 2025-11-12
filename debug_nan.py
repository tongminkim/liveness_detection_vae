"""Debug NaN issue in reconstruction loss"""

import numpy as np
import torch
from model_bandvae import BandSplitVAE
from config_bandvae import FullFeatureConfig

# Load model
config = FullFeatureConfig()
checkpoint_path = "/home/elicer/liveness_detection/model1/runs/bandvae_full/best.pt"

print("Loading model...")
model = BandSplitVAE(
    C_in_per_band=config.C_in_per_band,
    C_h=config.C_h,
    C_z=config.C_z,
    dilations=config.dilations
)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded successfully!")
print(f"C_in_per_band: {config.C_in_per_band}")

# Create dummy input
T = 150
C_in = config.C_in_per_band  # 320

print(f"\nCreating dummy input: [1, {C_in}, {T}]")
x_lf = torch.randn(1, C_in, T).float()
x_bp = torch.randn(1, C_in, T).float()
x_hf = torch.randn(1, C_in, T).float()

print(f"Input LF stats: min={x_lf.min():.4f}, max={x_lf.max():.4f}, mean={x_lf.mean():.4f}")
print(f"Input BP stats: min={x_bp.min():.4f}, max={x_bp.max():.4f}, mean={x_bp.mean():.4f}")
print(f"Input HF stats: min={x_hf.min():.4f}, max={x_hf.max():.4f}, mean={x_hf.mean():.4f}")

# Forward pass
print("\nForward pass...")
with torch.no_grad():
    recons, mus, logvars, fused = model(x_lf, x_bp, x_hf)

print("\nReconstructions:")
print(f"  LF shape: {recons['lf'].shape}, stats: min={recons['lf'].min():.4f}, max={recons['lf'].max():.4f}, mean={recons['lf'].mean():.4f}, has_nan={torch.isnan(recons['lf']).any()}")
print(f"  BP shape: {recons['bp'].shape}, stats: min={recons['bp'].min():.4f}, max={recons['bp'].max():.4f}, mean={recons['bp'].mean():.4f}, has_nan={torch.isnan(recons['bp']).any()}")
print(f"  HF shape: {recons['hf'].shape}, stats: min={recons['hf'].min():.4f}, max={recons['hf'].max():.4f}, mean={recons['hf'].mean():.4f}, has_nan={torch.isnan(recons['hf']).any()}")

# Compute losses
loss_lf = torch.abs(recons['lf'] - x_lf).mean().item()
loss_bp = torch.abs(recons['bp'] - x_bp).mean().item()
loss_hf = torch.abs(recons['hf'] - x_hf).mean().item()

print("\nReconstruction losses:")
print(f"  LF: {loss_lf}")
print(f"  BP: {loss_bp}")
print(f"  HF: {loss_hf}")

# Check model parameters
print("\nChecking model parameters for NaN...")
has_nan = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"  {name} has NaN!")
        has_nan = True

if not has_nan:
    print("  All parameters are valid (no NaN)")

print("\n✓ Debug complete!")
