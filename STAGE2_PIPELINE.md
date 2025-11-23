# Stage 2: Negative Sampling Pipeline

## Overview

This document describes the 3-stage pipeline for implementing and comparing negative sampling strategies for liveness detection.

## Pipeline Structure

```
Stage 1: Pre-training (One-Class Learning)
    ↓
    20GBprocessed (Real/Live data only)
    ↓
    VAE learns normal lip motion patterns
    ↓
    Output: runs/stage1_pretrain/stage1_pretrained.pt

Stage 2: Fine-tuning with Negative Sampling (Two Methods)
    ↓
    test_balanced_npz (50% for training: Real + Fake)
    ↓
    ├─ Method 1: Margin Loss (Contrastive Learning)
    │  └─ Output: runs/stage2_method1_margin/stage2_method1_best.pt
    │
    └─ Method 2: Discriminator in Latent Space
       └─ Output: runs/stage2_method2_discriminator/stage2_method2_best.pt

Stage 3: Final Evaluation
    ↓
    test_balanced_npz (50% for testing: Real + Fake)
    ↓
    Compare both methods
    ↓
    Output: runs/stage2_comparison/
```

## Data Split

### Source Data
- **Stage 1 Training**: `/home/elicer/liveness_detection/model1/20GBprocessed`
  - 35,016 Real/Live samples
  - Train: 31,514 (90%)
  - Val: 3,502 (10%)

- **Stage 2 Training + Final Test**: `/home/elicer/liveness_detection/model1/test_balanced_npz`
  - Split: `data_split.json` (random seed 42)
  - Fake directories used: `audio-driven`, `dfl`, `dffs`
  - Fake directories excluded: `fo`, `fsgan`

### Split Details
```json
{
  "stage2_train": {
    "real": 1,088 samples,
    "fake": 538 samples
  },
  "final_test": {
    "real": 1,089 samples,
    "fake": 539 samples
  }
}
```

## Hyperparameters

### Stage 1: Pre-training (논문 기준)
```python
T_fixed = 300         # Sequence length (changed from 150)
fc_low = 2.0          # Low cutoff frequency (Hz)
fc_high = 8.0         # High cutoff frequency (Hz)
filter_order = 4      # Butterworth filter order
C_h = 48              # Hidden dimension
C_z = 12              # Latent dimension
dilations = [1,2,4]   # Temporal dilations
lr = 1e-3             # Learning rate
epochs = 20           # Training epochs
batch_size = 64       # Batch size
```

**Features**: FULL mode (position + velocity + acceleration + angle + angle_rate)
- 40 landmarks × 8 features = 320 channels per band

### Stage 2: Fine-tuning

**Method 1: Margin Loss**
```python
lr = 1e-4             # Lower learning rate for fine-tuning
epochs = 10           # Fewer epochs
margin = 0.5          # Minimum reconstruction loss for fake samples
lambda_margin = 1.0   # Weight for margin loss
```

**Method 2: Discriminator**
```python
lr = 1e-4             # Lower learning rate for fine-tuning
epochs = 10           # Fewer epochs
lambda_cls = 1.0      # Weight for classification loss
hidden_dim = 128      # Discriminator hidden dimension
```

## Method Details

### Method 1: Margin Loss (Contrastive Learning)

**Goal**: Push fake samples to have high reconstruction loss

**Loss Function**:
```
L_total = L_real + λ × max(0, margin - L_fake)

where:
- L_real: Reconstruction loss for real samples (minimize)
- L_fake: Reconstruction loss for fake samples
- margin: Minimum threshold for fake samples (0.5)
- λ: Weight for margin loss (1.0)
```

**Evaluation Metric**: Separation (fake_loss - real_loss)

**Detection Strategy**: Use reconstruction loss as anomaly score
- Real samples: LOW reconstruction loss
- Fake samples: HIGH reconstruction loss (above margin)

### Method 2: Discriminator in Latent Space

**Goal**: Classify Real vs Fake based on latent representations

**Architecture**:
```
Input: z_concat = [z_lf, z_bp, z_hf]  (C_z × 3 = 36 dimensions)
    ↓
Linear(36, 128) → ReLU → Dropout(0.3)
    ↓
Linear(128, 64) → ReLU → Dropout(0.3)
    ↓
Linear(64, 1) → Sigmoid
    ↓
Output: P(Fake)
```

**Loss Function**:
```
L_total = L_reconstruction + λ × L_classification

where:
- L_reconstruction: VAE reconstruction loss
- L_classification: Binary cross-entropy loss
- λ: Weight for classification loss (1.0)
```

**Evaluation Metric**: Classification accuracy

**Detection Strategy**: Use discriminator output as anomaly score
- Real samples: LOW probability (close to 0)
- Fake samples: HIGH probability (close to 1)

## Execution Workflow

### 1. Data Preparation (COMPLETED ✓)
```bash
python split_test_balanced.py
# Output: data_split.json
```

### 2. Stage 1: Pre-training (IN PROGRESS 🔄)
```bash
python train_stage1_pretrain.py
# Current status: Epoch 3/20
# Output: runs/stage1_pretrain/stage1_pretrained.pt
```

### 3. Stage 2: Fine-tuning (PENDING ⏳)

**After Stage 1 completes, run both methods:**

```bash
# Method 1: Margin Loss
python train_stage2_method1_margin.py
# Output: runs/stage2_method1_margin/stage2_method1_best.pt

# Method 2: Discriminator
python train_stage2_method2_discriminator.py
# Output: runs/stage2_method2_discriminator/stage2_method2_best.pt
```

### 4. Final Evaluation (PENDING ⏳)

**After both methods complete:**

```bash
python evaluate_stage2_methods.py
# Outputs:
#   - runs/stage2_comparison/comparison_metrics.json
#   - runs/stage2_comparison/tsne_comparison.png
#   - runs/stage2_comparison/score_distributions.png
```

## Expected Outputs

### Stage 1: Pre-training
- `runs/stage1_pretrain/stage1_pretrained.pt`: Best pre-trained model checkpoint
- `runs/stage1_pretrain/train.log`: Training log

### Stage 2: Method 1 (Margin Loss)
- `runs/stage2_method1_margin/stage2_method1_best.pt`: Best fine-tuned model
- Training metrics: total_loss, real_rec, fake_rec, separation

### Stage 2: Method 2 (Discriminator)
- `runs/stage2_method2_discriminator/stage2_method2_best.pt`: Best fine-tuned model
- Training metrics: total_loss, rec_loss, cls_loss, accuracy

### Stage 3: Comparison
- `comparison_metrics.json`: Comprehensive metrics for both methods
  - AUC, Accuracy, Precision, Recall, F1-score
  - Separation, Thresholds
  - Mean scores for Real/Fake samples

- `tsne_comparison.png`: t-SNE visualization of latent spaces
  - Side-by-side comparison of Method 1 vs Method 2
  - Blue points: Real samples
  - Red points: Fake samples

- `score_distributions.png`: Anomaly score distributions
  - Histograms for Real vs Fake samples
  - Optimal thresholds marked
  - AUC scores displayed

## Key Differences Between Methods

| Aspect | Method 1: Margin Loss | Method 2: Discriminator |
|--------|----------------------|-------------------------|
| **Approach** | Anomaly detection | Binary classification |
| **Training** | Unsupervised + Margin | Supervised |
| **Additional Parameters** | None | Discriminator network (~25k params) |
| **Anomaly Score** | Reconstruction loss | Sigmoid(logits) |
| **Optimization** | Push fake loss above margin | Minimize classification error |
| **Expected Strength** | Better generalization | Better discrimination |

## Files Created

1. `split_test_balanced.py` - Data split script
2. `data_split.json` - Fixed train/test split
3. `dataset_stage2.py` - Dataset loader for Stage 2
4. `train_stage1_pretrain.py` - Stage 1 pre-training script
5. `train_stage2_method1_margin.py` - Method 1 fine-tuning script
6. `train_stage2_method2_discriminator.py` - Method 2 fine-tuning script
7. `evaluate_stage2_methods.py` - Comparison evaluation script
8. `STAGE2_PIPELINE.md` - This documentation

## Notes

- **Random Seed**: 42 (for reproducibility)
- **Feature Mode**: FULL (8 features per landmark)
- **Same Split**: Both methods use identical train/test splits
- **Mixed Precision**: AMP enabled for faster training
- **Evaluation**: Both methods evaluated on held-out final_test split

## Next Steps

1. Wait for Stage 1 pre-training to complete (~80 minutes total)
2. Run Stage 2 Method 1 (Margin Loss)
3. Run Stage 2 Method 2 (Discriminator)
4. Compare results using evaluation script
5. Select best method based on AUC, F1-score, and separation metrics
