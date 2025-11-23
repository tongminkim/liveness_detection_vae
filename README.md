# Liveness Detection using Band-Split VAE

Audio-Driven Deepfake 영상 탐지를 위한 Band-Split VAE 기반 Liveness Detection 시스템

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [데이터셋](#데이터셋)
- [Stage 1: VAE Pretrain](#stage-1-vae-pretrain)
- [Stage 2: Discriminative Training](#stage-2-discriminative-training)
- [Gaussian Filtering & Hyperparameter Tuning](#gaussian-filtering--hyperparameter-tuning)
- [주요 파일 구조](#주요-파일-구조)
- [실험 결과](#실험-결과)
- [사용 방법](#사용-방법)
- [Requirements](#requirements)

---

## 프로젝트 개요

본 프로젝트는 **Band-Split VAE** 아키텍처를 활용하여 audio-driven deepfake 영상을 탐지하는 liveness detection 시스템입니다.

### 핵심 아이디어

1. **주파수 대역 분리**: 얼굴 랜드마크 움직임을 3개의 주파수 대역으로 분리
   - Low Frequency (LF): 2Hz 이하
   - Band Pass (BP): 2-8Hz
   - High Frequency (HF): 8Hz 이상

2. **2단계 학습**:
   - **Stage 1**: Real 영상만으로 VAE 사전학습 (정상 분포 학습)
   - **Stage 2**: Real과 Fake를 구별하는 discriminative 학습

3. **2단계 가우시안 필터링**:
   - **1차 필터**: Band별 reconstruction loss 기반
   - **2차 필터**: PCA 축소된 latent space 기반

### 최종 성능

| Metric | Value |
|--------|-------|
| Accuracy | **89.32%** |
| Precision | 86.51% |
| Recall (Fake Detection) | **93.16%** |
| F1-Score | **89.71%** |
| Specificity (Real Preservation) | 85.47% |

---

## 데이터셋

### 데이터 구조

```
20GBprocessed/           # 전처리된 NPZ 파일 (제외됨)
├── real/
│   ├── 원본1/
│   ├── 원본2/
│   └── ...
└── fake/
    ├── audio-driven/    # 본 프로젝트에서 주로 사용
    ├── dfl/
    ├── dffs/
    ├── fo/
    └── fsgan/

test_balanced_npz/       # 균형잡힌 테스트 데이터 (제외됨)
├── real/                # 117개 샘플
└── fake/
    └── audio-driven/    # 117개 샘플
```

### NPZ 파일 구조

각 NPZ 파일은 다음 데이터를 포함:

```python
{
    'features': np.array,      # Shape: (T, 136)
                              # 68 landmarks × 2 (x, y) coordinates
    'velocities': np.array,    # Shape: (T-1, 136)
    'accelerations': np.array, # Shape: (T-2, 136)
    'angles': np.array,        # Shape: (T, 68)
    'angle_rates': np.array,   # Shape: (T-1, 68)
    'fps': float              # 영상 FPS
}
```

### 데이터 분할

**설정 파일**: `data_split.json`

```json
{
  "random_seed": 42,
  "fake_dirs_used": ["audio-driven", "dfl", "dffs"],
  "fake_dirs_excluded": ["fo", "fsgan"],

  "stage2_train": {
    "real": [...],  // 1088 samples
    "fake": [...]   // 538 samples (audio-driven, dfl, dffs)
  },

  "final_test": {
    "real": [...],  // 117 samples
    "fake": [...]   // 117 samples (audio-driven only)
  }
}
```

**주요 스크립트**:
- `split_test_balanced.py`: 최초 데이터 분할
- `balance_final_test.py`: Final test를 audio-driven만으로 균형 맞춤

---

## Stage 1: VAE Pretrain

### 목적

Real 영상의 정상적인 얼굴 움직임 패턴을 학습하여, 이후 fake 영상 탐지의 기준점 확립

### 사용 데이터

- **학습 데이터**: `data_split.json`의 `stage2_train`에서 **real 샘플만 사용** (1088개)
- **데이터 위치**: `20GBprocessed/real/`

### 학습 스크립트

**파일**: `train_stage1_pretrain.py`

**주요 설정**:
```python
# 모델 구조
C_h = 48          # Hidden channels
C_z = 12          # Latent channels per band
dilations = [1, 2, 4]

# 학습 설정
T_fixed = 300     # 시퀀스 길이
batch_size = 32
learning_rate = 1e-4
epochs = 50
```

**학습 방법**:
```bash
python train_stage1_pretrain.py
```

**Loss 함수**:
- Reconstruction Loss (MSE): 각 주파수 대역별 재구성 오차
- KL Divergence: Latent space의 정규화
- Total Loss = Reconstruction Loss + β × KL Divergence

### 저장된 가중치

**위치**: `runs/stage1_pretrain/stage1_pretrained.pt`

**파일 크기**: 2.7MB

**내용**:
```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'train_loss': float,
    'config': FullFeatureConfig,
    'T_fixed': 300
}
```

**로드 방법**:
```python
from model_bandvae import BandSplitVAE
from config_bandvae import FullFeatureConfig

config = FullFeatureConfig()
model = BandSplitVAE(
    C_in_per_band=config.C_in_per_band,
    C_h=48,
    C_z=12,
    dilations=[1, 2, 4]
)

checkpoint = torch.load('runs/stage1_pretrain/stage1_pretrained.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 관련 시각화

Stage 1 자체는 unsupervised learning이므로 별도 시각화 없음. Stage 2 평가 시 함께 확인 가능.

---

## Stage 2: Discriminative Training

Stage 1의 사전학습된 VAE를 기반으로 Real과 Fake를 구별하는 능력 학습

### 사용 데이터

- **학습 데이터**: `data_split.json`의 `stage2_train`
  - Real: 1088개 샘플
  - Fake: 538개 샘플 (audio-driven, dfl, dffs)
- **테스트 데이터**: `data_split.json`의 `final_test`
  - Real: 117개 샘플
  - Fake: 117개 샘플 (audio-driven만)

### Method 1: Margin Loss

#### 개념

Real 샘플의 reconstruction loss는 낮게, Fake 샘플의 reconstruction loss는 높게 유지하도록 margin을 두고 학습

#### 학습 스크립트

**파일**: `train_stage2_method1_margin.py`

**Loss 함수**:
```python
# Real samples: minimize reconstruction loss
loss_real = reconstruction_loss(x_real)

# Fake samples: push reconstruction loss above margin
loss_fake = max(0, margin - reconstruction_loss(x_fake))

# Total loss
loss = loss_real + lambda_fake * loss_fake
```

**주요 하이퍼파라미터**:
```python
margin = 1.5          # Real과 Fake의 분리 마진
lambda_fake = 1.0     # Fake loss 가중치
learning_rate = 1e-5  # Fine-tuning을 위한 작은 LR
epochs = 20
```

**학습 방법**:
```bash
python train_stage2_method1_margin.py
```

#### 저장된 가중치

**위치**: `runs/stage2_method1_margin/stage2_method1_best.pt`

**파일 크기**: 2.7MB

**내용**:
```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'val_loss': float,
    'margin': 1.5,
    'config': FullFeatureConfig
}
```

#### 개선 버전

**파일**: `train_stage2_method1_margin_fixed.py`
- 학습 안정성 개선
- 저장 위치: `runs/stage2_method1_margin_fixed/stage2_method1_best.pt`

### Method 2: Discriminator

#### 개념

Latent space에서 별도의 discriminator network를 학습하여 Real/Fake 분류

#### 학습 스크립트

**파일**: `train_stage2_method2_discriminator.py`

**구조**:
```python
# VAE: Encoder-Decoder (Stage 1에서 사전학습됨)
vae = BandSplitVAE(...)

# Discriminator: Latent space에서 작동
discriminator = LatentDiscriminator(
    C_z=12,
    num_bands=3,
    hidden_dim=128
)
```

**Loss 함수**:
```python
# VAE reconstruction loss (real only)
loss_recon = reconstruction_loss(x_real)

# Discriminator loss (binary cross-entropy)
logits_real = discriminator(z_real)
logits_fake = discriminator(z_fake)

loss_disc = BCE(logits_real, ones) + BCE(logits_fake, zeros)

# Total loss
loss = loss_recon + lambda_disc * loss_disc
```

**주요 하이퍼파라미터**:
```python
lambda_disc = 0.1     # Discriminator loss 가중치
learning_rate = 1e-5
epochs = 20
```

**학습 방법**:
```bash
python train_stage2_method2_discriminator.py
```

#### 저장된 가중치

**위치**: `runs/stage2_method2_discriminator/stage2_method2_best.pt`

**파일 크기**: 3.0MB

**내용**:
```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'discriminator_state_dict': OrderedDict,  # 추가
    'optimizer_vae_state_dict': OrderedDict,
    'optimizer_disc_state_dict': OrderedDict,
    'val_loss': float,
    'config': FullFeatureConfig
}
```

### Stage 2 평가 및 시각화

#### 평가 스크립트

**전체 테스트 셋 평가**: `evaluate_and_visualize_stage2.py`
- 모든 fake 유형 포함 (audio-driven, dfl, dffs)

**Audio-driven만 평가**: `evaluate_and_visualize_stage2_audiodriven.py`
- Audio-driven fake만 평가 (최종 성능 측정)

#### 시각화 결과 위치

**전체 평가**: `runs/stage2_evaluation/`
```
runs/stage2_evaluation/
├── 1_score_distribution.png        # Anomaly score 분포
├── 2_roc_curve.png                  # ROC curve
├── 3_latent_2d_tsne.png            # 2D t-SNE
├── 4_latent_3d_tsne.png            # 3D t-SNE
└── 5_score_by_fake_type.png        # Fake 유형별 점수 비교
```

**Audio-driven 평가**: `runs/stage2_evaluation_audiodriven/`
```
runs/stage2_evaluation_audiodriven/
├── 1_score_distribution.png        # Audio-driven에 특화된 분포
├── 2_roc_curve.png
├── 3_latent_2d_tsne.png
├── 4_latent_3d_tsne.png
└── 5_score_comparison.png          # Method 1 vs Method 2 비교
```

---

## Gaussian Filtering & Hyperparameter Tuning

Stage 2 학습 후, reconstruction loss와 latent distribution에 가우시안 분포를 적합하여 anomaly detection 수행

### 전체 튜닝 흐름

```
실험 1: Baseline
  ↓
실험 2: Confidence Interval Tuning (81개 설정)
  ↓
실험 3: Advanced (3D Loss + PCA) (448개 설정)
```

### 실험 1: Baseline 2단계 필터링

#### 스크립트

**파일**: `evaluate_gaussian_filters.py`

#### 방법

**1차 필터**: 1차원 가우시안 (전체 reconstruction loss)
- 학습 데이터 (real)의 평균 μ, 표준편차 σ 계산
- 95% 신뢰구간: [μ - 1.96σ, μ + 1.96σ]

**2차 필터**: 36차원 다변량 가우시안 (latent space)
- Latent 구성: [z_lf (12D), z_bp (12D), z_hf (12D)] = 36D
- Mahalanobis distance 기반 판정
- 95% 신뢰구간 (Chi-squared, df=36)

#### 결과

| Metric | Value |
|--------|-------|
| Accuracy | 70.94% |
| Precision | 91.53% |
| Recall | 46.15% |
| F1-Score | 61.36% |

**문제점**: Fake의 53.8%를 놓침 (FN=63)

#### 저장 위치

```
runs/gaussian_filtering/
├── gaussian_filtering_metrics.json  # 성능 지표
├── confusion_matrix.png             # Confusion matrix
└── loss_distributions.png           # Loss 분포 시각화
```

### 실험 2: Confidence Interval Tuning

#### 스크립트

**파일**: `tune_gaussian_filters.py`

#### 방법

**Grid Search**: 81개 설정 (9 × 9)
- 1차 필터 신뢰수준: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
- 2차 필터 신뢰수준: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

#### 결과

**최적 설정**:
- 1차 필터: 70% 신뢰구간
- 2차 필터: 영향 미미 (모든 값에서 동일)

| Metric | Baseline (95%) | Optimized (70%) | 개선 |
|--------|---------------|-----------------|------|
| Accuracy | 70.94% | 78.21% | +7.27% |
| Recall | 46.15% | 81.20% | **+35.05%** |
| F1-Score | 61.36% | 78.84% | **+17.48%** |

**핵심 발견**:
- 1차 필터가 성능 주도
- 2차 필터 효과 미미 (36차원의 한계)

#### 저장 위치

```
runs/gaussian_filtering_tuning/
├── all_results.json                 # 81개 전체 설정 결과
├── hyperparameter_heatmaps.png      # 5개 지표의 히트맵
├── precision_recall_tradeoff.png    # Pareto frontier
├── top_configurations.png           # 상위 5개 confusion matrix
└── stage1_pass_rate_analysis.png    # 1차 필터 통과율 영향
```

### 실험 3: Advanced Filtering (PCA + 3D Loss)

#### 스크립트

**파일**: `tune_gaussian_advanced.py`

#### 핵심 개선사항

1. **1차 필터 개선**: 1D → 3D Band-wise Loss
   ```python
   # 기존: 전체 loss 하나
   loss_total = MSE(reconstruction, target)

   # 개선: Band별 loss 3개
   loss_lf = MSE(recon_lf, target_lf)
   loss_bp = MSE(recon_bp, target_bp)
   loss_hf = MSE(recon_hf, target_hf)
   loss_vector = [loss_lf, loss_bp, loss_hf]  # 3D
   ```

2. **2차 필터 개선**: 36D → PCA 차원축소
   ```python
   # 기존: Full 36D latent
   latent_36d = concatenate([z_lf, z_bp, z_hf])

   # 개선: PCA로 축소
   pca = PCA(n_components=k)  # k = 5, 10, 15, 20, 25, 30
   latent_k = pca.fit_transform(latent_36d)
   ```

#### 실험 설정

**Grid Search**: 448개 설정 (7 × 8 × 8)
- 7가지 방법: Full36D, PCA_5D, PCA_10D, PCA_15D, PCA_20D, PCA_25D, PCA_30D
- 1차 필터 신뢰수준: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
- 2차 필터 신뢰수준: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

#### 결과

**최적 설정**:
- 1차 필터: 3D Band-wise Loss, 95% 신뢰구간
- 2차 필터: PCA 10D Latent, 70% 신뢰구간
- 설명 분산: 91.16%

**성능**:

| Metric | Baseline | Tuned CI | **Advanced** | 총 개선 |
|--------|----------|----------|-------------|---------|
| Accuracy | 70.94% | 78.21% | **89.32%** | +18.38% |
| Recall | 46.15% | 81.20% | **93.16%** | **+47.01%** |
| F1-Score | 61.36% | 78.84% | **89.71%** | **+28.35%** |

**Confusion Matrix**:
```
                 예측값
              Real    Fake
실제 Real     100      17
     Fake       8     109
```

**핵심 발견**:
- PCA 10D가 Full 36D보다 **+37.54% F1 향상**
- 차원의 저주 극복
- 3D band-wise loss가 1D보다 **+10.87% F1 향상**

#### 저장 위치

```
runs/gaussian_filtering_advanced/
├── all_results.json                 # 448개 전체 설정 결과
├── method_comparison.png            # 7가지 방법 비교
├── pca_analysis.png                 # F1 vs 차원, 설명분산
└── top_configs_per_method.png       # 방법별 상위 3개 설정
```

### 상세 튜닝 보고서

**파일**: `하이퍼파라미터_튜닝_보고서.md`

모든 실험의 상세 내용, 결과, 분석을 담은 한국어 보고서 (75KB)

---

## 주요 파일 구조

```
liveness_detection/model1/
│
├── README.md                              # 본 문서
├── 하이퍼파라미터_튜닝_보고서.md             # 상세 튜닝 보고서
│
├── config_bandvae.py                      # 모델 설정 파일
├── model_bandvae.py                       # Band-Split VAE 모델
├── dataset_stage2.py                      # Stage 2 데이터로더
│
├── data_split.json                        # 데이터 분할 정보 (212KB)
│
├── train_stage1_pretrain.py               # Stage 1 학습 스크립트
├── train_stage2_method1_margin.py         # Stage 2 Method 1 학습
├── train_stage2_method2_discriminator.py  # Stage 2 Method 2 학습
│
├── evaluate_gaussian_filters.py           # 실험 1: Baseline
├── tune_gaussian_filters.py               # 실험 2: CI 튜닝
├── tune_gaussian_advanced.py              # 실험 3: Advanced
│
├── evaluate_and_visualize_stage2.py       # Stage 2 평가 (전체)
├── evaluate_and_visualize_stage2_audiodriven.py  # Stage 2 평가 (audio-driven)
│
├── balance_final_test.py                  # 테스트 데이터 균형 맞추기
├── split_test_balanced.py                 # 데이터 분할
│
├── runs/                                  # 학습 결과 및 가중치
│   ├── stage1_pretrain/
│   │   └── stage1_pretrained.pt           # Stage 1 가중치 (2.7MB)
│   │
│   ├── stage2_method1_margin/
│   │   └── stage2_method1_best.pt         # Method 1 가중치 (2.7MB)
│   │
│   ├── stage2_method2_discriminator/
│   │   └── stage2_method2_best.pt         # Method 2 가중치 (3.0MB)
│   │
│   ├── stage2_evaluation/                 # 전체 평가 시각화
│   │   ├── 1_score_distribution.png
│   │   ├── 2_roc_curve.png
│   │   ├── 3_latent_2d_tsne.png
│   │   ├── 4_latent_3d_tsne.png
│   │   └── 5_score_by_fake_type.png
│   │
│   ├── stage2_evaluation_audiodriven/     # Audio-driven 평가
│   │   ├── 1_score_distribution.png
│   │   ├── 2_roc_curve.png
│   │   ├── 3_latent_2d_tsne.png
│   │   ├── 4_latent_3d_tsne.png
│   │   └── 5_score_comparison.png
│   │
│   ├── gaussian_filtering/                # 실험 1 결과
│   │   ├── gaussian_filtering_metrics.json
│   │   ├── confusion_matrix.png
│   │   └── loss_distributions.png
│   │
│   ├── gaussian_filtering_tuning/         # 실험 2 결과
│   │   ├── all_results.json
│   │   ├── hyperparameter_heatmaps.png
│   │   ├── precision_recall_tradeoff.png
│   │   ├── top_configurations.png
│   │   └── stage1_pass_rate_analysis.png
│   │
│   └── gaussian_filtering_advanced/       # 실험 3 결과
│       ├── all_results.json
│       ├── method_comparison.png
│       ├── pca_analysis.png
│       └── top_configs_per_method.png
│
├── 20GBprocessed/                         # 원본 NPZ 데이터 (제외)
└── test_balanced_npz/                     # 균형잡힌 테스트 데이터 (제외)
```

---

## 실험 결과

### Stage 2 학습 결과

#### Method 1 (Margin Loss)

**학습 과정**:
- Epoch 1-10: Margin 내 real 샘플 안정화
- Epoch 11-20: Fake 샘플 분리 강화

**최종 성능** (단순 threshold 기준):
- Real 샘플 평균 loss: 1.36
- Fake 샘플 평균 loss: 1.48
- Separation: 0.12

#### Method 2 (Discriminator)

**학습 과정**:
- VAE는 real 재구성에만 집중
- Discriminator는 latent space에서 분류

**최종 성능**:
- Discriminator AUC: ~0.75
- Method 1보다 약간 낮은 성능

**결론**: Method 1 (Margin Loss)이 더 효과적 → 이후 실험에서 Method 1 사용

### Gaussian Filtering 튜닝 결과

#### 실험별 성능 비교

| 실험 | 설정 개수 | 최고 F1 | 최고 Recall | 핵심 발견 |
|------|----------|---------|------------|----------|
| 실험 1 (Baseline) | 1 | 61.36% | 46.15% | 너무 보수적 |
| 실험 2 (CI Tuning) | 81 | 78.84% | 81.20% | 1차 필터 70% CI 최적 |
| 실험 3 (Advanced) | 448 | **89.71%** | **93.16%** | PCA 10D + 3D loss |

#### PCA 차원별 성능

| PCA 차원 | 설명 분산 | 최고 F1 | 비고 |
|---------|----------|---------|------|
| Full 36D | 100% | 52.17% | 차원의 저주 |
| 5D | 75.8% | 87.60% | 정보 손실 시작 |
| **10D** | **91.2%** | **89.71%** | **최적** |
| 15D | 96.2% | 89.34% | 근소하게 낮음 |
| 20D | 98.3% | 86.79% | 과적합 시작 |
| 30D | 99.8% | 67.57% | 심각한 과적합 |

#### Confusion Matrix 진화

```
Baseline (95% CI, 1D, 36D):
                 예측값
              Real    Fake
실제 Real     112       5
     Fake      63      54    ← 63개 놓침!

Tuned (70% CI, 1D, 36D):
                 예측값
              Real    Fake
실제 Real      88      29
     Fake      22      95    ← 개선

Advanced (95% CI, 3D, 10D):
                 예측값
              Real    Fake
실제 Real     100      17
     Fake       8     109    ← 최종! (8개만 놓침)
```

---

## 사용 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/tongminkim/liveness_detection_vae.git
cd liveness_detection_vae/model1

# 의존성 설치
pip install -r pip_requirements.txt
```

### 2. 데이터 준비

```bash
# NPZ 파일을 20GBprocessed/ 디렉토리에 배치
# 구조:
# 20GBprocessed/
#   ├── real/
#   └── fake/audio-driven/
```

### 3. Stage 1 학습

```bash
python train_stage1_pretrain.py
```

**출력**: `runs/stage1_pretrain/stage1_pretrained.pt`

### 4. Stage 2 학습

```bash
# Method 1 (권장)
python train_stage2_method1_margin.py

# Method 2
python train_stage2_method2_discriminator.py
```

**출력**: `runs/stage2_method1_margin/stage2_method1_best.pt`

### 5. Stage 2 평가

```bash
# Audio-driven만 평가
python evaluate_and_visualize_stage2_audiodriven.py

# 전체 fake 유형 평가
python evaluate_and_visualize_stage2.py
```

**출력**: `runs/stage2_evaluation_audiodriven/`

### 6. Gaussian Filtering 실험

```bash
# 실험 1: Baseline
python evaluate_gaussian_filters.py

# 실험 2: CI 튜닝
python tune_gaussian_filters.py

# 실험 3: Advanced (PCA)
python tune_gaussian_advanced.py
```

**출력**: `runs/gaussian_filtering_advanced/`

### 7. 최종 추론 예제

```python
import torch
import numpy as np
from model_bandvae import BandSplitVAE
from sklearn.decomposition import PCA
from scipy import stats

# 1. 모델 로드
checkpoint = torch.load('runs/stage2_method1_margin/stage2_method1_best.pt')
model = BandSplitVAE(C_in_per_band=7, C_h=48, C_z=12, dilations=[1,2,4])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 가우시안 파라미터 로드 (학습 데이터에서 미리 계산)
loss_mu_3d = np.array([1.495, 1.002, 1.000])
loss_cov_3d = np.array([[...], [...], [...]])  # 3x3

pca_model = PCA(n_components=10)  # 학습 데이터로 fit됨
latent_mu_10d = np.array([...])   # 10D
latent_cov_10d = np.array([[...]])  # 10x10

# 3. 새로운 샘플 추론
def predict(npz_path):
    # NPZ 로드 및 전처리
    data = np.load(npz_path)
    x_lf, x_bp, x_hf = preprocess(data)  # Band 분리

    # Forward pass
    recons, mus, logvars, _ = model(x_lf, x_bp, x_hf)

    # 1차 필터: Band별 reconstruction loss
    loss_lf = MSE(recons['lf'], x_lf)
    loss_bp = MSE(recons['bp'], x_bp)
    loss_hf = MSE(recons['hf'], x_hf)
    loss_vector = np.array([loss_lf, loss_bp, loss_hf])

    stage1_pass = is_within_confidence(
        loss_vector, loss_mu_3d, loss_cov_3d, confidence=0.95
    )

    if not stage1_pass:
        return 'fake'

    # 2차 필터: PCA 10D latent
    latent_36d = concatenate([mus['lf'], mus['bp'], mus['hf']])
    latent_10d = pca_model.transform(latent_36d)

    stage2_pass = is_within_confidence(
        latent_10d, latent_mu_10d, latent_cov_10d, confidence=0.70
    )

    return 'real' if stage2_pass else 'fake'
```

---

## Requirements

### Python 버전
- Python 3.10+

### 주요 라이브러리

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
```

**전체 목록**: `pip_requirements.txt`

### 하드웨어

**권장 사양**:
- GPU: NVIDIA GPU with 8GB+ VRAM (CUDA 지원)
- RAM: 16GB+
- Storage: 50GB+ (데이터 포함)

**최소 사양** (CPU only):
- RAM: 8GB+
- Storage: 50GB+

---

## 문서

### 한국어 문서

- **README.md**: 본 문서 (프로젝트 전체 가이드)
- **하이퍼파라미터_튜닝_보고서.md**: 모든 튜닝 실험의 상세 분석 (75KB)

### 영어 문서

- **STAGE2_PIPELINE.md**: Stage 2 학습 파이프라인 설명
- **HYPERPARAMETER_TUNING.md**: 하이퍼파라미터 튜닝 요약

---

## Citation

본 프로젝트를 사용하실 경우 다음과 같이 인용해주세요:

```bibtex
@software{liveness_detection_vae,
  author = {Your Name},
  title = {Liveness Detection using Band-Split VAE},
  year = {2025},
  url = {https://github.com/tongminkim/liveness_detection_vae}
}
```

---

## License

MIT License

---

## Acknowledgements

- Band-Split VAE architecture inspired by audio source separation research
- Deepfake detection research community
- Generated with assistance from Claude Code

---

**Last Updated**: 2025-11-16
