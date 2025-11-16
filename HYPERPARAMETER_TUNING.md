# Hyperparameter Tuning Strategy

## 목표
Band-Split VAE의 최적 하이퍼파라미터 찾기 (10시간 이내)

## 시간 제약
- **총 시간**: 10시간 (600분)
- **모델 크기**: ~22K parameters (매우 가벼움)
- **Training set**: 57,111 segments (live)
- **Test set**: 190 segments (live)
- **Fake set**: 190 segments (5가지 deepfake 기법)

## 선택한 전략: 옵션 2 (Efficient with Pruning)

### 설정
- **Trials**: 15
- **Epochs per trial**: 10
- **Pruning**: MedianPruner 사용
- **예상 시간**: 15 × 40분 (평균) = 600분 (10시간)

### 시간 분해
```
1 trial = 40-60분
├─ Training (10 epochs): 30-50분
├─ Evaluation (live + fake): 5-8분
└─ Visualization (6종): 3-5분
```

### Pruning 전략
- **Pruner**: MedianPruner
  - 처음 5개 trial은 끝까지 실행 (baseline 확보)
  - 3 epoch 이후부터 pruning 시작
  - Validation loss가 median보다 높으면 중단
- **효과**: 나쁜 trial 조기 중단 → 시간 30-40% 절약

---

## 탐색할 하이퍼파라미터

### 1. Band-Split 경계값
주파수 대역 분리 cutoff frequencies

| Parameter | Range | Description |
|-----------|-------|-------------|
| `fc_low` | [1.5, 2.0, 2.5] Hz | LF/BP 경계 (기본: 2.0) |
| `fc_high` | [7.0, 8.0, 9.0] Hz | BP/HF 경계 (기본: 8.0) |

**이론적 배경**:
- LF (Low-Frequency): 전체적인 움직임, 대화 리듬
- BP (Band-Pass): 주요 발화 모션 (핵심 liveness cue)
- HF (High-Frequency): 미세한 떨림, deepfake artifacts

### 2. Model Dimensions
VAE의 representation capacity

| Parameter | Range | Description |
|-----------|-------|-------------|
| `C_h` | [48, 64, 96] | Hidden channels (기본: 48) |
| `C_z` | [12, 16] | Latent dimension (기본: 12) |

**Trade-off**:
- 크게 → 더 많은 정보 포착, 하지만 overfitting 위험
- 작게 → 빠른 학습, 하지만 underfitting 위험

### 전체 탐색 공간
- Grid search: 3 × 3 × 3 × 2 = **54 조합** (불가능)
- Optuna TPE: **15 trials**로 유망한 영역 집중 탐색

---

## 평가 지표

### Primary Metric (Optuna Objective)
**Validation Reconstruction Loss** (Live data)
- One-class learning: Live 데이터만으로 학습
- 낮은 loss = Live 패턴을 잘 학습

### Secondary Metrics (분석용)
Trial 완료 후 test set에서 계산:
1. **Live Reconstruction Loss**: Live를 얼마나 잘 재구성하는가
2. **Fake Reconstruction Loss**: Fake의 재구성 특성 (논문: 더 낮을 수 있음)
3. **Separation**: Live vs Fake의 분리도
4. **Latent Space Quality**: PCA/t-SNE 시각화로 분리 확인

---

## 시각화 (각 trial마다 6종 생성)

### 1. Reconstruction Loss - Overall
- **Type**: Histogram (Live vs Fake)
- **X-axis**: Total reconstruction loss
- **Y-axis**: Frequency
- **목적**: Live와 Fake의 전체 분포 비교

### 2. Reconstruction Loss - 2D (HF vs LF)
- **Type**: Scatter plot
- **X-axis**: HF reconstruction loss
- **Y-axis**: LF reconstruction loss
- **Color**: Live (blue) / Fake (red)
- **목적**: 주파수 대역별 분리 패턴 확인

### 3. Reconstruction Loss - 3D (HF, BP, LF)
- **Type**: 3D scatter plot
- **Axes**: HF, BP, LF reconstruction loss
- **Color**: Live (blue) / Fake (red)
- **목적**: 3개 대역의 전체적인 분리 공간 확인

### 4. Latent Space - PCA PC1
- **Type**: Histogram (Live vs Fake)
- **X-axis**: PC1 score
- **Y-axis**: Frequency
- **목적**: 주요 분산 방향에서의 분리도 확인

### 5. Latent Space - t-SNE 2D
- **Type**: Scatter plot
- **Axes**: t-SNE component 1 & 2
- **Color**: Live (blue) / Fake (red)
- **목적**: 비선형 manifold에서의 클러스터링 확인

### 6. Training Curves (추가)
- **Type**: Line plot
- **X-axis**: Epoch
- **Y-axis**: Loss (train/val)
- **목적**: 학습 안정성 및 수렴 확인

---

## 디렉토리 구조

```
runs/hyperparameter_tuning/
├── study.log                          # 전체 진행 로그 (진행률 포함)
├── best_hyperparameters.json          # 최종 best hyperparameters
├── optuna_study.db                    # Optuna study database
│
├── trial_000_fc_low_1.5_fc_high_7.0_C_h_48_C_z_12/
│   ├── best.pt                        # Best model weights
│   ├── train.log                      # Training log (per-epoch)
│   ├── metrics.json                   # Evaluation metrics
│   ├── reconstruction_loss_overall.png
│   ├── reconstruction_loss_2d_hf_lf.png
│   ├── reconstruction_loss_3d.png
│   ├── latent_pca_pc1.png
│   ├── latent_tsne_2d.png
│   └── training_curves.png
│
├── trial_001_fc_low_2.0_fc_high_8.0_C_h_64_C_z_16/
│   └── ... (동일 구조)
│
└── ... (총 15개 trials)
```

---

## 실행 방법

### 1. Fake 데이터 전처리
```bash
python3 preprocess_fake_data.py
```
- 입력: `/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_변조`
- 출력: `/home/elicer/liveness_detection/model1/processed_fake_test` (190 segments)
- 5가지 기법에서 균등하게 샘플링

### 2. 하이퍼파라미터 튜닝 (백그라운드)
```bash
nohup python3 tune_hyperparameters.py > runs/hyperparameter_tuning/study.log 2>&1 &
```

### 3. 진행 상황 모니터링
```bash
# 실시간 로그 확인
tail -f runs/hyperparameter_tuning/study.log

# 진행률 확인
grep "Trial.*completed" runs/hyperparameter_tuning/study.log

# Optuna dashboard (선택사항)
optuna-dashboard sqlite:///runs/hyperparameter_tuning/optuna_study.db
```

### 4. 결과 확인
```bash
# Best hyperparameters
cat runs/hyperparameter_tuning/best_hyperparameters.json

# 각 trial 결과 비교
ls -lh runs/hyperparameter_tuning/trial_*/metrics.json
```

---

## 로그 포맷

### study.log
```
[2025-11-14 19:30:00] Starting hyperparameter tuning...
[2025-11-14 19:30:00] Total trials: 15, Epochs per trial: 10
[2025-11-14 19:30:00] Pruning: MedianPruner (startup=5, warmup=3)
[2025-11-14 19:30:01] ========================================
[2025-11-14 19:30:01] Trial 0/15 (0.0%)
[2025-11-14 19:30:01] Hyperparameters: fc_low=1.5, fc_high=7.0, C_h=48, C_z=12
[2025-11-14 19:30:01] ========================================
[2025-11-14 19:35:22] Epoch 1/10 - train_loss: 0.523, val_loss: 0.487 [5.3min]
[2025-11-14 19:40:43] Epoch 2/10 - train_loss: 0.421, val_loss: 0.398 [5.4min]
...
[2025-11-14 20:25:15] Trial 0 completed - Best val_loss: 0.312 [55.2min]
[2025-11-14 20:25:20] Evaluation: Live loss=0.315, Fake loss=0.289
[2025-11-14 20:25:30] Visualizations saved to trial_000_fc_low_1.5_fc_high_7.0_C_h_48_C_z_12/
[2025-11-14 20:25:31] ========================================
[2025-11-14 20:25:31] Trial 1/15 (6.7%)
...
[2025-11-14 05:30:00] ========================================
[2025-11-14 05:30:00] All trials completed! (10.1 hours)
[2025-11-14 05:30:00] Best trial: 7
[2025-11-14 05:30:00] Best val_loss: 0.287
[2025-11-14 05:30:00] Best hyperparameters: fc_low=2.0, fc_high=8.0, C_h=64, C_z=16
[2025-11-14 05:30:00] Results saved to runs/hyperparameter_tuning/best_hyperparameters.json
```

---

## 예상 결과

### Best Hyperparameters
예상되는 최적 조합 (참고용):
```json
{
  "fc_low": 2.0,
  "fc_high": 8.0,
  "C_h": 64,
  "C_z": 16,
  "val_loss": 0.287,
  "live_test_loss": 0.291,
  "fake_test_loss": 0.254,
  "separation_score": 0.037
}
```

### 패턴 예측
- **fc_low**: 2.0Hz가 최적일 가능성 (논문 기본값)
- **fc_high**: 7-8Hz 사이 (너무 높으면 noise 증가)
- **C_h**: 64가 sweet spot (48은 작고, 96은 과도)
- **C_z**: 16이 더 좋을 가능성 (충분한 latent capacity)

---

## 주의사항

### 1. Fake의 낮은 Reconstruction Loss
논문에 따르면 **Fake가 오히려 더 낮은 reconstruction loss**를 보일 수 있음:
- 이유: Smoother motion → easier to reconstruct
- 전략: 비정상적으로 낮은 reconstruction = spoof
- 해결: Multi-band analysis로 미묘한 차이 포착

### 2. Pruning 주의
- 처음 5개 trial은 pruning 없이 끝까지 실행
- Median 계산을 위한 baseline 필요
- 너무 공격적인 pruning은 좋은 trial을 놓칠 수 있음

### 3. 시각화 해석
- **분리도**: Live와 Fake가 명확히 구분되는가?
- **Overlap**: 겹치는 영역이 많으면 해당 조합은 비효율적
- **Latent space**: 클러스터가 형성되는가?

### 4. One-Class Learning
- Live 데이터만으로 학습
- Validation loss는 Live만 사용
- Fake는 evaluation에만 사용

---

## 다음 단계

튜닝 완료 후:
1. **Best hyperparameters로 최종 학습** (epochs=30-50)
2. **Full test set 평가** (모든 fake 데이터)
3. **Threshold 선택** (Youden's J or percentile)
4. **최종 성능 측정** (Accuracy, AUC, EER)
5. **논문과 비교** (POC: 73.5% 정확도 vs Band-Split)

---

## 참고
- 논문: TeamC_Research_proposal.pdf
- 프로젝트 개요: PROJECT.md
- 모델 구조: model_bandvae.py
- Config: config_bandvae.py
