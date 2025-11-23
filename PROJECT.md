# Liveness Detection Project (Band-Split VAE)

## 프로젝트 개요
Frequency-Decoupled Feature-Space VAE를 사용한 One-Class Lip-Liveness Detection 구현

**목표**: 입술 랜드마크의 시간적 패턴을 주파수 대역별로 학습하여 실제(live) 영상과 딥페이크(fake) 영상을 구분

**참고 논문**: TeamC_Research_proposal.pdf (model1 폴더)

**구현 대상**: 논문 Section 2의 **실제 제안 방법론** (Band-Split VAE)
- ✅ 3-band frequency decomposition (LF/BP/HF)
- ✅ Per-band feature-space VAE
- ✅ Learnable weighted fusion
- ❌ POC (Section 3의 single-stream version)는 구현하지 않음

---

## 데이터셋 정보

### 원본 데이터 경로
```
/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_원본/원본1/
```

### 데이터 구조
- 총 파일 수: **3,020개** (비디오 파일)
- 구조: `{person_id}/{person_id}_{video_num}.mp4`
- 영상 길이: 약 90초 내외
- 형식: MP4 비디오

### 전처리 완료
1. **5초 세그먼트로 분할** (디스크 절약, 고속 처리)
2. **MediaPipe로 얼굴 랜드마크 추출**
   - Left eye: 16 landmarks
   - Right eye: 16 landmarks
   - Lips outer: 21 landmarks
   - Lips inner: 21 landmarks
   - Face center: 1 landmark
   - **모델 사용**: Lips 40개만 사용 (outer 20 + inner 20)
3. **기하학적 정규화 (Procrustes/Umeyama alignment)**
   - Scale-rotation-translation 정규화
   - 프레임별 reference shape 정렬

---

## 방법론 (논문 Section 2 기반)

**논문의 핵심 아이디어**:
- OC-VAE (Khalid & Woo, 2020)의 one-class 학습 패러다임
- AG-FAS (Long et al., 2024)의 anomalous residual reasoning
- **본 논문의 기여**: Frequency-decoupled feature-space reconstruction

**논문과의 차이점**:
- 논문: FIR filter bank 사용
- 본 구현: **Butterworth IIR filter 사용** (병목 해소, 10-100배 빠름)
- 핵심 아이디어(주파수 대역 분리)는 동일하게 유지

### 1. Feature Engineering - 2가지 버전 구현

**본 프로젝트는 2가지 Feature 버전을 지원합니다:**

#### 【SIMPLE 버전】 Position + Velocity만 사용
입술 랜드마크에서 다음 특징들을 추출:
- **Position** (x, y): 정규화된 좌표 (2차원)
- **Velocity** (Δx, Δy): 1차 미분, fps 스케일링 (2차원)

**총 Feature 차원**: 40 landmarks × 4 features = **160 channels per band**

**특징**:
- 핵심 모션 정보만 사용 (위치 + 속도)
- 모델 경량화
- 학습 속도 빠름
- **3-band split**: LF/BP/HF 각각 160 channels
- 총 파라미터: **~156K** (3 VAEs)

#### 【FULL 버전】 논문과 유사한 모든 Feature 사용
입술 랜드마크에서 다음 특징들을 추출:
- **Position** (x, y): 정규화된 좌표 (2차원)
- **Velocity** (Δx, Δy): 1차 미분, fps 스케일링 (2차원)
- **Acceleration** (Δ²x, Δ²y): 2차 미분, fps² 스케일링 (2차원)
- **Angle** (θ): `atan2(Δy, Δx)` with temporal unwrapping (1차원)
- **Angle-rate** (Δθ): 각도의 시간 미분, fps 스케일링 (1차원)

**총 Feature 차원**: 40 landmarks × 8 features = **320 channels per band**

**특징**:
- 논문과 유사한 풍부한 feature
- 더 많은 정보 (가속도, 각도 변화율 등)
- Fake의 monotone alignment 포착 가능
- **3-band split**: LF/BP/HF 각각 320 channels
- 총 파라미터: **~191K** (3 VAEs)

**핵심 아이디어**:
- Position: 랜드마크의 절대 위치
- Velocity: 움직임의 방향과 속도
- Acceleration: 움직임의 변화 (급격한 움직임 vs 부드러운 움직임)
- Angle: 랜드마크 간 기하학적 관계
- Angle-rate: 각도의 시간적 변화 (회전 정보)
- **Fake 영상의 특징**: smoother한 패턴, low-variance한 모션, monotone alignment

### 2. Frequency-Decoupled Architecture (Band-Split VAE)

**논문 Section 2.1의 핵심 아키텍처 구현 (Figure 1)**:

논문의 전체 파이프라인:
```
Input: Lip landmarks [T, K, 2]  # T=150 frames (5초 @ 30fps), K=40 landmarks
  ↓
Step 1: Geometric Normalization (Section 2.2)
  - Umeyama similarity transform
  - Scale-rotation-translation 제거
  ↓
Step 2: Frequency Decomposition (Section 2.1)
  - 논문: "A lightweight FIR filter bank separates the signal into
           low-, mid-, and high-frequency streams"
  - 본 구현: Butterworth IIR filter (병목 해소)

  Filter Bank → 3 Frequency Bands:
  ├─ Low-Frequency (LF): 0 ~ 2Hz    (전체적인 움직임, 대화 리듬)
  ├─ Band-Pass (BP): 2 ~ 8Hz        (주요 발화 모션)
  └─ High-Frequency (HF): 8 ~ 15Hz  (미세한 떨림, artifact)
  ↓
Step 3: Per-Band Feature Engineering (Section 2.2)
  - "construct per-landmark channels: position, velocity, acceleration,
     angle/slope, angle-rate"

  각 대역에서 derivatives 계산:
  - LF band: [x_L, Δx_L, Δ²x_L, θ_L, Δθ_L]  → [C_in, T]
  - BP band: [x_B, Δx_B, Δ²x_B, θ_B, Δθ_B]  → [C_in, T]
  - HF band: [x_H, Δx_H, Δ²x_H, θ_H, Δθ_H]  → [C_in, T]

  where C_in = K × F_dim:
    - FULL: K=40 × F_dim=8 = 320 channels
    - SIMPLE: K=40 × F_dim=4 = 160 channels
  ↓
Step 4: 3 Independent Feature-Space VAEs (Section 2.1)
  - "for each band b ∈ {L, B, H}, the encoder fb takes the concatenated
     feature tensor and outputs (μb, log σ²b)"

  ┌─────────────────┬─────────────────┬─────────────────┐
  │ VAE_LF          │ VAE_BP          │ VAE_HF          │
  │ Encoder_L       │ Encoder_B       │ Encoder_H       │
  │   ↓ μ_L, σ²_L   │   ↓ μ_B, σ²_B   │   ↓ μ_H, σ²_H   │
  │ z_L ~ N(μ,σ²)   │ z_B ~ N(μ,σ²)   │ z_H ~ N(μ,σ²)   │
  │   ↓             │   ↓             │   ↓             │
  │ Decoder_L       │ Decoder_B       │ Decoder_H       │
  │   ↓ x̂_L         │   ↓ x̂_B         │   ↓ x̂_H         │
  │ Loss_L          │ Loss_B          │ Loss_H          │
  └─────────────────┴─────────────────┴─────────────────┘
  ↓
Step 5: Learnable Weighted Fusion (Section 2.1)
  - "a learnable projection-based fusion combines per-band reconstructions"

  x̂_fused = Σ_b W_b · x̂_b
  where W = softmax([w_L, w_B, w_H])
  ↓
Step 6: Anomaly Scoring (Section 2.4)
  - "compact feature-space norm and KL"

  S_ano = Σ_b [α·‖Λ_b ⊙ (F_b - F̂_b)‖₁ + β_b·KL_b]

  where:
    - Recon_b: per-band reconstruction error
    - KL_b: KL divergence per band
    - β_b: per-band weight (default 1.0)
```

**모델 구조 비교**:

| 항목 | SIMPLE | FULL |
|------|--------|------|
| Input channels per band (C_in) | 160 | 320 |
| Number of bands | 3 (LF/BP/HF) | 3 (LF/BP/HF) |
| Hidden channels (C_h) | 48 | 48 |
| Latent dimension (C_z) | 12 | 12 |
| **총 파라미터** | **~156K** | **~191K** |
| TCN layers per VAE | 3 encoder + 3 decoder | 3 encoder + 3 decoder |
| Dilations | [1,2,4] → [4,2,1] | [1,2,4] → [4,2,1] |
| Learnable fusion weights | 3 (w_L, w_B, w_H) | 3 (w_L, w_B, w_H) |

### 3. One-Class Learning
- **Live 데이터만으로 학습** (Fake는 test에만 사용)
- VAE가 live 데이터의 정상 분포를 학습
- Fake는 reconstruction error가 다르게 나타남
  - 논문 결과: Fake가 오히려 더 낮은 reconstruction error
  - 이유: Smoother motion → easier to reconstruct
  - 전략: **비정상적으로 낮은 reconstruction = spoof**

### 4. Loss Functions (논문 Section 2.3)

**논문의 Loss 구조**:

**Per-Band Feature-Space Reconstruction with Derivative-Aware Consistency** (Section 2.3):
```
L_rec = Σ_b ‖Λ_b ⊙ (F_b - F̂_b)‖₁
```
where:
- F_b = [x_b, Δx_b, Δ²x_b, θ_b, Δθ_b] (per band)
- Λ_b = diag(1, η₁, η₂, ν₁, ν₂): feature weighting
- ‖·‖₁: L1 norm (논문에서 명시)

**KL Regularization** (Section 2.3):
```
L_KL = Σ_b β_b · KL(q_b(z_b|F_b) ‖ p(z))
```
where:
- q_b(z_b|F_b): encoder의 posterior distribution
- p(z) = N(0, I): prior (standard Gaussian)
- β_b: per-band weight (본 구현에서는 모두 1.0)

**Total Loss** (Section 2.3):
```
L = L_rec + L_KL + L_mix + L_decor
```

본 구현에서는 L_mix와 L_decor는 사용하지 않음 (논문에서도 "small fusion penalty"):
```
L = L_rec + L_KL
```

**핵심 개념**:
- **Derivative-consistency**: difft(x̂_b) ≈ Δx̂_b, difft(Δx̂_b) ≈ Δ²x̂_b
  - Decoder가 직접 모든 derivatives를 출력
  - 시간적 일관성 보장
- **Angle & Angle-rate**: θ, Δθ가 fake의 "monotone alignment" 포착
  - Fake는 overly consistent motion → 낮은 reconstruction error
  - 논문: "making them easier to reconstruct and thus lowering their
            corresponding reconstruction losses"

---

## POC 실험 결과 (논문)
- 데이터: 424 live / 424 fake
- 정확도: **73.5%** (single-stream version, no band split)
- Reconstruction Loss:
  - Fake: 0.208 ± 0.07 (낮음)
  - Live: 0.319 ± 0.10 (높음)
- 임계값 전략: 20th percentile (rec < τ₂₀ → fake)

**Band-Split의 장점** (논문 제안):
- 주파수별로 다른 spoof artifacts 포착
- Low-freq: 전체적인 움직임 패턴
- High-freq: 미세한 떨림, artifact

---

## 구현 계획

### Phase 1: 데이터 전처리
- [x] 데이터셋 경로 확인 (3,020개 파일)
- [x] MediaPipe 랜드마크 추출 파이프라인 (병렬 처리)
- [x] 5초 세그먼트로 분할 (디스크 저장)
- [x] Procrustes 정규화 구현
- [x] Feature engineering (position + velocity + acceleration + angle + angle-rate)

### Phase 2: 모델 구현
- [x] FIR Filter Bank 구현 (LF/BP/HF 3-band split)
- [x] 3개 독립 TCN-VAE 구현
- [x] Learnable weighted fusion
- [x] Per-band reconstruction + KL loss
- [x] Anomaly scoring function

### Phase 3: 학습 및 평가
- [ ] One-class training (live only, 2 versions)
  - [ ] FULL version (5 features, 3-band)
  - [ ] SIMPLE version (2 features, 3-band)
- [ ] Anomaly scoring function
- [ ] Threshold selection (Youden's J or live-only percentile)
- [ ] Evaluation on fake data

---

## 환경 설정

### 필요 라이브러리
```bash
pip install torch torchvision
pip install opencv-python mediapipe
pip install numpy scipy scikit-learn
pip install ffmpeg-python
```

### 시스템 정보
- Working directory: `/home/elicer/liveness_detection/model1`
- Available disk: **190GB**
- Device: CUDA (GPU)

---

## 현재 진행 상황
- [x] 논문 리뷰 완료
- [x] POC 코드 분석 완료
- [x] 데이터셋 확인 완료 (3,020개 비디오)
- [x] 데이터 전처리 파이프라인 구현 완료 (병렬 처리, 5초 세그먼트)
- [x] 랜드마크 추출 완료 (~15,000개 세그먼트)
- [x] Feature Engineering 구현 완료 (2가지 버전)
- [x] **FIR Filter Bank 구현 완료 (3-band split)**
- [x] **Band-Split VAE 모델 구현 완료**
- [x] 학습 스크립트 구현 완료
- [ ] 모델 학습 및 평가 진행 예정

---

## 사용 방법

### 학습 실행

**FULL 버전 (5 features, 3-band)**:
```bash
python3 train_bandvae.py --mode full --epochs 20 --batch_size 64
```

**SIMPLE 버전 (2 features, 3-band)**:
```bash
python3 train_bandvae.py --mode simple --epochs 20 --batch_size 64
```

**자동 순차 학습 (FULL → SIMPLE)**:
```bash
chmod +x train_both_bandvae.sh
./train_both_bandvae.sh
```

**추가 옵션**:
```bash
python3 train_bandvae.py --mode full \
  --epochs 20 \
  --batch_size 64 \
  --lr 0.001 \
  --device cuda
```

### 모델 저장 경로
- FULL 버전: `runs/bandvae_full/best.pt`
- SIMPLE 버전: `runs/bandvae_simple/best.pt`

### 로그 확인
```bash
# FULL 버전 로그 (실시간)
tail -f runs/bandvae_full/train.log

# SIMPLE 버전 로그
tail -f runs/bandvae_simple/train.log
```

### 데이터셋 테스트
```bash
python3 dataset_bandvae.py  # 두 버전 모두 테스트
python3 config_bandvae.py   # Configuration 비교
python3 model_bandvae.py    # 모델 구조 테스트
```

---

## 주요 참고사항

### 1. Band-Split Parameters (Butterworth Filter)

**논문**: "A lightweight FIR filter bank separates the signal..."

**본 구현**: Butterworth IIR filter (병목 해소, 10-100배 빠름)

**Filter Parameters**:
- **fc_low**: 2.0 Hz (Low-pass cutoff)
- **fc_high**: 8.0 Hz (High-pass cutoff)
- **order**: 4 (Butterworth filter order, 논문의 FIR numtaps 대신)
  - Order 4-6이 일반적 (sharper cutoff vs. phase distortion trade-off)
  - SOS (second-order sections) format으로 numerical stability 보장

**Frequency Bands** (논문과 동일):
- **LF (Low-Frequency)**: 0 ~ 2 Hz
  - 전체적인 움직임, 대화 리듬
  - Live의 자연스러운 variation 포착
- **BP (Band-Pass)**: 2 ~ 8 Hz
  - 주요 발화 모션, 입술 움직임의 fundamental frequency
  - 핵심 liveness cue가 있는 대역
- **HF (High-Frequency)**: 8 ~ 15 Hz (Nyquist = 15Hz @ 30fps)
  - 미세한 떨림, micro-motion
  - Fake의 smoothing artifact 포착

### 2. Feature 차원
- **SIMPLE**: K=40 landmarks × 4 features = **160 channels per band**
- **FULL**: K=40 landmarks × 8 features = **320 channels per band**
- **Bands**: 3 (LF/BP/HF)

### 3. 시간 길이
- T=150 frames (5초 @ 30fps)

### 4. 정규화
- Umeyama similarity transform으로 scale/rotation/translation 제거

### 5. 처리 속도
- 병렬 처리 (7 workers) → 약 4.6초/비디오
- 현재 완료: ~15,000개 세그먼트

### 6. 파라미터 수
- **SIMPLE**: ~156K (3 VAEs × 52K each)
- **FULL**: ~191K (3 VAEs × 64K each)
- 논문 POC (single-stream): ~60K

---

## 파일 구조

```
model1/
├── PROJECT.md                      # 본 문서
├── TeamC_Research_proposal.pdf     # 논문
│
├── preprocess_data_fast.py         # 데이터 전처리 (병렬)
├── processed_live/                 # 전처리된 데이터 (~15K npz files)
│
├── model_bandvae.py                # Band-Split VAE 모델
├── dataset_bandvae.py              # Band-Split Dataset
├── config_bandvae.py               # Configuration (SIMPLE/FULL)
├── train_bandvae.py                # 학습 스크립트
├── train_both_bandvae.sh           # 자동 순차 학습 (FULL → SIMPLE)
│
└── runs/
    ├── bandvae_full/               # FULL 버전 결과
    │   ├── train.log
    │   ├── best.pt
    │   └── checkpoint_epoch*.pt
    └── bandvae_simple/             # SIMPLE 버전 결과
        ├── train.log
        ├── best.pt
        └── checkpoint_epoch*.pt
```

---

## 참고 자료
- 논문: `/home/elicer/liveness_detection/model1/TeamC_Research_proposal.pdf`
- 주요 기법:
  - OC-FakeDect (Khalid & Woo, 2020): One-class VAE for deepfake detection
  - AG-FAS (Long et al., 2024): Anomalous-cue guided anti-spoofing
  - **Frequency-Decoupled Feature-Space VAE**: Band-split architecture for fine-grained temporal modeling
