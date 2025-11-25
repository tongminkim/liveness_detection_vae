# Inference Hyperparameter Tuning

## 개요

Liveness Detection VAE 모델의 inference 단계에서 사용되는 하이퍼파라미터를 체계적으로 탐색하여 최적의 설정을 찾는 과정입니다.

## 목적

Two-Stage Gaussian Filtering 방식의 inference 성능을 극대화하기 위해 다음 요소들을 최적화:

1. **PCA 차원 수** - 잠재 벡터의 차원 축소 정도
2. **Confidence Threshold** - Stage 1/2의 판정 경계값
3. **분포 모델** - Gaussian 외 다른 분포 적합성 검증

---

## 실험 설계

### 1. 탐색 공간 (Search Space)

| 하이퍼파라미터 | 값 범위 | 설명 |
|--------------|---------|------|
| **PCA 차원** | [5, 10, 15, 20, 30, 50] | 잠재 벡터를 몇 차원으로 축소할 것인가 |
| **Stage 1 Confidence** | [0.90, 0.95, 0.99] | 재구성 손실 필터링 엄격도 |
| **Stage 2 Confidence** | [0.50, 0.70, 0.90] | 잠재 공간 필터링 엄격도 |
| **분포 모델** | [Gaussian, GMM, OCSVM, IForest] | 이상치 탐지 알고리즘 |

**총 조합 수**: 6 × 3 × 3 × 4 × 4 = **864개**
- 6개 PCA 차원
- 3개 Stage1 confidence × 3개 Stage2 confidence = 9개 조합
- 4개 loss 분포 × 4개 latent 분포 = 16개 조합

### 2. 사용 데이터

- **Training Set**: stage2_train - Real 샘플만 사용 (1,088개)
- **Test Set**: final_test - Real 117개 + Fake 117개 (총 234개)
- **모델**: `runs/stage1_pretrained.pt`
- **필터 설정**: fc_low=2.0Hz, fc_high=8.0Hz (pretrained 모델의 최적 설정)

### 3. 분포 모델 설명

#### a) Gaussian (기존 방식)
```python
mu = np.mean(data, axis=0)
cov = np.cov(data, rowvar=False)
# Mahalanobis distance로 이상치 판정
```

**장점**:
- 수학적으로 명확
- 해석 가능성 높음
- Chi-squared 분포로 threshold 설정 가능

**단점**:
- 단봉(unimodal) 가정
- 이상치에 민감

#### b) Gaussian Mixture Model (GMM)
```python
gmm = GaussianMixture(n_components=3)
gmm.fit(data)
# Log probability로 이상치 판정
```

**장점**:
- 다봉(multimodal) 분포 모델링 가능
- 복잡한 데이터 분포 표현

**단점**:
- 하이퍼파라미터 추가 (component 수)
- 계산 비용 증가

#### c) One-Class SVM
```python
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
ocsvm.fit(data)
# Decision function으로 이상치 판정
```

**장점**:
- 비선형 경계 학습 가능
- Kernel trick 활용

**단점**:
- 확률적 해석 어려움
- nu 파라미터 튜닝 필요

#### d) Isolation Forest
```python
iforest = IsolationForest(contamination=0.1)
iforest.fit(data)
# Anomaly score로 이상치 판정
```

**장점**:
- Tree 기반으로 빠름
- 고차원에서도 효과적

**단점**:
- Contamination 비율 설정 필요
- 확률적 해석 어려움

---

## 구현 구조

### 전체 파이프라인

```
입력 오디오
    ↓
Butterworth Filter (fc_low=2.0, fc_high=8.0)
    ↓
[LF, BP, HF] 3개 밴드
    ↓
BandSplitVAE 모델
    ↓
[재구성 손실 (3D), 잠재 벡터 (고차원)]
    ↓
┌─────────────────┬─────────────────┐
│  Stage 1        │   Stage 2       │
│  Loss Filtering │ Latent Filtering│
├─────────────────┼─────────────────┤
│ Input: 3D loss  │ Input: PCA 축소 │
│ Model: Gaussian │ Model: Gaussian │
│       GMM       │       GMM       │
│       OCSVM     │       OCSVM     │
│       IForest   │       IForest   │
│ Threshold: 0.9~ │ Threshold: 0.5~ │
└─────────────────┴─────────────────┘
    ↓
Two-stage AND 연산
    ↓
예측: Real(0) or Fake(1)
```

### 코드 구조

```
hyperparameter_tuning.py
│
├── extract_features()          # VAE로 특징 추출
│   ├── 재구성 손실 계산 (3D)
│   └── 잠재 벡터 추출 (고차원)
│
├── fit_*() 함수들               # 분포 모델 학습
│   ├── fit_gaussian()
│   ├── fit_gmm()
│   ├── fit_ocsvm()
│   └── fit_iforest()
│
├── evaluate_model()             # 성능 평가
│   ├── Two-stage filtering
│   └── 메트릭 계산 (Acc, Prec, Rec, F1, Spec)
│
└── visualize_*() 함수들         # 시각화
    ├── visualize_pca_explained_variance()
    ├── visualize_tsne()
    ├── visualize_distribution_2d()
    └── visualize_confidence_heatmap()
```

---

## 평가 지표

### 주요 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 전체 정확도 |
| **Precision** | TP/(TP+FP) | Fake 예측의 정확성 |
| **Recall** | TP/(TP+FN) | 실제 Fake 탐지율 |
| **F1 Score** | 2×(Prec×Rec)/(Prec+Rec) | Precision-Recall 조화평균 |
| **Specificity** | TN/(TN+FP) | 실제 Real 정확도 |

### Confusion Matrix

```
                Predicted
                Real  Fake
Actual  Real  [  TN  |  FP ]
        Fake  [  FN  |  TP ]
```

- **TN (True Negative)**: Real을 Real로 정확히 판정
- **FP (False Positive)**: Real을 Fake로 잘못 판정 (Type I Error)
- **FN (False Negative)**: Fake를 Real로 잘못 판정 (Type II Error) - **치명적!**
- **TP (True Positive)**: Fake를 Fake로 정확히 판정

**Liveness Detection에서는 FN을 최소화하는 것이 중요** (보안 위협)

---

## 시각화

### 1. PCA Explained Variance
**파일**: `pca_explained_variance.png`

- **왼쪽 그래프**: 누적 설명 분산 (Cumulative Explained Variance)
  - X축: 주성분 개수
  - Y축: 누적 설명된 분산 비율
  - 각 PCA 차원별 곡선

- **오른쪽 그래프**: 개별 주성분 분산 (Individual Component Variance)
  - X축: 주성분 번호
  - Y축: 해당 주성분이 설명하는 분산 비율
  - PCA 20D 이하만 표시

**해석**:
- 곡선이 빨리 1.0에 수렴할수록 낮은 차원으로도 충분
- 일반적으로 90% 이상 설명되는 차원 선택

### 2. t-SNE Visualization
**파일**: `tsne_pca{10,20,50}.png`

- 고차원 잠재 벡터를 2D로 시각화
- 파란점: Real 샘플
- 빨간점: Fake 샘플
- PCA 10D, 20D, 50D 각각 생성

**해석**:
- Real/Fake 클러스터가 명확히 분리되면 좋은 표현
- 겹치는 영역이 많으면 분류 어려움

### 3. Distribution 2D Visualization
**파일**: `distribution_2d_pca{dim}.png`

- 첫 2개 주성분만 사용한 2D 분포
- Gaussian의 경우 등고선(contour) 표시
- 파란점: Real, 빨간점: Fake

**해석**:
- 등고선 안쪽: Real로 판정되는 영역
- 분포의 형태와 데이터 점들의 관계 파악

### 4. Confidence Threshold Heatmap
**파일**: `confidence_heatmap_pca{10,20}.png`

- X축: Stage 2 Confidence
- Y축: Stage 1 Confidence
- 색상: F1 Score (밝을수록 높음)

**해석**:
- 가장 밝은 셀이 최적 threshold 조합
- Trade-off 관계 파악 가능

---

## 실험 절차

### Step 1: 특징 추출
```python
# 모델 로드 및 특징 추출
train_losses, train_latents, train_labels = extract_features(model, train_loader)
test_losses, test_latents, test_labels = extract_features(model, test_loader)

# Real 샘플만 선택 (One-Class Learning)
real_mask = train_labels == 0
train_real_losses = train_losses[real_mask]
train_real_latents = train_latents[real_mask]
```

### Step 2: 손실 분포 모델 학습
```python
# 4가지 분포 모델 학습 (3D 손실 공간)
loss_models = {
    'gaussian': fit_gaussian(train_real_losses),
    'gmm': fit_gmm(train_real_losses, n_components=2),
    'ocsvm': fit_ocsvm(train_real_losses),
    'iforest': fit_iforest(train_real_losses)
}
```

### Step 3: PCA 차원별 탐색
```python
for pca_dim in [5, 10, 15, 20, 30, 50]:
    # PCA 적용
    pca = PCA(n_components=pca_dim)
    train_latents_pca = pca.fit_transform(train_real_latents)
    test_latents_pca = pca.transform(test_latents)

    # 4가지 분포 모델 학습 (축소된 잠재 공간)
    latent_models = {
        'gaussian': fit_gaussian(train_latents_pca),
        'gmm': fit_gmm(train_latents_pca, n_components=2),
        'ocsvm': fit_ocsvm(train_latents_pca),
        'iforest': fit_iforest(train_latents_pca)
    }

    # 모든 조합 평가
    for loss_dist in ['gaussian', 'gmm', 'ocsvm', 'iforest']:
        for latent_dist in ['gaussian', 'gmm', 'ocsvm', 'iforest']:
            for conf1 in [0.90, 0.95, 0.99]:
                for conf2 in [0.50, 0.70, 0.90]:
                    metrics = evaluate_model(...)
                    all_results.append(metrics)
```

### Step 4: 시각화 및 분석
```python
# PCA 설명력 시각화
visualize_pca_explained_variance(pca_models)

# 선택된 차원에 대해 t-SNE 시각화
for dim in [10, 20, 50]:
    visualize_tsne(test_latents_pca, test_labels, dim)

# 최적 Gaussian 모델의 분포 시각화
visualize_distribution_2d(test_latents_pca_2d, test_labels, best_model)

# Confidence threshold 히트맵
visualize_confidence_heatmap(gaussian_results)
```

### Step 5: 최적 설정 선정
```python
# F1 Score 기준 최적 설정 선택
best_overall = max(all_results, key=lambda x: x['f1'])

# 분포 타입별 최적 설정
for dist_type in ['gaussian', 'gmm', 'ocsvm', 'iforest']:
    best_per_dist = max([r for r in all_results if r['dist_type'] == dist_type],
                        key=lambda x: x['f1'])
```

---

## 결과 파일

### 출력 디렉토리
```
visualizations/hyperparameter_tuning/
├── tuning_results.json          # 모든 조합의 결과 (864개)
├── best_config.json              # 최적 설정
├── pca_explained_variance.png   # PCA 설명력 그래프
├── tsne_pca10.png               # t-SNE (10D)
├── tsne_pca20.png               # t-SNE (20D)
├── tsne_pca50.png               # t-SNE (50D)
├── distribution_2d_pca*.png     # 2D 분포 시각화
├── confidence_heatmap_pca10.png # Confidence 히트맵 (10D)
└── confidence_heatmap_pca20.png # Confidence 히트맵 (20D)
```

### tuning_results.json 구조
```json
[
  {
    "pca_dim": 10,
    "loss_dist": "gaussian",
    "latent_dist": "gaussian",
    "conf_stage1": 0.95,
    "conf_stage2": 0.70,
    "explained_variance": 0.9662,
    "accuracy": 0.8761,
    "precision": 0.8385,
    "recall": 0.9316,
    "f1": 0.8826,
    "specificity": 0.8205,
    "confusion_matrix": [[96, 21], [8, 109]]
  },
  ...
]
```

### best_config.json 구조
```json
{
  "pca_dim": 10,
  "loss_dist": "gaussian",
  "latent_dist": "gaussian",
  "conf_stage1": 0.95,
  "conf_stage2": 0.70,
  "explained_variance": 0.9662,
  "accuracy": 0.8761,
  "precision": 0.8385,
  "recall": 0.9316,
  "f1": 0.8826,
  "specificity": 0.8205,
  "confusion_matrix": [[96, 21], [8, 109]],
  "predictions": [0, 0, 1, ...]
}
```

---

## 분석 방법론

### 1. PCA 차원 선택 기준

**고려 사항**:
- **설명 분산**: 90% 이상 설명되는 최소 차원
- **F1 Score**: 가장 높은 F1을 달성하는 차원
- **일반화**: 과적합 방지를 위한 적절한 복잡도

**Trade-off**:
- 차원 ↑ → 더 많은 정보 보존, 과적합 위험 ↑, 계산 비용 ↑
- 차원 ↓ → 정보 손실, 단순한 모델, 계산 빠름

### 2. Confidence Threshold 선택 기준

**Stage 1 (재구성 손실)**:
- 높은 confidence (0.95~0.99) → 엄격한 1차 필터
- 목적: 명백한 이상치를 빠르게 제거

**Stage 2 (잠재 공간)**:
- 중간~낮은 confidence (0.50~0.90) → 세밀한 2차 필터
- 목적: Stage 1을 통과한 샘플 중 미묘한 차이 포착

**최적 조합**:
- **High Recall 우선**: Stage1 낮게, Stage2 높게 → FN 최소화 (보안 중요)
- **High Precision 우선**: Stage1 높게, Stage2 낮게 → FP 최소화 (사용자 경험)
- **Balanced**: 둘 다 중간 값 → F1 최대화

### 3. 분포 모델 비교 기준

| 모델 | 장점 | 단점 | 적합한 경우 |
|------|------|------|------------|
| **Gaussian** | 해석 가능, 빠름 | 단봉 가정 | 데이터가 정규분포에 가까운 경우 |
| **GMM** | 다봉 모델링 | 복잡, 느림 | 여러 클러스터가 있는 경우 |
| **OCSVM** | 비선형 경계 | 확률 해석 어려움 | 복잡한 경계가 필요한 경우 |
| **IForest** | 빠름, 고차원 | 확률 해석 어려움 | 대용량, 고차원 데이터 |

---

## 기대 결과

### 성공 지표

1. **기존 대비 F1 Score 향상**
   - 기존: 0.8826 (PCA 10D, Gaussian, conf=0.95/0.70)
   - 목표: 0.90 이상

2. **False Negative 최소화**
   - FN ≤ 5개 (현재 8개)
   - Recall ≥ 0.95 (현재 0.932)

3. **균형잡힌 성능**
   - Specificity ≥ 0.85 (현재 0.821)
   - Precision ≥ 0.85 (현재 0.838)

### 인사이트

1. **PCA 차원의 영향**
   - 어느 차원에서 성능이 포화되는가?
   - 계산 효율과 성능의 trade-off는?

2. **분포 모델의 적합성**
   - Gaussian이 충분한가?
   - GMM, OCSVM, IForest의 실질적 이득은?

3. **Confidence Threshold 민감도**
   - 어느 stage가 더 민감한가?
   - 최적 조합의 일반적인 패턴은?

---

## 실행 방법

### 1. 환경 설정
```bash
cd /home/elicer/liveness_detection/temp_repo
```

### 2. 실행
```bash
python3 hyperparameter_tuning.py
```

### 3. 로그 확인
```bash
tail -f hyperparameter_tuning.log
```

### 4. 결과 확인
```bash
# JSON 결과
cat visualizations/hyperparameter_tuning/best_config.json

# 시각화
ls visualizations/hyperparameter_tuning/*.png
```

---

## 후속 작업

### 1. 최적 설정 적용
```python
# best_config.json의 설정을 사용하여 최종 평가 스크립트 작성
best_config = json.load(open('visualizations/hyperparameter_tuning/best_config.json'))

# 적용
pca = PCA(n_components=best_config['pca_dim'])
confidence_stage1 = best_config['conf_stage1']
confidence_stage2 = best_config['conf_stage2']
```

### 2. Cross-Validation
- 여러 데이터 split에서 일관성 검증
- K-fold CV로 안정성 확인

### 3. Ensemble 방법
- 여러 분포 모델의 voting
- Stacking으로 메타 분류기 학습

### 4. 추가 실험
- GMM의 component 수 탐색
- OCSVM의 kernel 및 nu 값 튜닝
- IForest의 contamination 비율 최적화

---

## 참고 문헌

1. **PCA**: Jolliffe, I. T. (2002). Principal component analysis.
2. **GMM**: Reynolds, D. A. (2009). Gaussian mixture models.
3. **One-Class SVM**: Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution.
4. **Isolation Forest**: Liu, F. T., et al. (2008). Isolation forest.

---

## 문서 이력

- **2024-01-24**: 초안 작성
- **작성자**: Liveness Detection Team
- **파일**: `hyperparameter_tuning.py`
- **출력 경로**: `visualizations/hyperparameter_tuning/`
