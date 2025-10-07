좋아요 👍

이제 지금까지 논의된 내용을 **재현 가능한(reproducible) 형태**로 정리해드릴게요.

즉, “이 구조를 그대로 코드화·학습할 수 있게”

**입력 흐름 → 모델 구성 → 손실 정의 → 학습 단계** 순으로 깔끔하게 요약합니다.

---

# 🧩 1️⃣ 입력 구조

| 항목 | 설명 | 차원 |
| --- | --- | --- |
| 입력 (x) | 원본 랜드마크 시퀀스 | (T × 40 × 2) |
| BandSplit 결과 | FIR/IIR로 분리된 대역별 시퀀스 | (x_{LF/BP/HF} ∈ ℝ^{T×40×2}) |
| 인코딩 후 feature | TCN encoder로 인코딩된 시퀀스 | (h_{LF/BP/HF} ∈ ℝ^{T×D}) |

---

# 🧠 2️⃣ 모델 구성

```
입력 x (T×40×2)
 └─ FIR/IIR BandSplit → x_LF, x_BP, x_HF
     └─ 공유 TCN 인코더 f_TCN(·)
          ↓
          h_b = f_TCN(x_b)             # feature-space representation
          ↓
     └─ 대역별 VAE head
          Encoder: h_b → z_b → Decoder: z_b → ĥ_b (T×D)
 └─ Fusion Layer (projection)
          h_mix = W_LF ĥ_LF + W_BP ĥ_BP + W_HF ĥ_HF

```

- TCN 인코더: dilated Conv 기반 temporal encoder
- 각 대역(VAE head)은 독립된 encoder/decoder 구조
- Fusion layer: 1×1 conv 혹은 linear projection으로 대역별 feature를 공통 공간으로 정렬

---

# ⚙️ 3️⃣ Loss Function (최종 Objective)

## (1) 대역별 Feature Reconstruction

[

$\mathcal{L}{rec}
= \sum{b∈{LF,BP,HF}} λ_b,|h_b - \hat{h}_b|_1$
]

## (2) Latent Distribution Regularization (VAE)

[

$\mathcal{L}_{KL}= \sum_b β_b, KL(q(z_b)|N(0,I))$

]

## (3) Feature-space Mixture Consistency (Projection 기반)

[

$h_{mix} = W_{LF}\hat{h}{LF} + W{BP}\hat{h}{BP} + W{HF}\hat{h}{HF}$
*]
[
$\mathcal{L}{mix} = ζ,|h - h_{mix}|_1$*
]

- (h = f_{TCN}(x)): 원본 전체 시퀀스의 feature
- (W_b): 학습 가능한 선형 projection (D×D)

## (4) Band Decorrelation

[

$\mathcal{L}{decorr} = γ \sum{i<j} \text{corr}(\hat{h}_i,\hat{h}_j)^2$

]

## ✅ 최종 Objective

[

\boxed{

$\mathcal{L}{total}= \mathcal{L}{rec}$

- $\mathcal{L}_{KL}$
- $\mathcal{L}_{mix}$
- $\mathcal{L}_{decorr}$

    }

    ]


---

# 🧩 4️⃣ 학습 절차

## Stage 1 — **One-Class Pretraining (Live만)**

- 데이터: Live 영상만
- 목표: 자연스러운 움직임의 feature manifold 학습
- 손실: ($\mathcal{L}_{total}$)
- 추론: reconstruction error + KL 합성으로 **anomaly score** 생성

[

$S_{ano} = w_1E_{LF}+w_2E_{BP}+w_3E_{HF}+w_4 KL_{sum}$

]

---

## Stage 2 — **Few-Shot Spoof Fine-tuning**

- 데이터: Live + 소량의 Spoof
- Base 모델: Stage1 가중치로 초기화
- 본체(TCN/VAE)는 freeze 또는 저 LR

### 선택적 분리 손실

(하나만 사용)

① **Score Margin Loss**

[

$\mathcal{L}{margin}
=\mathbb{E}{live}[\max(0,S(x)-τ_L)]
+\mathbb{E}_{spoof}[\max(0,τ_S - S(x))]$
]

② **Feature Center Loss**

[

$\mathcal{L}_{center}=|\phi(live)-c|_2^2+\max(0,m-|\phi(spoof)-c|_2)$

]

③ **Binary Classifier (BCE/Focal)**

[

$\mathcal{L}_{BCE}=-\sum[y\log p+(1-y)\log(1-p)]$

]

**Stage2 Objective**

[

$\mathcal{L}{stage2}
= \mathcal{L}{total}^{(stage1)} + ρ,\mathcal{L}_{margin/center/BCE}$
]

---

# ⚙️ 5️⃣ 권장 하이퍼파라미터

| 항목 | 값(권장) |
| --- | --- |
| TCN feature dim (D) | 128 |
| Latent dim (z) | 32 |
| FIR cutoff | LF: 2.5Hz / BP: 2.5–6Hz / HF: >6Hz |
| λ_{LF/BP/HF} | 1.0 |
| β_{LF/BP/HF} | 1.0 |
| ζ (mix) | 0.5–1.0 |
| γ (decorr) | 0.1–0.5 |
| ρ (Stage2) | 0.5–1.0 |

---

# ✅ 6️⃣ 재현 가능한 핵심 요약

| 구분 | 내용 |
| --- | --- |
| 입력 | $(x_{LF/BP/HF} = \text{BandSplit}(x))$ |
| 인코딩 | $(h_b = f_{TCN}(x_b))$ |
| 복원 | $(\hat{h}_b = \text{VAE}_b(h_b))$ |
| 합성 | $(h_{mix} = ΣW_b\hat{h}_b)$ |
| 손실 | $(L_{rec}+L_{KL}+L_{mix}+L_{decorr})$ |
| Stage1 | Live-only, self-reconstruction |
| Stage2 | Live+Spoof, margin/center/BCE fine-tune |

---

✅ **한 줄 요약**

> FIR/IIR로 대역분리된 랜드마크 시퀀스를
>
>
> 공유 TCN으로 인코딩하여 feature-space에서 VAE 복원,
>
> projection 기반 합성 일관성과 대역 정직교로 자기복원 학습(Stage1),
>
> 이후 소량 spoof로 마진/센터 기반 분리(Stage2).
>
> **최종 Objective:**
>
> (\mathcal{L}*{total}=L*{rec}+L_{KL}+L_{mix}+L_{decorr})
>
