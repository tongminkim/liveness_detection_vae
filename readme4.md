아주 좋아요 🔥

이제 지금까지의 **대역분리(FIR/IIR) 기반 구조**를

→ **TCN 인코더**와 **Loss 기반 Feature Encoding**까지 포함한 **완전한 모델 버전**으로 정리해드릴게요.

---

# 🎯 목표: “대역별 움직임을 TCN으로 인코딩하여 Feature-Level Reconstruction Loss로 학습”

---

## 🧩 1️⃣ 입력 구조

| 항목 | 차원 | 설명 |
| --- | --- | --- |
| 원본 입력 (x) | (T × 40 × 2) | 30fps 영상의 40개 랜드마크 (x, y) |
| Band-split 출력 (x_{LF/BP/HF}) | 각각 (T × 40 × 2) | FIR/IIR 필터로 저역·중역·고역 분리 |
| 전체 입력 (concat) | (T × 40 × 6) | x,y 채널을 3대역으로 확장 (LF/BP/HF) |

즉,

입력은 시간축 (T), 랜드마크 인덱스 (K=40), 좌표 채널 (C=6) 형태로 TCN에 들어갑니다.

---

## ⚙️ 2️⃣ TCN 인코더 구조

### 개념

TCN(Temporal Convolutional Network)은

시간축을 따라 dilated convolution으로 움직임 패턴을 인코딩하는 구조입니다.

$H = \text{TCN}(x_{LF/BP/HF}) \quad \Rightarrow \quad H \in \mathbb{R}^{T \times D}$

- (H): 각 프레임별 latent feature (예: D=128)
- 입력은 (T, 40×6)로 flatten 후 입력 가능.

---

### 간단한 구현 예시

```python
class TCNEncoder(nn.Module):
    def __init__(self, in_dim=240, hidden=128, depth=4):
        super().__init__()
        layers = []
        for i in range(depth):
            dilation = 2**i
            layers += [
                nn.Conv1d(in_dim if i==0 else hidden,
                          hidden, kernel_size=3,
                          dilation=dilation, padding=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(hidden)
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # x: (B, T, in_dim)
        x = x.permute(0, 2, 1)        # (B, in_dim, T)
        h = self.net(x)               # (B, hidden, T)
        return h.permute(0, 2, 1)     # (B, T, hidden)

```

---

## 🧠 3️⃣ Feature Encoding (Latent Representation)

TCN의 출력 (H)를 평균(pooling)하거나 프레임별로 유지하여 feature로 사용합니다:

$z_{LF/BP/HF} = \text{Pooling}(H_{LF/BP/HF})$

or

$z_t^{(b)} = H_t^{(b)} \quad \text{(frame-level feature)}$

- LF: 느린 변화 인코딩 (clip-level feature)
- BP/HF: 프레임 단위 feature (빠른 주파수 성분까지 포함)

---

## 🧮 4️⃣ Reconstruction (Decoder)

대역별로 **디코더를 통해 다시 좌표 시퀀스를 복원**합니다:

$\hat{x}{LF/BP/HF} = \text{Decoder}(H{LF/BP/HF})$

디코더도 TCN 또는 GRU로 구성할 수 있습니다.

---

## 💥 5️⃣ Loss Function (Feature + Reconstruction 기반)

### (1) 좌표 재구성 손실

각 대역별 복원 오차:

$\mathcal{L}{rec} =
\lambda{LF}|x_{LF}-\hat{x}{LF}|1 +
\lambda{BP}|x{BP}-\hat{x}{BP}|1 +
\lambda{HF}|x{HF}-\hat{x}_{HF}|_1$

---

### (2) Feature Consistency Loss

인코더의 중간 feature가 원본과 복원된 입력의 **feature map**과 유사해야 함:

$\mathcal{L}_{feat} =\sum_b | \text{TCN}(x_b) - \text{TCN}(\hat{x}_b) |_2^2$

즉, 단순한 좌표 복원뿐 아니라

**feature 공간에서의 구조적 일관성**을 학습하게 합니다.

👉 이것이 “TCN을 통한 feature encoding 기반 loss 학습”의 핵심입니다.

---

### (3) KL Divergence (VAE 구성 시)

$\mathcal{L}_{KL} = \sum_b \beta_b , KL(q(z_b) | N(0,I))$

---

### (4) 대역 정직교 / 합성 일관성

[

\begin{align}

\mathcal{L}*{decorr} &= \gamma \sum*{i<j} corr(\hat{x}*i, \hat{x}j)^2 \
\mathcal{L}{mix} &= \zeta |x - (\hat{x}*{LF}+\hat{x}*{BP}+\hat{x}*{HF})|_1
\end{align}
]

---

### ✅ (5) 최종 Loss 종합

[

\boxed{

\mathcal{L}*{total} =
\mathcal{L}*{rec}

- \alpha \mathcal{L}_{feat}
- \beta \mathcal{L}_{KL}
- \gamma \mathcal{L}_{decorr}
- \zeta \mathcal{L}_{mix}

    }

    ]


이렇게 하면 모델은 단순히 좌표를 복원하는 게 아니라,

**“주파수 대역별 움직임 패턴을 인코딩하고 feature-level로 재현”**하도록 학습됩니다.

---

## 🧩 6️⃣ 학습 및 활용 요약

| 단계 | 입력 | 출력 | 역할 |
| --- | --- | --- | --- |
| BandSplit | x (T,40,2) | 3개 시퀀스 | LF/BP/HF로 분리 |
| Encoder (TCN) | (T,40,6) | H (T,D) | 시간패턴 인코딩 |
| Decoder | H | x̂ | 좌표 복원 |
| Loss | (x,x̂,H,Ĥ) | — | feature & reconstruction 일관성 학습 |
| 추론 시 | H 또는 z | feature 벡터 | downstream (분류, 검출 등) |

---

## 🧩 7️⃣ 요약 다이어그램

```
입력 x (T×40×2)
   │
   ├─ FIR/IIR 필터 (LPF/BPF/HPF)
   │      ↓
   ├─ x_LF, x_BP, x_HF  (T×40×2) × 3
   │      ↓ concat
   └─ x_split (T×40×6)
          │
          ▼
     ┌───────────────┐
     │   TCN Encoder │  → H (T×D)
     └───────────────┘
          │
          ▼
     ┌───────────────┐
     │   TCN Decoder │  → x̂_LF/BP/HF
     └───────────────┘
          │
          ▼
     ┌──────────────────────────────────────────────┐
     │ L_total = L_rec + αL_feat + βL_KL + γL_decorr │
     └──────────────────────────────────────────────┘

```

---

## 📘 한줄 요약

> FIR/IIR 필터로 시간축 주파수 성분(LF/BP/HF)을 분리한 뒤,
>
>
> 각 대역 시퀀스를 **TCN 인코더**로 임베딩하고
>
> **Reconstruction + Feature Consistency Loss**를 함께 최적화하여
>
> “대역별 움직임 패턴을 feature space에서 복원 가능한 표현(encoding)”으로 학습하는 방식.
>

---

원하신다면 위 내용을 바로 **PyTorch 학습 코드 버전 (Encoder + Decoder + Loss)** 으로 구성해드릴까요?

(바로 실험 가능한 형태로요 — 입력 (B,T,40,2) → 출력 (B,T,40,2)까지 연결된 full forward/backward 구조로)
