좋습니다. 최종 **Objective(손실) 함수**와 **2-Stage 학습 시나리오(원클래스→소량 스푸프 반영)**를 핵심만 깔끔히 정리합니다.

---

# 1) 최종 Objective (Band-VAE + TCN)

## 기본 구성 (Stage 1에서도 동일)

대역별 재구성 + VAE 정규화 + 대역 정직교/합성 일관성(+옵션 항목):

$\boxed{\mathcal{L} ;=; \underbrace{\sum_{b\in{LF,BP,HF}}\lambda_b |x_b-\hat x_b|1}{\mathcal{L}{rec}}
;+;\underbrace{\beta{LF}KL(q(z_{LF})|N) + \beta_{BP}\sum_t KL(q(z_t^{BP})|N)+\beta_{HF}\sum_t KL(q(z_t^{HF})|N)}{\mathcal{L}{KL}}
;+;\underbrace{\gamma!!\sum_{i<j}!corr(\hat x_i,\hat x_j)^2}{\mathcal{L}{decorr}}
;+;\underbrace{\zeta ,|x-(\hat x_{LF}+\hat x_{BP}+\hat x_{HF})|1}{\mathcal{L}{mix}}
;+;[\eta,\mathcal{L}{band} ;+; \alpha,\mathcal{L}{sync} ;+; \delta,\mathcal{L}{instab}]_{\text{선택}}}$

- $(\mathcal{L}{band})$*: STFT 기반 대역 에너지 중복 억제(필요 시),
$(\mathcal{L}{sync}, \mathcal{L}_{instab}):$* 기존 시계열 지표를 (원본 vs 복원) 또는 대역별로 계산해 가중합.
- 위 항들은 문서의 정의를 그대로 따릅니다.
- 모델/파이프라인 개요(대역분리→공유 TCN→3 VAE헤드→합성)는 다음과 같습니다.

---

# 2) Stage 1 — 원클래스(진짜 영상만) 사전학습

목표: **진짜(live)**에서 “각 대역 자체복원 능력 + 대역 간 독립성”을 학습해, 스푸핑 시 오차가 자연스럽게 커지도록 만듭니다.

- **학습 데이터**: live만.
- **손실**: 위의 $(\mathcal{L})$ 그대로(분류 항 없음).
- **출력/점수(추론용)**: 대역별 오차 통계, KL·스펙트럼 불일치 등으로 **원클래스 점수 $(S_{ano})$** 계산.

    $(;S_{ano}=w_{LF}E_{LF}+w_{BP}E_{BP}+w_{HF}E_{HF})$ (필수 코어 지표들은 문서 정의대로 구성)

- **임계값**: 개발셋에서 ACER 최소 등 기준으로 (\tau) 선택.
- 설계/수식 레퍼런스는 문서 Stage-1 절에 정리되어 있습니다.

> 핵심: “대역별 자기복원 월등함”을 먼저 만들면, 스푸프는 자연히 높은 오차/불일치를 보입니다.
>

---

# 3) Stage 2 — 소량 스푸프로 “더 멀어지게” 만드는 구체화

Stage-1 가중치를 기반으로 **few-shot 스푸프**를 활용해 결정경계를 날카롭게 합니다. 아래 3가지가 실전에서 간단하고 강합니다.

## 3.1 “점수 마진(hinge)” 방식: **오류 기반 마진 학습**

Stage-1의 **원클래스 점수**를 그대로 사용해, live와 spoof의 **점수 마진**을 강제:

[

$\mathcal{L}{margin}
=;\mathbb{E}{x\sim live},[\max(0,, S(x)-\tau_{live})]
;+;\mathbb{E}{x\sim spoof},[\max(0,, \tau{spoof}-S(x))]$
]

- 직관: live는 **낮은 점수**(복원 잘 됨), spoof는 **높은 점수**(복원 나쁨)를 갖도록 **소프트 마진**을 건다.
- (S)는 문서의 코어 지표(E_LF/BP/HF, ΔSpec, KL_sum 등) 가중합으로 구현.
- 간단하고 **few-shot에 강함**(분류기 없이도 경계 정교화).

## 3.2 **Feature-space 중심(센터) + 분리** 방식

Stage-1 잠재/피처(예: (z_{LF}), 프레임 (z_t^{HF}), 또는 φ)를 사용해,

live는 중심으로 모으고(spoof와의 마진), spoof는 떨어뜨립니다:

[

$\mathcal{L}{center}
=\underbrace{| \phi(x{live})-c|2^2}{\text{live 집단 중심}}
;+;\underbrace{\max\big(0,, m-|\phi(x_{spoof})-c|2\big)}{\text{spoof 마진}}$
]

- (\phi): 문서의 특징 벡터(코어 15~25D 권장) 혹은 (H/z) 요약.
- 장점: **적은 스푸프**로도 견고한 분리.

## 3.3 **경량 분류기(φ 전용) + 칼리브레이션**

Stage-1을 **freeze**(또는 low-LR)하고, φ로 작은 MLP/로지스틱을 학습(BCE/Focal, class-weight).

플랫/아이소토닉으로 **확률 보정** 후 최종 임계.

---

## 3.4 “스푸프를 더 못 맞추도록” 직접 유도(선택)

- **HF 가중 상향**: (\mathcal{L}*{rec})에서 (\lambda*{HF}!\uparrow) (HF에 위조 아티팩트가 몰릴 때). 문서의 의도와 부합.
- **GRL(Gradient Reversal) 또는 Max-rec on spoof**:

    스푸프 배치에 대해 (-\mathcal{L}_{rec})로 역전파(혹은 GRL) → **스푸프 복원 악화** 유도(대신 live 안정성 저하 주의).

- **대역 정직교 강화**: 스푸프에서 (\mathcal{L}_{decorr}) 가중치를 올려, “대역 간 에너지 섞임”을 더 억제.

---

## 3.5 Stage-2 최종 손실 예시(간단 조합)

분류기 없이 **점수 마진**만 쓰는 버전(간결/견고):

[

\boxed{

\mathcal{L}^{(2)} ;=; \mathcal{L};(\text{Stage-1의 } \mathcal{L}*{rec}!+!\mathcal{L}*{KL}!+!\mathcal{L}*{decorr}!+!\mathcal{L}*{mix})

;+;\rho ,\mathcal{L}_{margin}

}

]

분류기를 함께 쓰는 버전:

[

\boxed{

\mathcal{L}^{(2)} ;=; \mathcal{L};+;\rho ,\mathcal{L}_{BCE}\big(\text{MLP}(\phi(x)), y\big)

\quad(\text{class-weighted, Focal 가능})

}

]

- φ의 구성과 소형 분류기 권장은 문서 2.1~2.4 절을 그대로 따르면 됩니다(특징 리스트, 학습/보정, 융합).

---

# 4) 실무 체크리스트

- **Stage-1**: live만으로 (\mathcal{L}) 최소화(대역별 복원↑, 정직교/합성 일관성 유지).
- **Stage-2**:
    1. **소수의 스푸프** 수집(도메인별 1~K장 수준),
    2. φ 추출하여 **점수 마진** 또는 **경량 분류기** 학습(+플랫 보정),
    3. 필요 시 HF 가중↑, VAE는 **freeze 또는 저학습률**.
- **최종 추론**: (S = w\cdot S_{cls} + (1-w)\cdot \text{Norm}(S_{ano})) 융합도 유효(개발셋으로 (w) 튜닝).

---

필요하시면 위 수식 그대로 돌아가는 **미니 학습 루프(파이토치, Stage-1/2 스켈레톤)**도 바로 만들어드릴게요.
