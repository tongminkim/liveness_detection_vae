https://arxiv.org/html/2412.01373v1

# 1) 직관과 목표

- 입술 랜드마크 시퀀스에는 **느린 대역(아티큘레이션 리듬)**, **중간 대역(음절/프레임 수준 움직임)**, **고속/미세 대역(잔진동, 재생 플리커, 압축 노이즈 등)**이 공존.
- 기존 LF/HF 병렬 두 갈래보다, **다층 스케일로 쪼개고 스케일 간 생성적 의존성**을 주면:
    - 저역이 **큰 궤적**을 먼저 설명,
    - 그 위에 중역이 **세부 패턴**을 보강,
    - 최상위 고역이 **잔차/미세 아티팩트**를 추가 → **설명력↑, 분해능↑**.

---

# 2) 대역 분해 옵션 (두 가지 중 택1 또는 혼합)

**A. 고정 필터/웨이블릿 기반 분해(권장 시작점)**

- 시간영역 1D FIR/BPF/HPF 또는 **다중해상도 웨이블릿(예: Haar/Db4)**로 `x → {x_LF, x_BP, x_HF}` (또는 `{a3, d3, d2, d1}`)로 분해.
- **장점**: 안정/해석 용이.
- **단점**: 고정 필터라 데이터 적응성↓.

**B. 학습가능 필터뱅크**

- **Depthwise Conv1d**(groups=채널) 커널을 웨이블릿/FIR로 초기화, **대역 마스크/정규화**(대역 외 에너지를 패널티)로 passband를 유지하며 학습.
- **장점**: 도메인 적응성↑, 성능↑.
- **단점**: 초기엔 대역 누수 가능 → 마스크/정규화 필요.

> 처음엔 A로 안정적으로 돌리고, 성능 튜닝 단계에서 B로 확장 추천.
>

---

# 3) 계층적 생성 모형 (Ladder/Hierarchical VAE 스타일)

## 3.1 생성 과정(Top-down prior)

스케일을 1(저역)–3(고역)으로 두고, **상위 스케일이 하위 스케일의 prior를 결정**:

[

$\begin{aligned}
&\text{(Level 1, 저역)} && z_1 \sim \mathcal{N}(0, I) \
&\text{(Level 2, 중역)} && z_2 \sim \mathcal{N}\big(\mu_2(z_1), \operatorname{diag},\sigma_2^2(z_1)\big) \
&\text{(Level 3, 고역)} && z_3 \sim \mathcal{N}\big(\mu_3(z_2), \operatorname{diag},\sigma_3^2(z_2)\big) \
&\text{디코더} && \hat{x}_1 = D_1(z_1), \quad \hat{x}_2 = D_2(z_2, \hat{x}_1), \quad \hat{x}_3 = D_3(z_3, \hat{x}_1, \hat{x}_2) \
&\text{합성} && \hat{x} = \hat{x}_1 + \hat{x}_2 + \hat{x}_3 \quad\text{(대역 합성 / 웨이블릿 역변환)}
\end{aligned}$

]

- (D_s): 시간 U-Net/TCN 디코더(스킵/업샘플 포함).
- **조건부 디코딩**: 상위 레벨 출력을 하위 레벨 디코더에 컨텍스트로 제공(위상/크기 정합 안정화).

## 3.2 추론(Posterior) 네트워크

- **Bottom-up 통계** + **Top-down prior**를 융합하는 Ladder VAE 스타일:


    $q(z_3 \mid x),\quad q(z_2 \mid x, z_3),\quad q(z_1 \mid x, z_2)$

- 실전 구현은 **공유 TCN 인코더** (H\in\mathbb{R}^{T\times D})에서 스케일별 통계 헤드를 분기:
    - Level1(clip-level): (H) → pooling → ($(\mu_1,\log\sigma_1^2))$
    - Level2/3(frame-level or window): (H_t) → $((\mu_{2,t},\log\sigma_{2,t}^2),(\mu_{3,t},\log\sigma_{3,t}^2))$

---

# 4) 손실(ELBO + 스케일/대역 특화 규제)

## 4.1 멀티-스케일 ELBO

스케일별 재구성 타깃을 분리(대역분해 사용 시):

[

$\mathcal{L}\text{ELBO}
= \sum{s=1}^{3} \Big(
\underbrace{\mathbb{E}{q}[\log p(x_s \mid z{\le s})]}{\text{재구성}}
;-;
\underbrace{\beta_s, \mathrm{KL}(q(z_s \mid \cdot),|,p(z_s \mid z{<s}))}_{\text{KL}}
\Big),$
]

- (x_1=x_{LF},; x_2=x_{BP},; x_3=x_{HF}) (웨이블릿이면 대응 스케일 계수).
- **β-밸런싱**: 상위 레벨에 더 큰 β(( \beta_1 \ge \beta_2 \ge \beta_3))로 정보 배분을 유도.

## 4.2 합성 일관성

- 대역/스케일 합성 결과가 원본과 맞도록:

    [

    $\mathcal{L}_\text{mix}=|;x - (\hat{x}_1+\hat{x}_2+\hat{x}_3);|_1$

    ]


## 4.3 대역 정직교/누수 억제(필수)

- 시간영역 상호상관/공분산 억제:

    [

    $\mathcal{L}\text{decorr}=\gamma\sum{i<j} \mathrm{corr}(\hat{x}_i,\hat{x}_j)^2$

    ]

- 주파수 마스크 일치(대역 누수 방지):

    [

    $\mathcal{L}\text{band}=\eta \sum{s} \big|, (1-M_s)\odot \mathcal{F}(\hat{x}_s), \big|_2^2$

    ]

    - (M_s): 스케일 s의 통과 대역 마스크(STFT/DFT 도메인).

## 4.4 스케일별 구조적 규제(권장)

- **저역(레벨1)**: **시간 매끈함**(TV/2차차분), **AR(1)/GP prior**로 리듬 보존.
- **고역(레벨3)**: **희소성/스파이크** 유도(( |\hat{x}_3|_1 ) 또는 Group-Lasso).
- **중역(레벨2)**: 저역/고역 사이 **분리 유지**(Mutual Info 낮추기; 간단히 cross-cov 패널티).

> 최종 손실 예시
>
>
> [
>
> $\mathcal{L}$
>
> $= \mathcal{L}\text{ELBO} + \mathcal{L}\text{mix}$
>
- $\mathcal{L}\text{decorr} + \mathcal{L}\text{band}$
- $\lambda_1 \mathcal{R}\text{smooth}^{(L1)} + \lambda_3 \mathcal{R}\text{sparse}^{(L3)}.$

    ]


---

# 5) 네트워크 구성(권장 사양)

**입력**: (x\in\mathbb{R}^{T\times K\times 2}) (랜드마크), 선택: Δ/Δ² 추가 → 채널 확장.

**공유 인코더**: 1D **TCN** (dilations 1,2,4,8,16), 채널 64→128.

**스케일 분기**:

- Level1 Head(clip): mean+attn pooling → (d_1=32)
- Level2 Head(frame): per-frame → (d_2=32)
- Level3 Head(frame): per-frame → (d_3=64)

    **디코더**: 시간 U-Net/TCN. 레벨이 내려갈수록 깊이/폭 증가(특히 L3).


**웨이블릿 라우팅(옵션)**:

- FWD: (x\rightarrow{a_m,d_m,\dots}) (per-coord)
- 디코더는 각 계수망을 복원, **IWT**로 (\hat{x}) 재합성.
- 장점: 다운샘플/업샘플이 규격화되어 **스케일 정합**이 쉬움.

---

# 6) posterior collapse 방지 팁

- **KL free-bits**(각 (z_s)에 최소 KL) 또는 **KL-warmup/cyclical**.
- **KL balancing**: 상위 레벨에 정보가 모이도록 β 스케줄 차등화.
- **Skip-context**: (\hat{x}_{<s})를 (D_s) 컨텍스트에 넣어 학습 안정화.

---

# 7) Liveness 점수화(원클래스/세미슈퍼바이즈드 겸용)

**원클래스 점수(권장)**

- 스케일별 오차 통계: $(E_s = |x_s-\hat{x}_s|) (mean/p90/max)$
- 대역 비율: $(E_3/(E_1+\epsilon),; E_2/(E_1+\epsilon))$
- 밴드 누수 지표: $(\mathcal{L}_\text{band}) 항의 값$
- 스케일 규제 크기: $(\mathcal{R}\text{sparse}^{(L3)}, \mathcal{R}\text{smooth}^{(L1)})$
- 합성 일치: $(|x-(\hat{x}_1+\hat{x}_2+\hat{x}_3)|)$

→ 가중합으로 스코어 S 만들고 개발셋에서 τ 선정(ACER 최소).

**세미-슈퍼바이즈드(분류기 추가)**

- 위 지표로 **클립-레벨 φ 벡터** 만들고 **소형 분류기(MLP/로지스틱/XGBoost)** 학습(스푸프 소량일 때 강력).

---

# 8) PyTorch 의사 코드(핵심 흐름)

```python
# x: [B,T,K,2] -> (선택) Δ/Δ² 추가 후 [B,T,C]
x_bands = band_split(x)       # (LF,BP,HF) or wavelet coeffs
H = shared_tcn(features(x))   # [B,T,D]

# Posterior heads
mu1, logvar1 = head1_clip(H)
mu2, logvar2 = head2_frame(H)
mu3, logvar3 = head3_frame(H)

z1 = reparam(mu1, logvar1)                   # [B,d1]
z2 = reparam(mu2, logvar2)                   # [B,T,d2]
z3 = reparam(mu3, logvar3)                   # [B,T,d3]

# Top-down priors
p2_mu, p2_logvar = prior2(z1)
p3_mu, p3_logvar = prior3(pool_t(z2))        # or frame-wise

# Decoders (conditional)
x1_hat = dec1(z1)                            # [B,T,K,2] low-band
x2_hat = dec2(z2, x1_hat)                    # mid-band
x3_hat = dec3(z3, x1_hat, x2_hat)            # high-band
x_hat = x1_hat + x2_hat + x3_hat

# Losses
L_rec = L1(x_bands.LF, x1_hat) + L1(x_bands.BP, x2_hat) + L1(x_bands.HF, x3_hat)
KL1 = kl_normal(mu1, logvar1, 0, 0)
KL2 = kl_normal(mu2, logvar2, p2_mu, p2_logvar)
KL3 = kl_normal(mu3, logvar3, p3_mu, p3_logvar)
L_mix = L1(x, x_hat)
L_decorr, L_band = decorrelation_terms(x1_hat, x2_hat, x3_hat)

loss = (L_rec + beta1*KL1 + beta2*KL2 + beta3*KL3
        + L_mix + gamma*L_decorr + eta*L_band
        + smooth_L1(x1_hat) + sparse_L1(x3_hat))

```

---

# 9) 실험/어블레이션 체크리스트

1. **분해 방식**: 고정 대역 vs 학습가능 필터뱅크 vs 웨이블릿
2. **계층 prior**: 독립 (p(z_s)=\mathcal{N}(0,I)) vs 조건부 (p(z_s|z_{<s}))
3. **스케일 수**: 2-레벨(LF/HF) vs 3-레벨(LF/BP/HF)
4. **규제 항**: ( \mathcal{L}*\text{decorr}, \mathcal{L}*\text{band}, ) smooth/sparse on/off
5. **점수화**: 원클래스 S 단독 vs 분류기 결합
6. **도메인 일반화**: 교차 데이터셋(LivDet-Face, OULU, SiW 등 프로토콜 유사)

---

# 10) 장단점 및 기여 포인트

**장점**

- **생성 분해의 정합성**: “저→중→고” 누적 설명로직이 **설명가능**.
- **아티팩트 포착력**: 고역 전용 레벨이 **스푸핑 잔차**를 잘 흡수.
- **도메인 적응**: 학습가능 필터뱅크/웨이블릿 파라미터로 **데이터별 최적 대역** 학습 가능.

**기여로 어필할 점**

- **주파수-계층 결합 VAE**: 단순 병렬 LF/HF가 아닌 **조건부 prior**로 연결된 **계층형 생성모형**.
- **대역 누수 억제 손실**(frequency-mask decorrelation) 제안.
- **스케일별 구조규제**(저역-스무딩/고역-희소)로 **해석성+일반화** 동시 강화.

**주의점**

- 대역 누수/정렬 문제 → $( \mathcal{L}\text{band}, \mathcal{L}\text{decorr} )$로 제어.
- posterior collapse → KL-warmup, free-bits, β-balancing.
- 계산량 ↑ → TCN 경량화, 레벨별 채널 차등, mixed precision.

---

필요하면, 이 설계를 **당신 데이터셋(FPS, T, K)에 맞춘 정확한 하이퍼파라미터 표**와 **학습 스크립트 템플릿(PyTorch, wandb 로깅, 체크포인트, 곡선 저장)**까지 바로 만들어줄게.
