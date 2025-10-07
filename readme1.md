좋아요—이번 버전은 **실제 대역필터(LP/BP/HP)**로 신호를 물리적으로 분해한 뒤, 각 대역을 전용 VAE 헤드로 복원·합성하는 **대역분리형 멀티-헤드 VAE**입니다. (전 버전은 “전역 vs 로컬 관점” 분리였다면, 이번은 **신호처리 기반 스펙트럼 분리**가 핵심)

# 0) 개요

- 입력: 입술 랜드마크 시퀀스 `x ∈ R^{T×K×2}` (FPS=f)
- 전처리: 좌표 정규화(+옵션: Δ, Δ² 파생특성)
- **대역분리**: `x_LF = LPF(x)`, `x_BP = BPF(x)`, `x_HF = HPF(x)`
- **인코더**: 공유 TCN/GRU로 프레임 임베딩 `H ∈ R^{T×D}`
- **헤드**: `LF-VAE`, `BP-VAE`, `HF-VAE` (각각 자기 대역만 재구성)
- **합성**: `x̂ = x̂_LF + x̂_BP + x̂_HF`
- 목적: 대역별 복원 정밀도와 대역 간 정직교(Decorrelation)를 강제 → 스푸핑 아티팩트의 **특정 대역 집중**을 더 잘 드러냄

---

# 1) 대역필터 설계(권장값; FPS=f)

- 나이퀴스트: `f_N = f/2`. 컷오프는 **정규화 주파수** `ω_c = f_c / f_N`로 구현.
- 대역 권장(경험적 가이드; 튜닝 포인트):
    - **LF (slow articulation)**: `0 ~ 2.5 Hz` → `LPF(ω_c ≈ 2.5/f_N)`
    - **BP (articulatory mid-band)**: `2.5 ~ 6 Hz` → `BPF(ω_low=2.5/f_N, ω_high=6/f_N)`
    - **HF (micro-tremor/노이즈)**: `> 6 Hz` → `HPF(ω_c ≈ 6/f_N)`
- 필터 타입
    - 간단/안정: **FIR (windowed-sinc, Hamming, N=63~127)**, 지연 일정
    - 저지대역 날카롭게: **Butterworth/Chebyshev IIR**(bi-dir filtfilt로 위상왜곡 보정)
- 구현: 좌표 채널별로 독립 필터링(또는 K×2 전체에 동일 필터)

> 예: FPS=30 → f_N=15
>
> - LF: 0~2.5Hz, BP: 2.5~6Hz, HF: 6~15Hz

---

# 2) 모델 아키텍처

## 2.1 공유 인코더

- `x`(혹은 `concat[x, Δx, Δ²x]`) → 1D **TCN(64→128, dilation 1,2,4,8)** → `H[T,D]`
- 이유: 대역별 VAE가 **동일 표현 공간**에서 출발해야 상호 비교/정직교 제약이 잘 작동

## 2.2 대역별 VAE 헤드 (3개)

모두 같은 골격, 파라미터만 다르게:

- 인퍼런스 경로:
    - **LF-VAE** 입력: `x_LF`와 `H`를 조건으로 인코딩
        - 잠재: clip-level `z_LF ∈ R^{d_LF}` (예: 32)
        - 디코더: TCN/GRU로 `x̂_LF[T,K,2]` 복원
    - **BP-VAE** 입력: `x_BP`, `H` (프레임/짧은 윈도우 잠재 추천)
        - 잠재: frame-level `z_t^{BP} ∈ R^{d_BP}` (예: 32~64)
        - 디코더: 로컬 TCN으로 `x̂_BP`
    - **HF-VAE** 입력: `x_HF`, `H` (프레임/로컬)
        - 잠재: frame-level `z_t^{HF} ∈ R^{d_HF}` (예: 64)
        - 디코더: 얕은 TCN로 `x̂_HF`
- **조건부(추천)**: 각 디코더 입력에 `(해당 z) + (선택적으로 z_LF)`를 concat → 대역 간 위상/크기 정합 안정화

## 2.3 합성 및 잔차

- 최종 복원: `x̂ = x̂_LF + x̂_BP + x̂_HF`
- 대역별 오차: `E_LF=||x_LF−x̂_LF||`, `E_BP=…`, `E_HF=…`
- 스펙트럼 일치: `|STFT(x) − STFT(x̂)|` 보조 손실(가볍게)

---

# 3) 학습 목표(대표 손실)

각 대역별 **VAE Evidence Lower Bound (ELBO)** + 정합/정직교 보조항.

- 재구성(가중합):
    - `L_rec = λ_LF·||x_LF−x̂_LF||₁ + λ_BP·||x_BP−x̂_BP||₁ + λ_HF·||x_HF−x̂_HF||₁`
- KL:
    - `L_KL = β_LF·KL(q(z_LF)||N(0,I)) + β_BP·Σ_t KL(q(z_t^{BP})||N(0,I)) + β_HF·Σ_t KL(q(z_t^{HF})||N(0,I))`
- **대역 정직교(필수)**:
    - 시간영역 상호상관 억제:

        `L_decorr = γ·( corr(x̂_LF, x̂_BP)^2 + corr(x̂_LF, x̂_HF)^2 + corr(x̂_BP, x̂_HF)^2 )`

    - 또는 주파수 도메인 에너지 중복 억제:

        `L_band = η·(‖P_LF·Ŝ − Ŝ_LF‖² + …)` (P_*는 각 대역 마스크, Ŝ는 STFT)

- **합성 일관성**:
    - `L_mix = ζ·|| x − (x̂_LF+x̂_BP+x̂_HF) ||₁`
- (선택) 동기성/불안정성 보조(귀하의 기존 지표 연결):
    - `L_sync`, `L_instab`를 `x` vs `x̂` 또는 대역별로 계산해 가중합

> 최종: L = L_rec + L_KL + L_decorr + L_band + L_mix (+ α·L_sync + δ·L_instab)
>

---

# 4) 추론(스푸핑 점수)

- 기본: `S = w_LF·E_LF + w_BP·E_BP + w_HF·E_HF`
- 보조:
    - 스펙트럼 불일치 `ΔSpec = ‖STFT(x)−STFT(x̂)‖₁`
    - 대역별 이상 에너지 비율 `R_HF = ‖x_HF‖/‖x‖` 대비 `‖x̂_HF‖` 비정상 상승
    - (옵션) Δsync/instab 대역가중 합산
- 임계: 개발셋에서 τ 선택

---

# 5) 구현 팁

- **필터는 미분 불가**여도 OK(입력 분기용이므로 역전파 불필요). PyTorch에서 `torch.from_numpy`로 미리 컨볼루션 FIR 적용하거나, SciPy `filtfilt`로 오프로딩 후 텐서화.
- **지연 보정**: FIR 지연(`(N−1)/2`)은 세 대역 모두 동일 커널 길이로 맞추거나 `filtfilt` 사용.
- **에지 효과**: 패딩(반사/미러) 사용.
- **정직교 손실**은 배치·프레임 차원 평균으로 안정화.
- **파라미터 배분**: HF-VAE 디코더를 상대적으로 리치하게(스푸핑 아티팩트가 HF에 몰리는 경향).

---

# 6) 미니-스켈레톤(개념 코드: PyTorch 의사)

```python
class BandSplit:
    def __init__(self, fs, lf=2.5, bp=(2.5,6.0), hf=6.0, kind='fir', taps=127):
        # 미리 FIR 커널 생성 or IIR coeff 저장
        ...

    def __call__(self, x):  # x: [T,K,2]
        x_lf = lpf(x)  # same shape
        x_bp = bpf(x)
        x_hf = x - x_lf - x_bp  # 또는 직접 hpf(x)
        return x_lf, x_bp, x_hf

class SharedEncoder(nn.Module):
    def __init__(self, D=128): ...
    def forward(self, x_feat):  # [T,K,C]-> [T,D]
        return H

class VAE_Head(nn.Module):
    def __init__(self, latent_dim, mode='clip' or 'frame'):
        ...
    def forward(self, band_x, H):
        # mode=='clip': pool(H)->z   /  mode=='frame': per-frame z_t
        # decode -> x_hat_band
        return x_hat_band, kl

class BandVAE(nn.Module):
    def __init__(self):
        self.enc = SharedEncoder(D=128)
        self.lf = VAE_Head(32, mode='clip')
        self.bp = VAE_Head(48, mode='frame')
        self.hf = VAE_Head(64, mode='frame')
        self.split = BandSplit(fs=30)

    def forward(self, x):
        x_lf, x_bp, x_hf = self.split(x)
        H = self.enc(features(x))             # (option) concat Δ,Δ²
        xh_lf, kl_lf = self.lf(x_lf, H)
        xh_bp, kl_bp = self.bp(x_bp, H)
        xh_hf, kl_hf = self.hf(x_hf, H)
        xh = xh_lf + xh_bp + xh_hf
        return dict(xh=xh, xh_lf=xh_lf, xh_bp=xh_bp, xh_hf=xh_hf,
                    kl={'lf':kl_lf,'bp':kl_bp,'hf':kl_hf},
                    target={'lf':x_lf,'bp':x_bp,'hf':x_hf})

```

---

# 7) 언제 이 변형이 유리한가?

- **스푸핑 아티팩트가 특정 주파수대에 집중**될 때(재생 화면 플리커/압축 노이즈/소리 없는 입모양 복제 등), 대역별로 분해-복원하면 **해당 대역 오차가 선명히 분리**되어 판별력이 상승.
- **평가/설명성**: “HF에서만 이상치 급증”처럼 **대역별 근거 제시**가 쉬움.

---

---

→ 여기서 one class 로 학습 후에 각 vae loss에 간단한 분류기를 달아서 이진 분류 학습을 추가 진

# Stage 1 — 원클래스 사전학습 (Band-VAE)

- 입력: `x ∈ R^{T×K×2}` → FIR/IIR로 `x_LF, x_BP, x_HF`.
- 공유 인코더: TCN → `H ∈ R^{T×D}`.
- 3개 VAE 헤드: LF(clip-latent), BP/HF(frame-latent).
- 손실(예시):
    - 재구성: `L_rec = Σ_b λ_b ||x_b − x̂_b||₁` (b∈{LF,BP,HF})
    - KL: `L_KL = β_LF KL(z_LF) + β_BP Σ_t KL(z_t^BP) + β_HF Σ_t KL(z_t^HF)`
    - 대역 정직교: `L_decorr = γ Σ_{i<j} corr(x̂_i, x̂_j)^2`
    - 합성 일관성: `L_mix = ζ || x − (x̂_LF+x̂_BP+x̂_HF) ||₁`
    - (옵션) 시계열 지표: `L_sync, L_instab`
- 최종: `L_VAE = L_rec + L_KL + L_decorr + L_mix (+ αL_sync + δL_instab)`

이 단계에서 **스푸프 라벨 불필요**(live만).

# Stage 2 — “간단 분류기” 부착 (이진 학습)

VAE가 내는 신뢰도 좋은 신호들을 **특징 벡터 φ**로 모아, 경량 분류기를 학습합니다.

## 2.1 특징 벡터 φ (per-clip)

필수(재구성/통계):

- `E_LF, E_BP, E_HF`: 평균/중앙값/90퍼센타일/최대
- `E_ratio`: `E_HF / (E_LF + ε)`, `E_BP / (E_LF+ε)`
- `KL_sum`: 각 대역 KL의 합/평균
- `ΔSpec = ||STFT(x) − STFT(x̂)||₁` (대역별/전체)
- `R_HF = ||x_HF||₂ / ||x||₂` (상대 에너지)

시계열 안정성(선택):

- 프레임 jerk/가속도의 분산, 스파이크 횟수
- Δsync, Δsync_rel (원본 vs 복원)

잠재 요약(선택):

- `z_LF` (128→PCA/WHITEN→ 몇 차원)
- 프레임별 `z_t^HF`의 평균/분산/스펙트럼 에너지

→ `φ ∈ R^{~32–128}` 권장(간결·견고).

## 2.2 분류기 옵션

- **소형 MLP**(입력 φ → 2~3층 → σ): end-to-end 미세튜닝 가능
- **로지스틱/XGBoost/랜덤포리스트**: 작고 강건, 튜닝 쉬움
- 손실: class-weighted **BCE** 또는 **Focal**(불균형 완화)
- 캘리브레이션: **Platt/Isotonic** → 임계값 τ 선택(Youden/최소 ACER)

## 2.3 학습 시나리오 3종

1. **진짜 원클래스**(스푸프 전무):
    - 분류기 대신 **단일 점수 S_ano** = 가중합(`E_*`, `ΔSpec`, `KL_sum` 등)으로 임계치 결정.
    - 분류기 쓰고 싶다면 **합성 부정 샘플**을 만들어 semi-supervised로 학습:
        - 시간 셔플, 밴드 스와프(BP/HF 교차), 스펙트럼 노이즈 삽입, 랜드마크 위상 왜곡 등으로 **pseudo-spoof φ** 생성.
2. **few-shot 스푸프**(각 도메인 1~K장):
    - φ를 쌓아 소형 MLP/로지스틱을 학습(강한 L2/드롭아웃).
    - VAEs는 **freeze 또는 low-LR** 미세튜닝(특히 디코더는 freeze 권장).
3. **OC-SCMNet 스타일 부가 학습**(선택):
    - pseudo-negative φ를 더 다양화하려고, **메모리 뱅크**로 생성 φ의 코사인 중복 억제(간단 정규화 항 추가).

## 2.4 추론 결합

- **이중 점수 융합**:
    - `S_ano`(원클래스 점수)와 `S_cls`(분류기 출력)를 `S = w·S_cls + (1−w)·Norm(S_ano)`로 결합.
    - 가중치 w는 개발셋 ACER 최소화로 선택.

# 장점 / 리스크 / 대응

**장점**

- 원클래스의 **도메인 일반화** + 분류기의 **결정경계 학습** 결합.
- 입력 RGB/해상도/압축이 달라도 φ는 **오차·에너지·KL** 중심이라 **안정**.
- iBeta/LivDet 보고 지표(APCER/BPCER/ACER/EER) **임계 조정 용이**.

**리스크 & 대응**

- few-shot 과적합 → 드롭아웃·얼리스톱·강 L2, φ 차원 축소(PCA), K-fold로 도메인 홀드아웃.
- φ 누적 분포 드리프트 → **온-디바이스 리-캘리브레이션**(온도스케일/Isotonic).
- 분류기가 원본 특성으로 새는 현상 → **φ만 입력**(raw x/H 금지), VAE **freeze** 위주.

# 미니 의사 코드 (핵심 흐름)

```python
# Stage 1: train BandVAE on live only -> save weights
bandvae = BandVAE().train_on_live(live_loader)

# Stage 2: extract φ
X, y = [], []
for clip, label in dev_loader:           # label: live/spoof (또는 pseudo)
    out = bandvae.forward(clip)          # x̂_b, KLs 등
    phi = make_features(out, clip)       # 위에서 정의한 φ
    X.append(phi); y.append(label)

# 2-1) 원클래스 점수 임계치
tau = pick_threshold(X_live_only_features, criterion="min_ACER")

# 2-2) 소형 분류기
clf = MLP(phi_dim, hidden=[64,32], dropout=0.2)
clf.fit(X, y, loss="BCE", class_weight="balanced")
calibrator = platt_scaling(clf, X_val, y_val)

# Inference
phi_t = make_features(bandvae(clip_t), clip_t)
s_cls = calibrator(clf(phi_t))
s_ano = anomaly_score(phi_t)             # 가중합 of E_*, ΔSpec, KL
S = w*s_cls + (1-w)*norm(s_ano)
pred = (S >= τ_final)

```

# 빠른 실험 체크리스트

- Ablation
    1. 원클래스 S_ano 단독 vs S_cls 단독 vs 결합
    2. φ 구성(오직 재구성 vs +KL vs +ΔSpec)
    3. 분류기(로지스틱/MLP/XGBoost)
    4. HF-중심 특징만 vs 전 대역
- 보고: **APCER/BPCER/ACER/EER**, ROC-AUC/PR-AUC, 도메인별 성능 분해
- 캘리브레이션: TPR@FPR=10⁻²/10⁻³ (운영점)

---

- Stage 1에서 **안정적인 “대역별 자기복원 월등함”**을 학습해 두고,
- Stage 2에서 그 신호들을 **작고 강한 분류기**로 눌러 **결정경계**를 세워 주세요.

    초기에는 **분류기 얕게 + VAE freeze**로 시작하고, 필요할 때 **BP/HF 헤드만 소폭 미세튜닝**하면 리스크를 낮추면서 성능을 끌어올릴 수 있습니다.


# 뒷단 classifier 인풋

## 코어(권장 최소) φ 구성

- **대역별 재구성 오차 통계** (각 b∈{LF,BP,HF}):
    - `E_b_mean, E_b_p90, E_b_max` ← `E_b(t)=‖x_b(t)−x̂_b(t)‖₁`의 시간통계
- **대역 간 비율/대비**
    - `E_HF_ratio = E_HF_mean / (E_LF_mean+ε)`
    - `E_BP_ratio = E_BP_mean / (E_LF_mean+ε)`
- **KL 요약**
    - `KL_LF_sum`, `KL_BP_mean`, `KL_HF_mean` (프레임 KL의 합/평균)
- **합성 일관성/스펙트럼**
    - `ΔSpec = ‖STFT(x) − STFT(x̂)‖₁ / T`
- **대역 에너지 분포**
    - `R_HF = ‖x_HF‖₂ / ‖x‖₂`, `R_BP = ‖x_BP‖₂ / ‖x‖₂`

→ 이 최소 세트만으로도 15~25차원 내외의 깔끔한 φ가 됩니다.

## 확장(있으면 좋은) φ

- **프레임 동특성**: jerk/가속도 분산, 스파이크 개수(대역별)
- **동기성/불안정성**: `Δsync`, `Δsync_rel` (원본 vs 복원)
- **잠재 통계**:
    - `z_LF` 저차 요약(PCA 4~8D),
    - `z_t^HF`의 mean/var, 또는 프레임 간 코사인 유사도 평균
- **정직교 관련**: `corr(x̂_LF, x̂_BP)`, `corr(x̂_LF, x̂_HF)`

## 인풋 형태(예시)

- 예: 코어 18D + 확장 10D ≈ **28차원 벡터 φ ∈ ℝ^{28}**
- 배치마다 “클립 1개 → φ 1개”를 만들어 **소형 MLP/로지스틱/XGBoost**에 입력.
