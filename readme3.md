# 1) 모델 구조

- 경량, 원클래스, timeseries 고려

## 입력·전처리

- **기본**: 입술 **랜드마크 시퀀스** $x∈ \mathbb{R}^{T\times K\times 2}$

    예:  T=48∼64, K=20∼30 (lip부분 랜드마크 keypoints), 25–30fps, xy 좌표로 2차원

    (스무딩: EMA/칼만 살짝, 과도 스무딩 금지)


## 백본 인코더(공유) → LF/HF 두 개의 VAE 헤드

- **시간 인코더(랜드마크용, TCN 권장)**
    - 1D TCN 3–4층, dilation [1,2,4,8], 채널 64→128
    - 출력: 프레임별 임베딩 $h_t\in\mathbb{R}^{128}$
- **LF-VAE 헤드(clip-level 잠재)**
    - **Temporal pooling**(mean+attention) → $g\in\mathbb{R}^{128}$
    - 선형층 → $\mu_{LF}, \log\sigma_{LF} \in \mathbb{R}^{32}$ (**작은 차원**)
    - **LF-Decoder**(작은 TCN or GRU)로 **저역(느린 궤적)** 복원 $\hat{x}_{LF}$
- **HF-VAE 헤드(frame-level 잠재)**
    - 각 $h_t$ → 선형층 → $\mu_t,\log\sigma_t \in \mathbb{R}^{64}$ (**프레임별 잠재**)
    - **HF-Decoder**(작은 TCN)로 **고역 잔차/세부** 복원 $\hat{r}_{HF}$
- **합성 출력(해석 쉬움)**
    - 최종 복원: $\hat{x}=\hat{x}_{LF}+\hat{r}_{HF}$
    - 점수 계산은 $\hat{x}_{LF}$와 $\hat{r}_{HF}$의 **오차를 분리**해서 사용(E_smooth, E_detail)
- 파라미터 수: 랜드마크 베이스 약 1.2–1.8M → 코랩 무료버전도 가

# 2) 로스 함수 설계 (저·고주파 **역할 분리** + **시간 동역학** + **분해 해석성**)

## 공통: VAE 기본

- 재파라미터화 $z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$
- KL 항: $\mathrm{KL}\big(q(z|x)\,\|\,\mathcal{N}(0,I)\big)$

## (A) LF(VAE) — **저주파만** 잘 그리게

1. **저역 타깃 재구성 손실**
    - 입력을 **LPF(저역통과)** 해서 타깃 $\tilde{x}_{LP}$생성
    - $\mathcal{L}_{rec}^{LF}=\| \hat{x}_{LF}-\tilde{x}_{LP}\|_2^2$
2. **강한 β-KL (용량 축소)**
    - $\mathcal{L}_{KL}^{LF}=\beta_{LF}\,\mathrm{KL}(q(z_{LF}|x),\ \mathcal{N}(0,I)), β_{LF}≈4∼8$

        → 고주파 정보는 병목을 못 통과

3. **(선택) 시간 다운샘플/부드러움 유도**
    - 디코더 입력 시간축 stride=2 설정 또는 $\|\nabla_t \hat{x}_{LF}\|^2$ 가벼운 가중

결과: LF는 개구/폐구, 모음 전이, 턱의 느린 회전 등만 복원

## (B) HF(VAE) — **고주파/미세 동작**을 정확히

1. **잔차/고역 타깃 재구성**
    - 타깃$r_{HF}=x-\tilde{x}_{LP}$ 또는 **HPF(x)**
    - $\mathcal{L}_{rec}^{HF}=\| \hat{r}_{HF}-r_{HF}\|_2^2$
2. **STFT 가중 손실(고주파 강조)**
    - $\mathcal{L}_{stft}=\sum_f w(f)\,\|\mathrm{STFT}(\hat{r}_{HF})-\mathrm{STFT}(r_{HF})\|_2^2$

        여기서 **상위 1/3 대역 w(f)w(f)w(f)↑** (예 2–3배)

3. **시간 미분 손실(동역학 보존)**
    - $\mathcal{L}_{\nabla}=\lambda_1\|\nabla_t \hat{x}-\nabla_t x\|_1 + \lambda_2\|\nabla_t^2 \hat{x}-\nabla_t^2 x\|_1 + \lambda_3\|\nabla_t^3 \hat{x}-\nabla_t^3 x\|_1$
    - $jerk(\nabla_t^3)$ 항은 **자음 경계/급변**에 민감
4. **약한 β-KL (정보 보존)**
    - $\mathcal{L}_{KL}^{HF}=\beta_{HF}\,\mathrm{KL}(\cdot), \beta_{HF}\approx 0.5\sim1.0$
5. **(선택) 마스크드 시간 인페인팅**
    - 랜덤 **연속 프레임 마스크** 후 복원 → **과적합/쇼트컷 방지**
    - 손실$\mathcal{L}_{mask}$(마스크 위치만 계산)

결과: HF는 립 코너 잔진동, 치아/혀 순간 하이라이트, 자음 폐쇄·폭발 등 미세 패턴을 복원

## (C) LF/HF **분리와 일관성**을 위한 보조 항

1. **합성 일치(전체 복원 체크)**
    - $\hat{x}=\hat{x}_{LF}+\hat{r}_{HF}$가 원본과 맞는지:
    - $\mathcal{L}_{full}=\|\hat{x}-x\|_1$ (작은 가중)

        → 해석 가능한 분해 유지

2. **잠재 상관 억제(분리 강화)**
    - LF 잠재를 시간으로 늘리고(↑) HF 잠재와 **상호상관 최소화**:
    - $\mathcal{L}_{decor}=\| \mathrm{Corr}(z_{LF}^{\uparrow}, z_{HF})\|_F^2$ 또는 HSIC/BarlowTwins식 크로스-코릴레이션 제약
3. **지터-일관성(실데이터 안정화)**
    - 실(라이브) 클립에 **±1프레임 시간 워핑/프레임 드롭 5–10%** 적용한 x′x'x′에 대해
    - **HF 복원 일관성**: $\mathcal{L}_{jitter}=\|\hat{r}_{HF}(x')-\hat{r}_{HF}(x)\|_1$ (소가중)

        → **실제 데이터에서는 ΔE가 작도록** 학습(스푸프는 학습X → ΔE가 커지기 쉬움)


## (D) 총손실 (권장 가중 예시)

$\begin{aligned}
\mathcal{L} =\ &
\underbrace{\mathcal{L}_{rec}^{LF}}_{1.0}
+ \underbrace{\mathcal{L}_{KL}^{LF}}_{\beta_{LF}=6}
+ \underbrace{\mathcal{L}_{rec}^{HF}}_{0.5}
+ \underbrace{\mathcal{L}_{stft}}_{0.5}
+ \underbrace{\mathcal{L}_{\nabla}}_{0.3}
+ \underbrace{\mathcal{L}_{KL}^{HF}}_{\beta_{HF}=0.8}\\
&+ \underbrace{\mathcal{L}_{full}}_{0.2}
+ \underbrace{\mathcal{L}_{decor}}_{0.1}
+ \underbrace{\mathcal{L}_{jitter}}_{0.1}
\end{aligned}$

- 계수는 개발셋에서 조정

---

# 3) 학습·추론 프로토콜

- **학습 데이터**: **실제(라이브)만** 사용(원-클래스)

    증강: 시간 지터(±1f), 프레임 드롭(5–10%), 속도 0.95–1.05×, 약한 공간 노이즈

- **스케줄**: AdamW(lr 1e-3), cosine decay, batch 16–32, **KL anneal**(0→target, 10–20 epoch)
- **추론(평가)**:
    - **$E_{smooth} =\|x-\hat{x}_{LF}\|, E_{detail} =\|r_{HF}-\hat{r}_{HF}\|$**
    - **$DRS E_{smooth}/(E_{detail}+\varepsilon)$**
    - **ΔE**: 미세 지터 입력 후 HF 오차 증가량
    - **C**: 동역학 복잡도(샘플 엔트로피/스펙트럴 플랫니스)
    - **최종 점수**: $S =\alpha\cdot DRS-\beta\cdot \Delta E+\gamma\cdot C$ → 임계값 τ로 Live/Spoof

# 참고논문

https://arxiv.org/abs/2401.09006

https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.pdf

# 데이터셋

https://www.kaggle.com/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet

https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html

https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=&topMenu=&srchOptnCnd=OPTNCND001&searchKeyword=%EB%A6%BD%EB%A6%AC%EB%94%A9&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=538
