Pipeline
>
>
> 좋습니다. 아래는 **입술 랜드마크 기반 동기성 / Liveness 분석**을 위한 전처리 과정을, 논문 수준에서 **재현 가능(reproducible)** 하게 요약한 정리입니다👇
>
> ---
>
> ## 🎯 **목표**
>
> 라벨 JSON에서 문장 단위로 segment한 원본 영상에 대해
>
> 👉 해당 구간의 랜드마크(입술 + 정규화용 기준점) 시퀀스를 추출
>
> 👉 논문 표준 방식으로 정규화
>
> 👉 `.npy` 파일로 저장
>
> ---
>
> ## 📄 **1. 입력 데이터**
>
> - **Video**: 한 번에 여러 문장을 발화하는 원본 영상 (fps 고정)
> - **JSON**: 문장별 `start_time`, `end_time` (초 단위) 정보 포함
>
> ---
>
> ## 🧠 **2. 문장 단위로 Segment**
>
> ```python
> # 예시
> for utt in label_json['utterances']:
>     start_t = utt['start']
>     end_t = utt['end']
>     segment = video[start_t:end_t]  # 정확히는 fps 기준으로 frame index 변환
>
> ```
>
> - `start_idx = int(start_t * fps)`
> - `end_idx = int(end_t * fps)`
> - segment 단위로 frame list를 추출
>
> ---
>
> ## 📝 **3. 랜드마크 추출**
>
> - 얼굴 랜드마크 검출기(예: MediaPipe FaceMesh, OpenFace, Dlib 등)를 사용
> - 각 frame별로 다음 점 추출:
>     - 입술 영역(예: MediaPipe 기준 0~67 중 입술 관련 인덱스)
>     - 정규화를 위한 기준점:
>         - 얼굴 중심 `C_t` (예: 코 tip 또는 입 중심)
>         - 기준쌍 `E_L, E_R` (눈 코너 or 입 코너)
> - 실패 프레임은 보간 처리(선형 interpolation)
>
> ---
>
> ## 🧭 **4. 좌표 정규화 (Per Frame)**
>
> 논문 표준 권장식 (얼굴 정렬 + 크기 정규화 + z-score):
>
> [
>
> X_{t,k}^{\text{norm}} =
>
> \frac{ R(\theta_t)\bigl( X_{t,k} - C_t \bigr) }{ |E_{R,t} - E_{L,t}| }
>
> ]
>
> - `R(θ_t)` : 좌우 눈(or 입 코너) 기준으로 회전 정렬(roll 보정)
> - `‖ER - EL‖` : 얼굴 폭으로 스케일 정규화
> - `C_t` : 얼굴 중심 기준 translation 제거
>
> 👉 이렇게 하면 얼굴 크기·위치·회전에 불변한 상대 좌표계로 변환됨.
>
> ---
>
> ## 📊 **5. Z-score 정규화 (Global)**
>
> - 모든 훈련 데이터에서 `mean`/`std`를 계산하고
> - 각 좌표에 대해 z-score로 정규화:
>
>     [
>
>     $$
>     Z = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}
>     $$
>
>     ]
>
> - 테스트/검증 시에는 훈련 집계값을 그대로 사용 → 재현성 확보
>
> ---
>
> ## 💾 **6. 저장 (per utterance)**
>
> ```python
> np.save(f'{out_dir}/{video_id}_{utt_id}.npy', normalized_landmarks)
>
> ```
>
> - shape: `(T, K, 2)` (T=프레임 수, K=랜드마크 개수, 2=x,y)
> - 필요 시 `npz`로 metadata(utt text 등) 함께 저장
>
> ---
>
 ## 📌 **재현성을 위한 팁**

 - 모든 랜드마크 검출기 설정(fps, confidence threshold, smoothing) 고정
 - train/test split 후 `mean/std`는 train만에서 산출
 - segment 단위로 랜드마크 추출 시 frame index rounding을 항상 같은 방식 사용 (floor vs round 통일)
 - 좌표계 변환(R, C_t) 방식은 함수화해서 deterministic하게 유지

---

이 방식은 입술 움직임 분석 논문들 (e.g., SyncNet, Wav2Lip, LipForensics)에서도 거의 공통적으로 쓰이는 pipeline입니다.

특히 정규화 식과 z-score 단계는 모델이 *사람의 얼굴 위치나 크기에 영향을 받지 않고* **순수하게 움직임 패턴**에 집중할 수 있게 합니다.
