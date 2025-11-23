"""
Fake 데이터 전처리 파이프라인
- Test set과 같은 크기 (190 segments) 생성
- 5가지 deepfake 기법에서 균등 샘플링
- preprocess_data_fast.py와 동일한 로직 사용
"""

import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp_proc
from functools import partial
import random

# MediaPipe Face Mesh 초기화는 각 프로세스에서 수행
mp_face_mesh = mp.solutions.face_mesh

# 얼굴 랜드마크 인덱스
LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]
LIPS_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
]
LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
]
FACE_CENTER = [1]

SELECTED_IDX = LEFT_EYE + RIGHT_EYE + LIPS_OUTER + LIPS_INNER + FACE_CENTER


def save_segment(landmarks, video_name, segment_idx, output_dir, video_shape):
    """세그먼트를 .npz로 저장 (Live 데이터와 동일한 포맷)"""
    landmarks_array = np.array(landmarks)  # (T, 75, 3)

    # Live 데이터와 같은 포맷으로 분리
    # SELECTED_IDX = LEFT_EYE(16) + RIGHT_EYE(16) + LIPS_OUTER(21) + LIPS_INNER(21) + FACE_CENTER(1) = 75
    left_eye = landmarks_array[:, 0:16, :2]      # (T, 16, 2) - x, y only
    right_eye = landmarks_array[:, 16:32, :2]    # (T, 16, 2)
    lips_outer = landmarks_array[:, 32:53, :2]   # (T, 21, 2)
    lips_inner = landmarks_array[:, 53:74, :2]   # (T, 21, 2)
    face_center = landmarks_array[:, 74:75, :2]  # (T, 1, 2)

    # 파일명
    filename = f"{video_name}_seg{segment_idx:03d}.npz"
    filepath = output_dir / filename

    # Live 데이터와 동일한 키로 저장 (2D coordinates)
    np.savez_compressed(
        filepath,
        left_eye=left_eye,
        right_eye=right_eye,
        lips_outer=lips_outer,
        lips_inner=lips_inner,
        face_center=face_center,
        size=np.array(video_shape)
    )


def process_single_video(args):
    """
    단일 비디오 처리 (멀티프로세싱용)

    Args:
        args: (video_path, output_dir, segment_duration, frame_skip)
    """
    video_path, output_dir, segment_duration, frame_skip = args

    # 각 프로세스마다 MediaPipe 초기화
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return f"✗ Cannot open: {video_path}"

        # 비디오 메타데이터
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frames_per_segment = int(segment_duration * original_fps)

        segment_idx = 0
        frame_count = 0
        segment_landmarks = []
        saved_count = 0

        video_name = Path(video_path).stem
        output_path = Path(output_dir)

        while True:
            ret, frame = cap.read()

            if not ret:
                # 마지막 세그먼트 처리
                if len(segment_landmarks) > 0:
                    save_segment(
                        segment_landmarks, video_name, segment_idx,
                        output_path, (height, width)
                    )
                    saved_count += 1
                break

            # 프레임 스킵 (속도 향상)
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe 처리
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                # 랜드마크 추출
                face_landmarks = results.multi_face_landmarks[0]
                lm_list = []

                for idx in SELECTED_IDX:
                    lm = face_landmarks.landmark[idx]
                    lm_list.append([lm.x, lm.y, lm.z])

                segment_landmarks.append(lm_list)

                # 세그먼트가 완성되면 저장
                if len(segment_landmarks) >= frames_per_segment:
                    save_segment(
                        segment_landmarks, video_name, segment_idx,
                        output_path, (height, width)
                    )
                    saved_count += 1
                    segment_idx += 1
                    segment_landmarks = []

            frame_count += 1

        cap.release()
        face_mesh.close()

        return f"✓ {video_name}: {saved_count} segments"

    except Exception as e:
        return f"✗ {video_path}: {str(e)}"


def main():
    # 설정
    fake_data_root = "/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_변조"
    output_dir = "/home/elicer/liveness_detection/model1/processed_fake_test"

    target_segments = 190  # Test set과 동일
    segment_duration = 5  # 5초
    frame_skip = 1  # 모든 프레임 사용
    num_workers = 7

    # 5가지 deepfake 기법
    fake_methods = ['audio_driven1', 'dffs1', 'dfl1', 'fo1', 'fsgan1']
    segments_per_method = target_segments // len(fake_methods)  # 38 each

    print("="*80)
    print("Fake Data Preprocessing for Hyperparameter Tuning")
    print("="*80)
    print(f"Target segments: {target_segments}")
    print(f"Methods: {len(fake_methods)}")
    print(f"Segments per method: {segments_per_method}")
    print(f"Segment duration: {segment_duration}s")
    print(f"Workers: {num_workers}")
    print("="*80)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 각 기법별로 비디오 수집 및 샘플링
    all_videos_to_process = []

    for method in fake_methods:
        method_dir = os.path.join(fake_data_root, method)

        if not os.path.exists(method_dir):
            print(f"Warning: {method_dir} does not exist, skipping...")
            continue

        # 모든 비디오 파일 찾기
        video_files = []
        for root, dirs, files in os.walk(method_dir):
            for f in files:
                if f.endswith('.mp4'):
                    video_files.append(os.path.join(root, f))

        print(f"\n{method}: Found {len(video_files)} videos")

        # 랜덤 샘플링 (각 비디오에서 약 5-6개 세그먼트 나오므로)
        # 38 segments 필요 → 약 7-8개 비디오 필요
        videos_needed = min(10, len(video_files))  # 여유있게 10개 선택
        sampled_videos = random.sample(video_files, videos_needed)

        print(f"{method}: Sampled {len(sampled_videos)} videos")

        # 처리 대상 추가
        for video_path in sampled_videos:
            all_videos_to_process.append(
                (video_path, output_dir, segment_duration, frame_skip)
            )

    print(f"\nTotal videos to process: {len(all_videos_to_process)}")
    print("\nStarting preprocessing...")

    # 병렬 처리
    with mp_proc.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, all_videos_to_process),
            total=len(all_videos_to_process),
            desc="Processing videos"
        ))

    # 결과 출력
    print("\n" + "="*80)
    print("Processing Results:")
    print("="*80)
    success_count = sum(1 for r in results if r.startswith("✓"))
    fail_count = sum(1 for r in results if r.startswith("✗"))

    print(f"Success: {success_count}/{len(results)}")
    print(f"Failed: {fail_count}/{len(results)}")

    # 실제 생성된 세그먼트 수 확인
    output_path = Path(output_dir)
    total_segments = len(list(output_path.glob("*.npz")))

    print(f"\nTotal segments created: {total_segments}")
    print(f"Target segments: {target_segments}")

    # 각 기법별 세그먼트 수 확인
    print("\nSegments per method:")
    for method in fake_methods:
        method_segments = len(list(output_path.glob(f"{method}*.npz")))
        print(f"  {method}: {method_segments}")

    # 목표보다 많으면 랜덤하게 제거
    if total_segments > target_segments:
        print(f"\nToo many segments ({total_segments}), removing excess...")
        all_segment_files = list(output_path.glob("*.npz"))
        random.shuffle(all_segment_files)

        # 초과분 제거
        for seg_file in all_segment_files[target_segments:]:
            seg_file.unlink()

        print(f"Removed {total_segments - target_segments} segments")
        print(f"Final count: {len(list(output_path.glob('*.npz')))}")

    print("\n" + "="*80)
    print("Fake data preprocessing completed!")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    main()
