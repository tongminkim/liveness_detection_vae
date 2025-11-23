"""
데이터 전처리 파이프라인 (최적화 버전)
- 멀티프로세싱으로 병렬 처리
- 프레임 샘플링으로 속도 향상
- 비디오에서 15초 세그먼트 추출
- MediaPipe로 얼굴 랜드마크 추출
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

# MediaPipe Face Mesh 초기화는 각 프로세스에서 수행
mp_face_mesh = mp.solutions.face_mesh

# 얼굴 랜드마크 인덱스 (사용자 제공 코드와 동일)
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

# 전체 선택된 랜드마크 인덱스
SELECTED_IDX = LEFT_EYE + RIGHT_EYE + LIPS_OUTER + LIPS_INNER + FACE_CENTER


def process_single_video(args):
    """
    단일 비디오 처리 (멀티프로세싱용)

    Args:
        args: (video_path, output_dir, segment_duration, frame_skip)
    """
    video_path, output_dir, segment_duration, frame_skip = args

    # 각 프로세스마다 MediaPipe 초기화 (GPU 자동 사용)
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

            # 랜드마크 추출
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # 선택된 랜드마크 추출
                landmarks = np.array([
                    [face_landmarks.landmark[idx].x * width,
                     face_landmarks.landmark[idx].y * height]
                    for idx in SELECTED_IDX
                ], dtype=np.float32)

                segment_landmarks.append(landmarks)
            else:
                # 얼굴 미감지시 이전 프레임 복사
                if len(segment_landmarks) > 0:
                    segment_landmarks.append(segment_landmarks[-1].copy())
                else:
                    segment_landmarks.append(
                        np.zeros((len(SELECTED_IDX), 2), dtype=np.float32)
                    )

            frame_count += 1

            # 세그먼트 완료 체크 (프레임 스킵 고려)
            if frame_count >= (segment_idx + 1) * frames_per_segment:
                if len(segment_landmarks) >= 30:  # 최소 프레임 수
                    save_segment(
                        segment_landmarks, video_name, segment_idx,
                        output_path, (height, width)
                    )
                    saved_count += 1
                segment_idx += 1
                segment_landmarks = []

        cap.release()
        face_mesh.close()

        return f"✓ {video_name}: {saved_count} segments"

    except Exception as e:
        return f"✗ Error {video_path}: {e}"


def save_segment(landmarks_list, video_name, segment_idx, output_path, size):
    """세그먼트를 .npz 파일로 저장"""
    landmarks_array = np.array(landmarks_list)  # [T, K, 2]

    # 각 영역별로 분리
    idx = 0

    def slice_region(cnt):
        nonlocal idx
        sl = landmarks_array[:, idx:idx + cnt, :]
        idx += cnt
        return sl

    left_eye = slice_region(len(LEFT_EYE))
    right_eye = slice_region(len(RIGHT_EYE))
    lips_outer = slice_region(len(LIPS_OUTER))
    lips_inner = slice_region(len(LIPS_INNER))
    face_center = slice_region(len(FACE_CENTER))

    # 저장
    h, w = size
    save_path = output_path / f"{video_name}_seg{segment_idx:03d}.npz"
    np.savez_compressed(
        save_path,
        left_eye=left_eye,
        right_eye=right_eye,
        lips_outer=lips_outer,
        lips_inner=lips_inner,
        face_center=face_center,
        size=np.array([h, w], dtype=np.float32)
    )


def process_dataset_parallel(
    data_dir,
    output_dir,
    segment_duration=15,
    frame_skip=2,  # 매 2프레임마다 1개만 처리 (30fps → 15fps)
    num_workers=6,  # 병렬 프로세스 수
    max_videos=None
):
    """
    데이터셋을 병렬로 처리

    Args:
        data_dir: 원본 비디오 디렉토리
        output_dir: 출력 디렉토리
        segment_duration: 세그먼트 길이 (초)
        frame_skip: 프레임 스킵 간격 (2 = 매 2프레임마다 1개)
        num_workers: 병렬 프로세스 수
        max_videos: 처리할 최대 비디오 수
    """
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 비디오 파일 목록
    video_files = sorted(glob.glob(os.path.join(data_dir, "**/*.mp4"), recursive=True))

    if max_videos:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} videos")
    print(f"Output: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Frame skip: {frame_skip} (effective FPS: ~{30/frame_skip:.1f})")
    print()

    # 이미 처리된 파일 확인
    existing = set()
    if os.path.exists(output_dir):
        existing_files = glob.glob(os.path.join(output_dir, "*.npz"))
        for f in existing_files:
            # 파일명에서 비디오 이름 추출 (seg 제외)
            basename = Path(f).stem
            video_name = '_'.join(basename.split('_')[:-1])  # _seg000 제거
            existing.add(video_name)

    # 아직 처리되지 않은 비디오만 필터링
    to_process = []
    for vf in video_files:
        video_name = Path(vf).stem
        if video_name not in existing:
            to_process.append(vf)

    if len(to_process) < len(video_files):
        print(f"Skipping {len(video_files) - len(to_process)} already processed videos")
        print(f"Processing {len(to_process)} remaining videos\n")

    if len(to_process) == 0:
        print("All videos already processed!")
        return

    # 인자 준비
    args_list = [
        (vf, output_dir, segment_duration, frame_skip)
        for vf in to_process
    ]

    # 멀티프로세싱으로 처리
    with mp_proc.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, args_list),
            total=len(args_list),
            desc="Processing videos"
        ))

    # 결과 출력
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)

    success = sum(1 for r in results if r.startswith("✓"))
    failed = sum(1 for r in results if r.startswith("✗"))

    print(f"Success: {success}")
    print(f"Failed: {failed}")

    # 에러 출력
    if failed > 0:
        print("\nErrors:")
        for r in results:
            if r.startswith("✗"):
                print(f"  {r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast parallel landmark extraction")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/elicer/liveness_detection/003.딥페이크/1.Training/원천데이터/train_원본/원본1",
        help="Input video directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/elicer/liveness_detection/model1/processed_live",
        help="Output directory"
    )
    parser.add_argument(
        "--segment_duration",
        type=int,
        default=15,
        help="Segment duration in seconds"
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=2,
        help="Process every Nth frame (2 = half fps, faster)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Max videos to process (for testing)"
    )

    args = parser.parse_args()

    process_dataset_parallel(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration,
        frame_skip=args.frame_skip,
        num_workers=args.num_workers,
        max_videos=args.max_videos
    )
