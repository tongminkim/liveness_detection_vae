"""
데이터 전처리 파이프라인
- 비디오에서 15초 세그먼트 추출 (실시간 처리, 저장 안 함)
- MediaPipe로 입술 랜드마크 추출
- Procrustes 정규화 적용
- Feature engineering (velocity, acceleration, angle, angle-rate)
"""

import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# MediaPipe Face Mesh 초기화
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


class VideoLandmarkExtractor:
    """비디오에서 얼굴 랜드마크를 추출하는 클래스 (눈 + 입술 + 얼굴 중심)"""

    def __init__(self, segment_duration=15, fps=30, max_detection_faces=1):
        """
        Args:
            segment_duration: 세그먼트 길이 (초)
            fps: 타겟 FPS (원본 비디오의 FPS와 다를 수 있음)
            max_detection_faces: 감지할 최대 얼굴 수
        """
        self.segment_duration = segment_duration
        self.target_fps = fps
        self.max_detection_faces = max_detection_faces

        # MediaPipe Face Mesh 초기화
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_detection_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_video(self, video_path):
        """
        비디오에서 15초 세그먼트별로 랜드마크 추출

        Args:
            video_path: 비디오 파일 경로

        Yields:
            dict: {
                'landmarks': np.array [T, K, 2],  # K=40 (lips)
                'segment_idx': int,
                'fps': float,
                'size': (height, width)
            }
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # 비디오 메타데이터
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frames_per_segment = int(self.segment_duration * original_fps)

        segment_idx = 0
        frame_count = 0
        segment_landmarks = []

        while True:
            ret, frame = cap.read()

            if not ret:
                # 마지막 세그먼트 처리
                if len(segment_landmarks) > 0:
                    yield self._process_segment(
                        segment_landmarks, segment_idx, original_fps, (height, width)
                    )
                break

            # RGB 변환 (MediaPipe는 RGB 입력 필요)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 랜드마크 추출
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                # 첫 번째 얼굴만 사용
                face_landmarks = results.multi_face_landmarks[0]

                # 선택된 랜드마크 추출 (눈 + 입술 + 얼굴중심)
                landmarks = np.array([
                    [face_landmarks.landmark[idx].x * width,
                     face_landmarks.landmark[idx].y * height]
                    for idx in SELECTED_IDX
                ], dtype=np.float32)  # [총 K, 2]

                segment_landmarks.append(landmarks)
            else:
                # 얼굴이 감지되지 않으면 이전 프레임 복사 (또는 제로)
                if len(segment_landmarks) > 0:
                    segment_landmarks.append(segment_landmarks[-1].copy())
                else:
                    # 첫 프레임인데 감지 안 되면 제로
                    segment_landmarks.append(np.zeros((len(SELECTED_IDX), 2), dtype=np.float32))

            frame_count += 1

            # 세그먼트 완료
            if frame_count % frames_per_segment == 0:
                yield self._process_segment(
                    segment_landmarks, segment_idx, original_fps, (height, width)
                )
                segment_idx += 1
                segment_landmarks = []

        cap.release()

    def _process_segment(self, landmarks_list, segment_idx, fps, size):
        """세그먼트 랜드마크를 numpy array로 변환"""
        landmarks_array = np.array(landmarks_list)  # [T, K, 2]

        return {
            'landmarks': landmarks_array,
            'segment_idx': segment_idx,
            'fps': fps,
            'size': size  # (height, width)
        }

    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# ============================================
# Procrustes (Umeyama) Normalization
# ============================================

def umeyama_similarity(X, Y):
    """
    Umeyama similarity transform
    X, Y: [N, 2] (float) → find s, R, t s.t. Y ≈ s R X + t
    Returns: s (scale), R (2x2 rotation), t (2, translation)
    """
    Xc = X.mean(axis=0)
    Yc = Y.mean(axis=0)
    X0 = X - Xc
    Y0 = Y - Yc

    var = (X0**2).sum() / X0.shape[0] + 1e-12
    U, S, Vt = np.linalg.svd((Y0.T @ X0) / X0.shape[0])
    R = U @ Vt

    # Reflection 방지
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = S.sum() / var
    t = Yc - s * (R @ Xc)

    return s, R, t


def apply_similarity(P, s, R, t):
    """Apply similarity transform to points P [M, 2]"""
    return (s * (P @ R.T)) + t


def normalize_landmarks_procrustes(landmarks, ref_frames=10):
    """
    Procrustes 정규화 적용

    Args:
        landmarks: [T, K, 2] numpy array
        ref_frames: reference shape을 계산할 프레임 수

    Returns:
        normalized_landmarks: [T, K, 2] numpy array
    """
    T, K, _ = landmarks.shape

    # Reference shape: 처음 ref_frames의 평균
    ref_shape = landmarks[:min(ref_frames, T)].mean(axis=0)  # [K, 2]

    # 각 프레임별로 정규화
    normalized = np.empty_like(landmarks)
    for t in range(T):
        try:
            s, R, tt = umeyama_similarity(landmarks[t], ref_shape)
            normalized[t] = apply_similarity(landmarks[t], s, R, tt)
        except np.linalg.LinAlgError:
            # Singular matrix 등의 에러 발생 시 원본 사용
            normalized[t] = landmarks[t]

    return normalized


# ============================================
# Feature Engineering
# ============================================

def central_diff(arr):
    """
    Central difference (length-preserving)
    arr: [T, K, D] -> output: [T, K, D]
    """
    d = np.empty_like(arr)
    d[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    d[0] = arr[1] - arr[0]
    d[-1] = arr[-1] - arr[-2]
    return d


def second_diff(arr):
    """
    Second-order difference (length-preserving)
    arr: [T, K, D] -> output: [T, K, D]
    """
    dd = np.empty_like(arr)
    dd[1:-1] = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    dd[0] = arr[1] - arr[0]
    dd[-1] = arr[-1] - arr[-2]
    return dd


def compute_features(landmarks, fps=30):
    """
    입술 랜드마크에서 모든 특징 추출

    Args:
        landmarks: [T, K, 2] normalized landmarks
        fps: frames per second

    Returns:
        features: dict with keys:
            - 'position': [T, K, 2]
            - 'velocity': [T, K, 2]
            - 'acceleration': [T, K, 2]
            - 'angle': [T, K, 1]
            - 'angle_rate': [T, K, 1]
    """
    # 1. Position (already normalized)
    position = landmarks  # [T, K, 2]

    # 2. Velocity (1st derivative)
    velocity = central_diff(landmarks) * fps  # [T, K, 2]

    # 3. Acceleration (2nd derivative)
    acceleration = second_diff(landmarks) * (fps ** 2)  # [T, K, 2]

    # 4. Angle/Slope
    # 각 랜드마크에서 다음 랜드마크로의 방향 벡터
    v = np.roll(landmarks, -1, axis=1) - landmarks  # [T, K, 2]
    angles = np.arctan2(v[..., 1], v[..., 0])  # [T, K]

    # Temporal unwrapping (시간축 방향)
    angles_unwrap = np.unwrap(angles, axis=0)[..., None]  # [T, K, 1]

    # 5. Angle-rate (angle의 시간 미분)
    angle_rate = central_diff(angles_unwrap) * fps  # [T, K, 1]

    return {
        'position': position,
        'velocity': velocity,
        'acceleration': acceleration,
        'angle': angles_unwrap,
        'angle_rate': angle_rate
    }


def features_to_array(features):
    """
    Feature dict를 single array로 변환

    Returns:
        feature_array: [T, K, F] where F = 2+2+2+1+1 = 8
    """
    return np.concatenate([
        features['position'],        # [T, K, 2]
        features['velocity'],        # [T, K, 2]
        features['acceleration'],    # [T, K, 2]
        features['angle'],           # [T, K, 1]
        features['angle_rate']       # [T, K, 1]
    ], axis=2)  # [T, K, 8]


# ============================================
# Main Processing Function
# ============================================

def process_video(video_path, output_dir, segment_duration=15, fps=30):
    """
    비디오 전체를 처리하고 .npz 파일로 저장

    Args:
        video_path: 입력 비디오 경로
        output_dir: 출력 디렉토리
        segment_duration: 세그먼트 길이 (초)
        fps: 타겟 FPS
    """
    extractor = VideoLandmarkExtractor(segment_duration=segment_duration, fps=fps)

    video_name = Path(video_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for segment_data in extractor.extract_from_video(video_path):
        landmarks = segment_data['landmarks']  # [T, K, 2] where K=len(SELECTED_IDX)
        segment_idx = segment_data['segment_idx']
        seg_fps = segment_data['fps']
        size = segment_data['size']

        # 프레임 수가 너무 적으면 스킵
        if landmarks.shape[0] < 30:  # 최소 1초 이상
            continue

        # 1. 각 영역별로 랜드마크 분리 (사용자 코드 방식과 동일)
        idx = 0

        def slice_region(cnt):
            nonlocal idx
            sl = landmarks[:, idx:idx + cnt, :]
            idx += cnt
            return sl

        left_eye = slice_region(len(LEFT_EYE))      # [T, 16, 2]
        right_eye = slice_region(len(RIGHT_EYE))    # [T, 16, 2]
        lips_outer = slice_region(len(LIPS_OUTER))  # [T, 21, 2]
        lips_inner = slice_region(len(LIPS_INNER))  # [T, 21, 2]
        face_center = slice_region(len(FACE_CENTER)) # [T, 1, 2]

        # 2. 저장 (사용자 코드 방식과 동일한 구조)
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

        print(f"✓ Saved: {save_path} | left_eye: {left_eye.shape}, lips_outer: {lips_outer.shape}")


def process_dataset(data_dir, output_dir, segment_duration=15, fps=30, max_videos=None):
    """
    데이터셋 전체를 처리

    Args:
        data_dir: 원본 비디오 디렉토리
        output_dir: 출력 디렉토리
        segment_duration: 세그먼트 길이 (초)
        fps: 타겟 FPS
        max_videos: 처리할 최대 비디오 수 (테스트용)
    """
    # 모든 MP4 파일 찾기
    video_files = sorted(glob.glob(os.path.join(data_dir, "**/*.mp4"), recursive=True))

    if max_videos:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} videos in {data_dir}")
    print(f"Output directory: {output_dir}\n")

    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            process_video(video_path, output_dir, segment_duration, fps)
        except Exception as e:
            print(f"\n✗ Error processing {video_path}: {e}")
            continue


# ============================================
# CLI Interface
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract lip landmarks from videos")
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
        help="Output directory for processed .npz files"
    )
    parser.add_argument(
        "--segment_duration",
        type=int,
        default=15,
        help="Segment duration in seconds"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (for testing)"
    )

    args = parser.parse_args()

    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration,
        fps=args.fps,
        max_videos=args.max_videos
    )
