import json
import random
from pathlib import Path

# JSON 파일 로드
json_path = "/home/elicer/liveness_detection/model1/data_split.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Random seed 설정 (재현성을 위해)
random.seed(data.get('random_seed', 42))

# Final test 섹션에서 audio-driven fake만 필터링
final_test = data.get('final_test', {})
all_fake = final_test.get('fake', [])
all_real = final_test.get('real', [])

# audio-driven fake만 선택
audio_driven_fake = [f for f in all_fake if '/audio-driven/' in f]

print(f"Total fake videos in final_test: {len(all_fake)}")
print(f"Audio-driven fake videos: {len(audio_driven_fake)}")
print(f"Total real videos in final_test: {len(all_real)}")

# Real 영상을 fake와 같은 개수로 랜덤 샘플링
num_fake = len(audio_driven_fake)
if len(all_real) >= num_fake:
    balanced_real = random.sample(all_real, num_fake)
else:
    print(f"Warning: Not enough real videos ({len(all_real)}) to match fake count ({num_fake})")
    balanced_real = all_real

print(f"\nBalanced counts:")
print(f"Real: {len(balanced_real)}")
print(f"Fake (audio-driven): {len(audio_driven_fake)}")

# 업데이트된 final_test 생성
data['final_test'] = {
    'real': balanced_real,
    'fake': audio_driven_fake
}

# 백업 생성
backup_path = json_path + '.backup'
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"\nBackup saved to: {backup_path}")

# 원본 파일 업데이트
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"Updated: {json_path}")
