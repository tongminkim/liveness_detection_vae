#!/bin/bash
# 데이터 처리 진행 상황 확인 스크립트

echo "========================================"
echo "데이터 처리 진행 상황"
echo "========================================"
echo ""

# 프로세스 상태
if ps aux | grep "preprocess_data_fast.py" | grep -v grep > /dev/null; then
    echo "✅ 프로세스 실행 중 (PID: $(cat preprocess.pid 2>/dev/null || echo 'Unknown'))"
else
    echo "❌ 프로세스 중단됨"
fi
echo ""

# 생성된 파일 수
FILE_COUNT=$(ls /home/elicer/liveness_detection/model1/processed_live/*.npz 2>/dev/null | wc -l)
EXPECTED_SEGMENTS=$((3020 * 19))
PROGRESS=$(echo "scale=2; $FILE_COUNT / $EXPECTED_SEGMENTS * 100" | bc)

echo "생성된 세그먼트: $FILE_COUNT / ~$EXPECTED_SEGMENTS (약 ${PROGRESS}%)"
echo ""

# 최근 로그 (진행률)
echo "=== 최근 진행률 ==="
tail -100 preprocess_full.log | grep -E "Processing videos:.*[0-9]+%" | tail -1
echo ""

# 디스크 사용량
echo "=== 디스크 사용량 ==="
DISK_USAGE=$(du -sh /home/elicer/liveness_detection/model1/processed_live 2>/dev/null | cut -f1)
DISK_AVAIL=$(df -h /home/elicer/liveness_detection/model1 | tail -1 | awk '{print $4}')
echo "사용 중: $DISK_USAGE"
echo "남은 용량: $DISK_AVAIL"
echo ""

# 에러 확인
ERROR_COUNT=$(grep -c "^✗" preprocess_full.log 2>/dev/null || echo 0)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  에러 발생: $ERROR_COUNT 건"
    echo "최근 에러:"
    grep "^✗" preprocess_full.log | tail -3
else
    echo "✅ 에러 없음"
fi
echo ""

echo "========================================"
echo "실시간 로그: tail -f preprocess_full.log"
echo "프로세스 중단: kill \$(cat preprocess.pid)"
echo "========================================"
