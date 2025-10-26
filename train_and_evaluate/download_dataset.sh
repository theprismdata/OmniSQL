#!/bin/bash

# OmniSQL 데이터셋 다운로드 스크립트
# uv 가상환경을 사용하여 데이터셋을 다운로드합니다.

set -e  # 에러 발생 시 스크립트 중단

echo "=========================================="
echo "OmniSQL 데이터셋 다운로드 스크립트"
echo "=========================================="

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "작업 디렉토리: $SCRIPT_DIR"

# uv 가상환경 활성화
if [ ! -d "omnisql_env" ]; then
    echo "가상환경이 없습니다. 생성 중..."
    uv venv omnisql_env
fi

echo "가상환경 활성화 중..."
source omnisql_env/bin/activate

# 필요한 패키지 설치
echo "필요한 패키지 설치 중..."
uv pip install datasets huggingface_hub

# 데이터셋 다운로드 실행
echo "데이터셋 다운로드 시작..."
python3 download_dataset.py "$@"

echo "=========================================="
echo "다운로드 완료!"
echo "=========================================="
