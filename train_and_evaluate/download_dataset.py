#!/usr/bin/env python3
"""
OmniSQL 데이터셋 다운로드 스크립트
HuggingFace에서 OmniSQL 데이터셋을 다운로드합니다.
"""

import os
import sys
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import argparse

def download_omnisql_dataset(cache_dir="./data", force_redownload=False, use_auth_token=False):
    """
    OmniSQL 데이터셋을 HuggingFace에서 다운로드합니다.
    
    Args:
        cache_dir (str): 데이터셋을 저장할 디렉토리
        force_redownload (bool): 이미 다운로드된 데이터셋이 있어도 다시 다운로드할지 여부
        use_auth_token (bool): HuggingFace 인증 토큰 사용 여부
    """
    
    print("=" * 60)
    print("OmniSQL 데이터셋 다운로드 시작")
    print("=" * 60)
    
    # 데이터 디렉토리 생성
    os.makedirs(cache_dir, exist_ok=True)
    print(f"데이터 저장 디렉토리: {os.path.abspath(cache_dir)}")
    
    try:
        # 다운로드 모드 설정
        download_mode = "force_redownload" if force_redownload else "reuse_dataset_if_exists"
        
        print(f"다운로드 모드: {download_mode}")
        print("데이터셋 다운로드 중... (시간이 오래 걸릴 수 있습니다)")
        
        # HuggingFace에서 데이터셋 다운로드
        if use_auth_token:
            print("🔐 HuggingFace 인증 토큰을 사용합니다.")
            dataset = load_dataset(
                'seeklhy/OmniSQL-datasets', 
                cache_dir=cache_dir,
                download_mode=download_mode,
                use_auth_token=True
            )
        else:
            dataset = load_dataset(
                'seeklhy/OmniSQL-datasets', 
                cache_dir=cache_dir,
                download_mode=download_mode
            )
        
        print("\n✅ 데이터셋 다운로드 완료!")
        print("=" * 60)
        print("데이터셋 정보:")
        print("=" * 60)
        
        # 데이터셋 구조 출력
        for split_name, split_data in dataset.items():
            print(f"📁 {split_name}: {len(split_data):,} samples")
            if len(split_data) > 0:
                print(f"   📋 컬럼: {list(split_data[0].keys())}")
                
                # 첫 번째 샘플의 일부 정보 출력
                first_sample = split_data[0]
                for key, value in first_sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"   📝 {key}: {value[:100]}...")
                    else:
                        print(f"   📝 {key}: {value}")
                print()
        
        print("=" * 60)
        print("다운로드된 데이터셋 위치:")
        print(f"📂 {os.path.abspath(cache_dir)}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        print("\n해결 방법:")
        print("1. 인터넷 연결을 확인하세요")
        print("2. HuggingFace 계정 로그인이 필요할 수 있습니다")
        print("   - huggingface-cli login 명령어로 로그인하세요")
        print("   - 또는 --use-auth-token 옵션을 사용하세요")
        print("3. 디스크 공간이 충분한지 확인하세요")
        print("4. --force-redownload 옵션으로 다시 시도해보세요")
        return False

def download_alternative():
    """
    대안 다운로드 방법 (ModelScope 사용)
    """
    print("\n" + "=" * 60)
    print("대안 다운로드 방법")
    print("=" * 60)
    print("ModelScope에서 직접 다운로드:")
    print("1. https://modelscope.cn/datasets/seeklhy/OmniSQL-datasets/summary 방문")
    print("2. 'Clone Dataset' 또는 'Download' 버튼 클릭")
    print("3. 다운로드된 파일을 ./data/ 디렉토리에 압축 해제")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='OmniSQL 데이터셋 다운로드')
    parser.add_argument('--cache-dir', default='./data', 
                       help='데이터셋을 저장할 디렉토리 (기본값: ./data)')
    parser.add_argument('--force-redownload', action='store_true',
                       help='이미 다운로드된 데이터셋이 있어도 다시 다운로드')
    parser.add_argument('--show-alternative', action='store_true',
                       help='대안 다운로드 방법 표시')
    parser.add_argument('--use-auth-token', action='store_true',
                       help='HuggingFace 인증 토큰 사용 (로그인이 필요한 경우)')
    
    args = parser.parse_args()
    
    if args.show_alternative:
        download_alternative()
        return
    
    success = download_omnisql_dataset(
        cache_dir=args.cache_dir,
        force_redownload=args.force_redownload,
        use_auth_token=args.use_auth_token
    )
    
    if not success:
        print("\n다운로드에 실패했습니다. 대안 방법을 확인하세요:")
        download_alternative()
        sys.exit(1)

if __name__ == "__main__":
    main()
