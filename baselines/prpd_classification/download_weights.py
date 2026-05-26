"""
HuggingFace Hub에서 PRPD 분류기 사전학습 가중치를 다운로드하는 헬퍼 스크립트

가중치 출처: AI Hub 데이터셋 71682 공식 제공
호스팅: https://huggingface.co/spaces/zero-coke/PRPD_baseline_pt

사용법:
    pip install huggingface_hub
    python download_weights.py
"""

import os
import sys
from typing import List, Tuple

# 절대경로 하드코딩 (코드 규칙)
SCRIPT_DIR: str = "/Users/ddongsmac/PycharmProjects/ClaudeCode/PRPD_repo/baselines/prpd_classification"
CHECKPOINT_DIR: str = os.path.join(SCRIPT_DIR, "checkpoints")

# HuggingFace 저장소 정보
HF_REPO_ID: str = "zero-coke/PRPD_baseline_pt"
HF_REPO_TYPE: str = "space"  # spaces URL이므로 space로 지정 (model repo로 변경 시 "model"로)

# 다운로드 대상 파일 목록: (HF상의 파일명, 로컬 저장 하위경로)
WEIGHT_FILES: List[Tuple[str, str]] = [
    ("Resnet_2023_11_18_16_09_22_4.pkl",       "resnet152/Resnet_2023_11_18_16_09_22_4.pkl"),
    ("Efficientnet_2023_11_18_16_05_02_10.pkl", "efficientnet_b0/Efficientnet_2023_11_18_16_05_02_10.pkl"),
]


def download_weights() -> None:
    """
    HuggingFace Hub에서 가중치 파일을 다운로드하여 checkpoints/ 폴더에 배치한다.

    동작:
        1. huggingface_hub 라이브러리 import (없으면 안내)
        2. checkpoints/{resnet152, efficientnet_b0}/ 폴더 생성
        3. hf_hub_download로 각 파일 다운로드
        4. 다운로드된 파일을 지정 경로로 이동
    """
    # huggingface_hub 라이브러리 확인
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub 라이브러리가 설치되어 있지 않습니다.")
        print("설치: pip install huggingface_hub")
        sys.exit(1)

    # 가중치 저장 폴더 생성 (이미 있어도 OK)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"[INFO] HuggingFace 저장소: {HF_REPO_ID} ({HF_REPO_TYPE})")
    print(f"[INFO] 저장 경로: {CHECKPOINT_DIR}\n")

    for hf_filename, local_subpath in WEIGHT_FILES:
        local_full_path: str = os.path.join(CHECKPOINT_DIR, local_subpath)

        # 이미 다운로드된 경우 스킵
        if os.path.exists(local_full_path):
            print(f"[SKIP] 이미 존재: {local_subpath}")
            continue

        # 로컬 하위 폴더 생성
        os.makedirs(os.path.dirname(local_full_path), exist_ok=True)

        print(f"[DOWNLOAD] {hf_filename} → {local_subpath}")
        try:
            # HF에서 다운로드 (캐시 디렉토리에 받은 후 local_dir로 복사)
            downloaded_path: str = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=hf_filename,
                local_dir=os.path.dirname(local_full_path),
            )
            print(f"[OK]   {downloaded_path}\n")
        except Exception as e:
            # 다운로드 실패 시 원인 출력 후 다음 파일 시도
            print(f"[FAIL] {hf_filename} 다운로드 실패: {e}\n")
            continue

    print("[DONE] 가중치 다운로드 완료")
    print(f"      가중치 위치: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    download_weights()
