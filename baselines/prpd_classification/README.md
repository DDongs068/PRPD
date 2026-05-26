# PRPD Classification Baseline

전력설비 **부분방전(Partial Discharge, PD)** 신호를 PRPD(Phase Resolved Partial Discharge) 이미지로 변환하여
딥러닝 분류기로 5개 클래스를 판별하는 **베이스라인 코드**입니다.

> **본 코드는 AI Hub에서 제공하는 데이터셋의 공식 베이스라인 모델 코드입니다. 자체 개발한 코드가 아니며, 학습/연구 목적의 참고용으로 본 레포에 보관·정리합니다.**

---

## 📌 출처 (Source)

- **데이터셋**: [AI Hub - 전력설비 부분방전 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71682)
- **공식 페이지**: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71682
- **제공처**: AI Hub (한국지능정보사회진흥원, NIA)
- **데이터셋 번호**: 71682
- **코드 출처**: 위 데이터셋 공식 제공 베이스라인 모델 소스코드 (`1.모델소스코드/PRPD_Classification/`)
- **사전학습 가중치 출처**: 위 데이터셋 공식 제공 AI 학습 모델 파일 (`2.AI학습모델파일/`)

> ⚠️ **저작권 안내**: 본 코드/가중치의 원 저작권은 AI Hub 및 데이터셋 구축 기관에 있습니다.
> 사용 시 반드시 AI Hub 이용약관 및 데이터셋 라이선스를 준수해 주세요.

---

## 🎯 분류 클래스 (5종)

| Class ID | 영문 | 한글 | 설명 |
|---|---|---|---|
| 0 | normal | 정상 | 부분방전이 없는 정상 신호 |
| 1 | noise | 노이즈 | 외부 잡음 (방전 X) |
| 2 | surface | 표면 방전 | 절연체 표면을 따라 발생하는 방전 |
| 3 | corona | 코로나 방전 | 고전압 도체 주변 공기 이온화 방전 |
| 4 | void | 보이드 방전 | 절연체 내부 공극에서 발생하는 방전 |

---

## 🏗️ 모델 구성

| 모델 | Backbone | Classifier Head | Batch | 비고 |
|---|---|---|---|---|
| **ResNet152** | torchvision pretrained | Dropout(0.5) → Linear(2048→1024) → ReLU → Linear(1024→5) | 64 | ImageNet 사전학습 |
| **EfficientNet-B0** | torchvision pretrained | Dropout(0.2) → Linear(1280→512) → ReLU → Linear(512→5) | 128 | ImageNet 사전학습 |

- **Optimizer**: Adam (lr=1e-4, β1=0.5, β2=0.999)
- **Loss**: CrossEntropyLoss
- **Scheduler**: 선형 LR decay (epoch > num_epochs - num_epochs_decay 시점부터)
- **Early Stopping**: validation loss 기준, patience=7
- **Augmentation**: 없음 (ToTensor + ImageNet Normalize + Resize(256,256))

---

## 📦 사전학습 가중치 다운로드 (HuggingFace)

가중치 파일은 GitHub 용량 제한(단일 파일 100MB)으로 인해 **HuggingFace Hub**에 별도 호스팅됩니다.

- **HF 저장소**: https://huggingface.co/spaces/zero-coke/PRPD_baseline_pt

### 방법 1: 자동 다운로드 스크립트 (권장)

```bash
pip install huggingface_hub
python download_weights.py
```

### 방법 2: 수동 다운로드

위 HuggingFace URL에서 직접 다운로드 후 아래 경로에 배치하세요:

```
baselines/prpd_classification/
└── checkpoints/
    ├── resnet152/
    │   └── Resnet_2023_11_18_16_09_22_4.pkl      (231 MB)
    └── efficientnet_b0/
        └── Efficientnet_2023_11_18_16_05_02_10.pkl (18 MB)
```

---

## 📁 파일 구조

```
baselines/prpd_classification/
├── README.md                              ← 본 문서
├── .gitignore
├── download_weights.py                    ← HF 가중치 다운로드 헬퍼
│
├── main_resnet.py                         ← ResNet152 학습 진입점
├── main_efficientnet.py                   ← EfficientNet-B0 학습 진입점
├── train.py                               ← 학습 루프 (Train 클래스)
├── network.py                             ← 모델 정의 (ResNet152, EfficientNet_b0)
├── module.py                              ← 모델 빌더, 평가지표, CM 저장
├── make_dataset.py                        ← 시간순 8:1:1 데이터 분할
├── data_loader.py                         ← Dataset 및 DataLoader
│
├── confusion_matrix_image.py              ← 샘플별 예측 결과 CSV 저장
├── make_confusion_matrix_resnet.py        ← ResNet 추론 진입점
└── make_confusion_matrix_efficientnet.py  ← EfficientNet 추론 진입점
```

---

## 🚀 실행 방법

### 1. 환경 준비

```bash
pip install torch torchvision pandas scikit-learn matplotlib seaborn tqdm pillow huggingface_hub
```

### 2. 데이터 준비

AI Hub에서 데이터셋을 다운로드한 뒤 아래와 같이 배치:

```
<DATA_ROOT>/
├── 원천데이터/
│   └── {PD종류}/{절연체}/{설비명}/*.png
└── 라벨링데이터/
    └── {PD종류}/{절연체}/{설비명}/*.json   (label.PD_type 정수 포함)
```

- **PD종류**: 정상 / 노이즈 / 표면방전 / 코로나방전 / 보이드방전
- **절연체**: 고체 / 액체 / 기체
- **설비명**: ACSR-OC / CNCV-W / TFR-CV / 7.2kV배전반 / 22.9kV배전반 / 25.8kVGIS / 단상유입변압기 / 전력용유입변압기 / 계기용변압기

### 3. 학습

```bash
# ResNet152 학습
python main_resnet.py \
    --raw_data_path /path/to/원천데이터/ \
    --model_path ./ResNet_Result \
    --result_path ./ResNet_Result/ \
    --device cuda:0

# EfficientNet-B0 학습
python main_efficientnet.py \
    --raw_data_path /path/to/원천데이터/ \
    --model_path ./EfficientNet_Result \
    --result_path ./EfficientNet_Result/ \
    --device cuda:0
```

### 4. 사전학습 가중치로 추론 (혼동행렬 생성)

```bash
python make_confusion_matrix_resnet.py \
    --load_model_path ./checkpoints/resnet152/Resnet_2023_11_18_16_09_22_4.pkl \
    --raw_data_path /path/to/원천데이터/ \
    --device cuda:0
```

---

## ⚠️ 코드 분석 시 발견된 이슈 (참고용)

원본 베이스라인 코드를 그대로 보관하고 있으나, 학습 시 아래 사항 인지가 필요합니다.

| 위치 | 이슈 | 영향 |
|---|---|---|
| `train.py:119, 160` | `valid_loss += valid_loss.item()` — tensor에 float 누적 | Validation/Test loss 값 부정확 가능성 |
| `train.py:101` | `param_group['lr'] = self.lr` — 감소된 `lr`이 아닌 원본 할당 | LR decay 실제 미적용 |
| `data_loader.py:25-29` | `Normalize → Resize` 순서 | 정규화 후 리사이즈 → 비효율 |
| 전반 | AMP / `torch.compile` 미적용 | 3090에서 2배 가속 여지 |
| 전반 | `cudnn.deterministic=True, benchmark=False` | 재현성 ↑ / 속도 ↓ (20~30%) |

---

## 📜 라이선스

- **원본 코드 및 가중치**: AI Hub 데이터셋 라이선스를 따릅니다.
  - [AI Hub 이용약관](https://aihub.or.kr/intrcn/intrcn.do?currMenu=151&topMenu=105) 참조
- **본 레포의 README/스크립트 추가분**: 학습/연구 목적 자유 사용

---

## 🔗 관련 링크

- 데이터셋 원본: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71682
- 사전학습 가중치(HF): https://huggingface.co/spaces/zero-coke/PRPD_baseline_pt
