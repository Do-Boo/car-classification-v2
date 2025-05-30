자동차 모델 분류 대회를 위한 딥러닝 프로젝트입니다.

## 프로젝트 개요

- **목표**: 396개 클래스의 자동차 이미지 분류
- **데이터**: 훈련 33,137장, 테스트 8,258장
- **평가지표**: Log Loss
- **모델**: EfficientNet 기반 분류 모델

## 프로젝트 구조

```
hecto_ai_challenge/
├── src/                    # 소스 코드
│   ├── models/            # 모델 정의
│   │   ├── __init__.py
│   │   ├── backbone_factory.py
│   │   └── model.py
│   ├── data/              # 데이터 로더
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── data_utils.py
│   ├── training/          # 학습 관련
│   │   ├── __init__.py
│   │   └── train.py
│   └── utils/             # 유틸리티
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       ├── metrics.py
│       └── utils.py
├── scripts/               # 실행 스크립트
│   ├── inference.py       # 추론 스크립트
│   ├── train_ensemble.py  # 앙상블 학습
│   └── ensemble_inference.py # 앙상블 추론
├── config/                # 설정 파일
│   └── default.yaml
├── checkpoints/           # 모델 체크포인트
├── data/                  # 데이터 폴더
│   ├── train/            # 훈련 데이터
│   ├── test/             # 테스트 데이터
│   └── test.csv          # 테스트 정보
├── docs/                  # 문서
│   └── project_plan.md
├── venv/                  # 가상환경
├── requirements.txt       # 의존성
├── README.md             # 프로젝트 설명
└── submission_folder_based.csv  # 최고 성능 제출 파일
```

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 훈련

```bash
# 기본 훈련
python src/training/train.py

# 특정 설정으로 훈련
python src/training/train.py custom_config
```

### 3. 추론

```bash
# 단일 모델 추론
python scripts/inference.py

# 앙상블 추론
python scripts/ensemble_inference.py
```

### 4. 앙상블

```bash
# 여러 예측 파일 앙상블
python scripts/train_ensemble.py [예측_파일_디렉토리] [출력_파일명]

# 예시
python scripts/train_ensemble.py . submission_ensemble.csv
```

## 성능 결과

### 최고 성능 모델
- **파일**: `submission_folder_based.csv`
- **높은 신뢰도(>0.8) 예측**: 91.9% (7,588개/8,258개)
- **평균 신뢰도**: 0.945
- **모델**: EfficientNet-B0 기반

### 훈련 설정
- **백본**: EfficientNet-B0 (pretrained)
- **옵티마이저**: AdamW (lr=1e-4)
- **스케줄러**: CosineAnnealingLR
- **배치 크기**: 64
- **에포크**: 30 (Early Stopping 적용)

## 주요 기능

### 1. 메모리 최적화
- 16GB 훈련 메모리, 8GB 검증 메모리
- 효율적인 배치 처리
- 자동 메모리 정리

### 2. 안전한 훈련
- 체크포인트 자동 저장
- 에러 복구 메커니즘
- 조기 종료 (Early Stopping)

### 3. 유연한 설정
- YAML 기반 설정 관리
- 다양한 백본 모델 지원
- K-Fold 교차 검증

### 4. 앙상블 지원
- 여러 모델 예측 결합
- 신뢰도 기반 분석
- 자동 통계 생성

## 📈 사용 팁

### 1. 메모리 부족 시
```bash
# 배치 크기 줄이기
# config/default.yaml에서 batch_size 조정
```

### 2. 다른 모델 사용
```bash
# config/default.yaml에서 backbone 변경
# 지원 모델: efficientnet-b0, efficientnet-b1, resnet50 등
```

### 3. 성능 모니터링
```bash
# WandB 로깅 활성화
# config/default.yaml에서 wandb.enabled: true
```

## 다음 단계

1. **최종 제출**: `submission_folder_based.csv` 사용
2. **성능 개선**: 앙상블 또는 추가 데이터 증강
3. **모델 실험**: 다른 백본 모델 시도

## 문제 해결

### 자주 발생하는 문제

1. **Import 오류**: Python path 확인
2. **메모리 부족**: 배치 크기 조정
3. **CUDA 오류**: GPU 메모리 확인

### 로그 확인
```bash
# 훈련 로그 확인
tail -f logs/training.log

# 추론 로그 확인  
tail -f logs/inference.log
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 
