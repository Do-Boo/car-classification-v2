# DACON 자동차 모델 분류 대회 프로젝트 계획

## 📊 프로젝트 개요
- **대회명**: DACON 자동차 모델 분류 대회
- **목표**: 396개 클래스의 자동차 이미지 분류
- **데이터**: 훈련 33,137장, 테스트 8,258장
- **평가지표**: Log Loss

## 🔍 현재 상황 (2025-01-27 업데이트)

### ✅ 완료된 작업
1. **문제점 분석 완료**
   - 클래스 매핑 추가 분석 (396개 클래스 유지)
   - 모델 성능 비교 분석
   - 제출 파일 품질 평가

2. **제출 파일 생성 완료**
   - `submission_folder_based.csv`: 최고 성능 (91.9% 높은 신뢰도)

3. **성능 분석 완료**
   - 신뢰도 기반 모델 비교
   - 예측 일치도 분석
   - 최적 제출 파일 선정

4. **프로젝트 구조 재정리 완료 (2025-01-27)**
   - ✅ 체계적인 폴더 구조로 재구성
   - ✅ v2 표기 완전 제거
   - ✅ import 경로 모두 수정
   - ✅ 실행 스크립트 정리
   - ✅ 문서 업데이트

### 🏆 **최종 추천: submission_folder_based.csv**
- **가장 높은 신뢰도**: 91.9%의 예측이 0.8 이상
- **안정적인 예측**: 높은 평균 신뢰도 (0.945)
- **일관된 성능**: 낮은 표준편차로 안정적

### 📁 **최종 프로젝트 구조 (재정리 완료)**
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
│   ├── ensemble_inference.py # 앙상블 추론
│   └── integration_test.py # 통합 테스트
├── config/                # 설정 파일
│   └── default.yaml
├── checkpoints/           # 모델 체크포인트 (v2 제거)
│   ├── best_model_fold0.pth
│   ├── class_to_idx.json
│   └── ...
├── data/                  # 데이터 폴더
│   ├── train/            # 훈련 데이터 (396개 차종)
│   ├── test/             # 테스트 데이터 (8,258장)
│   ├── test.csv          # 테스트 정보
│   └── sample_submission.csv
├── docs/                  # 문서
│   └── project_plan.md
├── venv/                  # 가상환경
├── requirements.txt       # 의존성 (v2 제거)
├── README.md             # 프로젝트 설명 (v2 제거)
└── submission_folder_based.csv  # 🏆 최고 성능 제출 파일
```

### 🎯 실행 방법 (새로운 구조)

#### 1. 훈련
```bash
# 기본 훈련
python src/training/train.py

# 특정 설정으로 훈련
python src/training/train.py custom_config
```

#### 2. 추론
```bash
# 단일 모델 추론
python scripts/inference.py

# 앙상블 추론
python scripts/ensemble_inference.py
```

#### 3. 앙상블
```bash
# 여러 예측 파일 앙상블
python scripts/train_ensemble.py . submission_ensemble.csv
```

### 🔧 주요 개선사항

#### 1. 체계적인 구조
- **src/**: 모든 소스 코드를 기능별로 분류
- **scripts/**: 실행 가능한 스크립트들
- **config/**: 설정 파일 통합 관리

#### 2. 명확한 네이밍
- v2 표기 완전 제거
- 직관적인 파일명 사용
- 일관된 네이밍 규칙 적용

#### 3. 안전한 Import
- 프로젝트 루트 기반 절대 경로
- 모든 스크립트에서 일관된 import 방식
- 패키지 구조 명확화

### 🎯 다음 단계
1. **최종 제출**: `submission_folder_based.csv` 사용
2. **프로젝트 문서화**: README 및 코드 정리 완료
3. **성능 개선 방안**: 앙상블 또는 추가 데이터 증강 고려

## 🚨 발견된 주요 문제점들 (해결 완료)

### 1. 클래스 매핑 문제 ✅ 해결완료
**문제**: 대회 규칙의 동일 클래스 3쌍 처리 로직 누락
**해결**: 클래스 병합 완료

### 2. 모델 아키텍처 불일치 ✅ 해결완료
**문제**: 훈련(ResNet50) vs 추론(EfficientNet-B4) 불일치
**해결**: EfficientNet-B0으로 통일

### 3. 성능 문제 ✅ 개선완료
**현재 상태**:
- 높은 신뢰도 예측 달성 (91.9%가 0.8 이상)
- 안정적인 예측 성능 확보
- 최적 모델 선정 완료

### 4. 프로젝트 구조 문제 ✅ 해결완료
**문제**: 복잡하고 일관성 없는 파일 구조
**해결**: 체계적인 src/ 기반 구조로 재정리

## 🔧 기술적 세부사항

### 모델 아키텍처
```python
backbone: EfficientNet-B0 (pretrained=True)
classifier: Linear(1280 → 396)
optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
scheduler: CosineAnnealingLR
```

### 데이터 처리
- **이미지 크기**: 224x224
- **배치 크기**: 64 (최적화됨)
- **증강**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- **정규화**: ImageNet 표준

### 하드웨어 성능 최적화
**현재 최적화된 설정**:
- 메모리 제한: 16GB (훈련), 8GB (검증)
- 워커 수: 8
- 배치 크기: 64
- 예상 훈련 시간: 27-33시간

## 📈 성능 개선 전략

### 1. 모델 개선
- [ ] 더 강력한 백본 (EfficientNet-B1, B2)
- [ ] 앙상블 (여러 에포크 모델 조합)
- [ ] Test Time Augmentation (TTA)

### 2. 데이터 개선
- [ ] 클래스 불균형 해결 (가중치, 오버샘플링)
- [ ] 증강 기법 최적화
- [ ] 외부 데이터 활용 검토

### 3. 훈련 최적화
- [ ] 학습률 스케줄링 개선
- [ ] Early Stopping 구현
- [ ] Gradient Accumulation

## 📞 다음 액션 아이템
1. ✅ 제출 파일 비교 분석 완료
2. ✅ 최적 모델 선정 완료
3. ✅ 프로젝트 구조 재정리 완료
4. ✅ 실행 환경 정리 완료
5. 🎯 최종 제출 파일 확정
6. 🚀 성능 개선을 위한 추가 실험 계획
