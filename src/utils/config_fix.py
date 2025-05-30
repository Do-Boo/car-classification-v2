#!/usr/bin/env python3
"""
수정된 설정 파일
올바른 클래스 수(393)와 ResNet50 백본 사용
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FixedConfig:
    """수정된 훈련 설정"""
    
    # 데이터 관련
    data_root: str = "data"
    train_csv: str = "data/train.csv"
    test_csv: str = "data/test.csv"
    
    # 모델 관련 - 수정된 부분
    backbone: str = "resnet50"  # EfficientNet에서 ResNet50으로 변경
    num_classes: int = 393      # 396에서 393으로 변경 (중복 클래스 제거)
    pretrained: bool = True
    
    # 훈련 관련
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 교차검증
    n_folds: int = 5
    fold: int = 0
    
    # 체크포인트
    checkpoint_dir: str = "checkpoints_v2"
    save_best_only: bool = True
    
    # 기타
    seed: int = 42
    num_workers: int = 4
    device: str = "cpu"  # GPU 사용 가능 시 "cuda"로 변경
    
    # 새로 추가된 설정
    class_mapping_path: str = "checkpoints_v2/class_to_idx.json"
    use_stratified_kfold: bool = True
    early_stopping_patience: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'data_root': self.data_root,
            'train_csv': self.train_csv,
            'test_csv': self.test_csv,
            'backbone': self.backbone,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'n_folds': self.n_folds,
            'fold': self.fold,
            'checkpoint_dir': self.checkpoint_dir,
            'save_best_only': self.save_best_only,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'device': self.device,
            'class_mapping_path': self.class_mapping_path,
            'use_stratified_kfold': self.use_stratified_kfold,
            'early_stopping_patience': self.early_stopping_patience
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FixedConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """설정 유효성 검사"""
        errors = []
        
        # 필수 디렉토리 확인
        if not os.path.exists(self.data_root):
            errors.append(f"데이터 루트 디렉토리가 존재하지 않음: {self.data_root}")
        
        if not os.path.exists(self.train_csv):
            errors.append(f"훈련 CSV 파일이 존재하지 않음: {self.train_csv}")
        
        if not os.path.exists(self.test_csv):
            errors.append(f"테스트 CSV 파일이 존재하지 않음: {self.test_csv}")
        
        # 클래스 매핑 파일 확인
        if not os.path.exists(self.class_mapping_path):
            errors.append(f"클래스 매핑 파일이 존재하지 않음: {self.class_mapping_path}")
        
        # 하이퍼파라미터 범위 확인
        if self.num_classes <= 0:
            errors.append(f"클래스 수는 양수여야 함: {self.num_classes}")
        
        if self.batch_size <= 0:
            errors.append(f"배치 크기는 양수여야 함: {self.batch_size}")
        
        if self.learning_rate <= 0:
            errors.append(f"학습률은 양수여야 함: {self.learning_rate}")
        
        if self.epochs <= 0:
            errors.append(f"에포크 수는 양수여야 함: {self.epochs}")
        
        if errors:
            print("❌ 설정 검증 실패:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ 설정 검증 성공")
        return True
    
    def create_checkpoint_dir(self):
        """체크포인트 디렉토리 생성"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"✅ 체크포인트 디렉토리 생성: {self.checkpoint_dir}")

def get_fixed_config() -> FixedConfig:
    """수정된 기본 설정 반환"""
    config = FixedConfig()
    
    # 체크포인트 디렉토리 생성
    config.create_checkpoint_dir()
    
    print("🔧 수정된 설정 로드:")
    print(f"   - 백본: {config.backbone}")
    print(f"   - 클래스 수: {config.num_classes}")
    print(f"   - 배치 크기: {config.batch_size}")
    print(f"   - 학습률: {config.learning_rate}")
    print(f"   - 에포크: {config.epochs}")
    
    return config

def save_config(config: FixedConfig, save_path: str):
    """설정을 JSON 파일로 저장"""
    import json
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"✅ 설정 저장: {save_path}")

def load_config(config_path: str) -> FixedConfig:
    """JSON 파일에서 설정 로드"""
    import json
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config = FixedConfig.from_dict(config_dict)
    print(f"✅ 설정 로드: {config_path}")
    
    return config

if __name__ == "__main__":
    # 설정 테스트
    config = get_fixed_config()
    
    # 설정 검증
    if config.validate():
        print("✅ 모든 설정이 올바릅니다!")
    else:
        print("❌ 설정에 문제가 있습니다!")
    
    # 설정 저장 테스트
    save_config(config, "checkpoints_v2/fixed_config.json") 