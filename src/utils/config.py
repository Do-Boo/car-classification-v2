"""
설정 관리 모듈 - Hydra 의존성 제거
환경변수와 코드 기반 설정으로 안정성 확보
"""

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    backbone: str = "resnet50"
    num_classes: int = 396
    pretrained: bool = True
    custom_head: bool = False
    dropout: float = 0.2
    
    # fallback 모델 (메인 모델 실패시)
    fallback_backbone: str = "resnet18"
    
    def __post_init__(self):
        """설정 검증"""
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0,1], got {self.dropout}")


@dataclass
class TrainConfig:
    """훈련 관련 설정"""
    root_dir: str = "./data/train"
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    
    # K-Fold 설정
    kfold: int = 5
    
    # 저장 및 체크포인트
    save_dir: str = "./checkpoints_v2"
    save_every_n_epochs: int = 5
    max_checkpoints_keep: int = 3
    
    # 메모리 관리
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # 조기 종료
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # 재시작 관련
    resume_path: Optional[str] = None
    auto_resume: bool = True
    
    def __post_init__(self):
        """설정 검증 및 경로 생성"""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.root_dir, exist_ok=True)


@dataclass
class AugmentationConfig:
    """데이터 증강 설정"""
    level: str = "medium"  # light, medium, heavy, car_specific
    
    # 고급 증강 기법
    use_cutmix: bool = False
    use_mixup: bool = False
    use_gridmask: bool = False
    use_autoaugment: bool = False
    
    # 증강 파라미터
    cutmix_alpha: float = 1.0
    mixup_alpha: float = 1.0
    autoaugment_policy: str = "imagenet"
    
    # 검증용 증강 (일반적으로 False)
    augment_validation: bool = False
    
    def __post_init__(self):
        """설정 검증"""
        valid_levels = ["light", "medium", "heavy", "car_specific"]
        if self.level not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}, got {self.level}")


@dataclass  
class InferenceConfig:
    """추론 관련 설정"""
    test_csv: str = "./data/test.csv"
    output_path: str = "./outputs/submission.csv"
    model_path: Optional[str] = None
    
    # 배치 설정
    batch_size: int = 64  # 추론시 더 큰 배치 사용
    num_workers: int = 4
    
    # TTA 설정
    use_tta: bool = False
    tta_times: int = 5
    
    # 메모리 관리
    max_memory_gb: float = 8.0
    force_cpu: bool = False
    
    def __post_init__(self):
        """설정 검증 및 경로 생성"""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


@dataclass
class EnsembleConfig:
    """앙상블 관련 설정"""
    pred_dir: str = "./outputs"
    output_dir: str = "./outputs" 
    output_name: str = "ensemble_submission.csv"
    
    # 앙상블 방법
    ensemble_method: str = "average"  # average, weighted, voting
    weights: Optional[List[float]] = None
    
    # 필터링
    min_confidence: float = 0.1
    filter_poor_predictions: bool = True


@dataclass
class WandBConfig:
    """WandB 로깅 설정"""
    enabled: bool = False  # 기본적으로 비활성화
    project_name: str = "hecto-ai-car-classification"
    entity: Optional[str] = None
    
    # 로깅 설정
    log_model: bool = True
    log_gradients: bool = False
    log_frequency: int = 100
    
    # 오프라인 모드
    offline: bool = False
    
    # WandB 관련 태그
    tags: List[str] = field(default_factory=lambda: ["v2", "car-classification"])


@dataclass
class SystemConfig:
    """시스템 관련 설정"""
    # GPU 설정
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = True
    
    # 디버깅
    debug: bool = False
    verbose: bool = True
    
    # 재현성
    seed: int = 42
    deterministic: bool = True
    
    # 성능 모니터링
    profile_memory: bool = False
    profile_time: bool = False


class Config:
    """전체 설정을 관리하는 메인 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        설정 초기화
        
        Args:
            config_path: JSON 설정 파일 경로 (선택사항)
        """
        # 기본 설정 로드
        self.model = ModelConfig()
        self.train = TrainConfig()
        self.augmentation = AugmentationConfig()
        self.inference = InferenceConfig()
        self.ensemble = EnsembleConfig()
        self.wandb = WandBConfig()
        self.system = SystemConfig()
        
        # 환경변수에서 설정 오버라이드
        self._load_from_env()
        
        # JSON 파일에서 설정 로드 (있으면)
        if config_path and os.path.exists(config_path):
            self.load_from_json(config_path)
            
        # 시스템 설정 적용
        self._apply_system_config()
    
    def _load_from_env(self):
        """환경변수에서 주요 설정 로드"""
        # 모델 설정
        if os.getenv("MODEL_BACKBONE"):
            self.model.backbone = os.getenv("MODEL_BACKBONE")
        if os.getenv("NUM_CLASSES"):
            self.model.num_classes = int(os.getenv("NUM_CLASSES"))
        
        # 훈련 설정
        if os.getenv("TRAIN_ROOT_DIR"):
            self.train.root_dir = os.getenv("TRAIN_ROOT_DIR")
        if os.getenv("BATCH_SIZE"):
            self.train.batch_size = int(os.getenv("BATCH_SIZE"))
        if os.getenv("LEARNING_RATE"):
            self.train.lr = float(os.getenv("LEARNING_RATE"))
        if os.getenv("EPOCHS"):
            self.train.epochs = int(os.getenv("EPOCHS"))
        if os.getenv("RESUME_PATH"):  # 체크포인트 경로 환경변수 추가
            self.train.resume_path = os.getenv("RESUME_PATH")
            print(f"✅ 체크포인트 경로 설정: {self.train.resume_path}")
        
        # 추론 설정
        if os.getenv("TEST_CSV"):
            self.inference.test_csv = os.getenv("TEST_CSV")
        if os.getenv("OUTPUT_PATH"):
            self.inference.output_path = os.getenv("OUTPUT_PATH")
        
        # WandB 설정
        if os.getenv("WANDB_ENABLED"):
            self.wandb.enabled = os.getenv("WANDB_ENABLED").lower() == "true"
        if os.getenv("WANDB_PROJECT"):
            self.wandb.project_name = os.getenv("WANDB_PROJECT")
        
        # 시스템 설정
        if os.getenv("DEVICE"):
            self.system.device = os.getenv("DEVICE")
        if os.getenv("DEBUG"):
            self.system.debug = os.getenv("DEBUG").lower() == "true"
    
    def _apply_system_config(self):
        """시스템 설정 적용"""
        # 디바이스 자동 감지
        if self.system.device == "auto":
            if torch.cuda.is_available():
                self.system.device = "cuda"
            else:
                self.system.device = "cpu"
        
        # 재현성 설정
        if self.system.deterministic:
            torch.manual_seed(self.system.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.system.seed)
                torch.cuda.manual_seed_all(self.system.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # 디버그 모드 설정
        if self.system.debug:
            self.train.epochs = min(self.train.epochs, 2)
            self.train.batch_size = min(self.train.batch_size, 8)
            self.inference.batch_size = min(self.inference.batch_size, 8)
            print("🐛 디버그 모드: 에포크/배치 크기 축소")
    
    def load_from_json(self, config_path: str):
        """JSON 파일에서 설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 각 섹션별로 설정 업데이트
            if "model" in config_dict:
                self._update_config(self.model, config_dict["model"])
            if "train" in config_dict:
                self._update_config(self.train, config_dict["train"])
            if "augmentation" in config_dict:
                self._update_config(self.augmentation, config_dict["augmentation"])
            if "inference" in config_dict:
                self._update_config(self.inference, config_dict["inference"])
            if "ensemble" in config_dict:
                self._update_config(self.ensemble, config_dict["ensemble"])
            if "wandb" in config_dict:
                self._update_config(self.wandb, config_dict["wandb"])
            if "system" in config_dict:
                self._update_config(self.system, config_dict["system"])
            
            print(f"✅ 설정 파일 로드 완료: {config_path}")
            
        except Exception as e:
            print(f"⚠️  설정 파일 로드 실패: {config_path} - {str(e)}")
            print("기본 설정을 사용합니다.")
    
    def _update_config(self, config_obj, config_dict: Dict[str, Any]):
        """설정 객체 업데이트"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                print(f"⚠️  알 수 없는 설정: {key}")
    
    def save_to_json(self, config_path: str):
        """현재 설정을 JSON 파일로 저장"""
        config_dict = {
            "model": self._dataclass_to_dict(self.model),
            "train": self._dataclass_to_dict(self.train),
            "augmentation": self._dataclass_to_dict(self.augmentation),
            "inference": self._dataclass_to_dict(self.inference),
            "ensemble": self._dataclass_to_dict(self.ensemble),
            "wandb": self._dataclass_to_dict(self.wandb),
            "system": self._dataclass_to_dict(self.system)
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 설정 저장 완료: {config_path}")
    
    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """dataclass를 dict로 변환"""
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if value is not None:
                result[field_name] = value
        return result
    
    def print_config(self):
        """현재 설정 출력"""
        print("\n" + "="*60)
        print("🔧 현재 설정")
        print("="*60)
        
        sections = [
            ("📱 모델", self.model),
            ("🏋️ 훈련", self.train), 
            ("🎨 증강", self.augmentation),
            ("🔮 추론", self.inference),
            ("🔗 앙상블", self.ensemble),
            ("📊 WandB", self.wandb),
            ("💻 시스템", self.system)
        ]
        
        for section_name, section_obj in sections:
            print(f"\n{section_name}:")
            for field_name in section_obj.__dataclass_fields__:
                value = getattr(section_obj, field_name)
                print(f"   {field_name}: {value}")
        
        print("="*60)
    
    def validate(self) -> bool:
        """전체 설정 검증"""
        try:
            # 각 섹션의 __post_init__ 호출해서 검증
            self.model.__post_init__()
            self.train.__post_init__()
            self.augmentation.__post_init__()
            self.inference.__post_init__()
            
            # 추가 교차 검증
            if self.train.batch_size > 128 and self.system.device == "cpu":
                print("⚠️  CPU에서 큰 배치 크기 사용 중")
            
            if self.wandb.enabled and not self.wandb.project_name:
                raise ValueError("WandB 활성화 시 project_name 필요")
            
            print("✅ 설정 검증 완료")
            return True
            
        except Exception as e:
            print(f"❌ 설정 검증 실패: {str(e)}")
            return False


def create_default_config() -> Config:
    """기본 설정 생성"""
    return Config()


def create_quick_test_config() -> Config:
    """빠른 테스트용 설정 생성"""
    config = Config()
    
    # 테스트용 설정 오버라이드
    config.model.backbone = "resnet18"
    config.train.epochs = 2
    config.train.batch_size = 8
    config.train.kfold = 2
    config.inference.batch_size = 8
    config.system.debug = True
    config.wandb.enabled = False
    
    return config


def create_production_config() -> Config:
    """프로덕션용 설정 생성"""
    config = Config()
    
    # 프로덕션 최적화 설정
    config.model.backbone = "efficientnet_b2"
    config.train.epochs = 50
    config.train.batch_size = 32
    config.augmentation.level = "car_specific"
    config.augmentation.use_cutmix = True
    config.inference.use_tta = True
    config.wandb.enabled = True
    
    return config


# 편의 함수들
def get_config(config_type: str = "default") -> Config:
    """설정 타입별 설정 객체 반환"""
    if config_type == "default":
        return create_default_config()
    elif config_type == "test":
        return create_quick_test_config()
    elif config_type == "production":
        return create_production_config()
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 설정 모듈 테스트")
    
    # 기본 설정 테스트
    config = create_default_config()
    config.print_config()
    config.validate()
    
    # JSON 저장/로드 테스트
    config.save_to_json("./test_config.json")
    
    # 환경변수 테스트
    os.environ["MODEL_BACKBONE"] = "resnet34"
    os.environ["BATCH_SIZE"] = "16"
    
    config2 = Config()
    print(f"\n환경변수 적용 결과:")
    print(f"backbone: {config2.model.backbone}")
    print(f"batch_size: {config2.train.batch_size}")
    
    print("\n✅ 설정 모듈 테스트 완료")