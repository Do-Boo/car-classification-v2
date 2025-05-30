"""
WandB 로깅 유틸리티
- 실험 추적 및 시각화
- 메트릭 로깅
- 이미지 및 파일 업로드
- 하이퍼파라미터 추적
"""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union
import os
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import warnings
from PIL import Image
import cv2

# 경고 메시지 무시
warnings.filterwarnings('ignore')

class WandBLogger:
    """WandB 로거 클래스"""
    
    def __init__(self, project_name: str, experiment_name: Optional[str] = None,
                 config: Optional[DictConfig] = None, tags: Optional[List[str]] = None,
                 notes: Optional[str] = None, offline: bool = False):
        """
        Args:
            project_name: WandB 프로젝트명
            experiment_name: 실험명 (없으면 자동 생성)
            config: 실험 설정
            tags: 태그 리스트
            notes: 실험 노트
            offline: 오프라인 모드
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.offline = offline
        self.run = None
        self.initialized = False
        
        # WandB 초기화
        try:
            # 설정 변환
            wandb_config = self._convert_config(config) if config else {}
            
            # WandB 시작
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                config=wandb_config,
                tags=tags,
                notes=notes,
                mode="offline" if offline else "online",
                reinit=True
            )
            
            self.initialized = True
            print(f"✅ WandB 로거 초기화 완료")
            print(f"   📊 프로젝트: {project_name}")
            print(f"   🔬 실험명: {self.run.name}")
            print(f"   🔗 URL: {self.run.url if not offline else 'Offline 모드'}")
            
        except Exception as e:
            print(f"⚠️  WandB 초기화 실패: {str(e)}")
            print("   📝 로컬 로깅으로 대체됩니다")
            self.initialized = False
    
    def _convert_config(self, config: DictConfig) -> Dict:
        """OmegaConf 설정을 dict로 변환"""
        if isinstance(config, DictConfig):
            return OmegaConf.to_container(config, resolve=True)
        return config
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """메트릭 로깅"""
        if self.initialized and self.run:
            self.run.log(metrics, step=step)
        else:
            # 로컬 로깅
            step_str = f"Step {step}: " if step is not None else ""
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"📊 {step_str}{metric_str}")
    
    def log_image(self, key: str, image: Union[np.ndarray, torch.Tensor, str], 
                  caption: Optional[str] = None, step: Optional[int] = None):
        """이미지 로깅"""
        if not self.initialized or not self.run:
            return
        
        try:
            # 이미지 타입별 처리
            if isinstance(image, str):
                # 파일 경로
                if os.path.exists(image):
                    wandb_image = wandb.Image(image, caption=caption)
                else:
                    print(f"⚠️  이미지 파일 없음: {image}")
                    return
            elif isinstance(image, torch.Tensor):
                # PyTorch 텐서
                if image.dim() == 4:  # 배치 차원 제거
                    image = image[0]
                if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW -> HWC
                    image = image.permute(1, 2, 0)
                image_np = image.detach().cpu().numpy()
                
                # 정규화 (0-255 범위로)
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                wandb_image = wandb.Image(image_np, caption=caption)
            else:
                # NumPy 배열
                wandb_image = wandb.Image(image, caption=caption)
            
            self.run.log({key: wandb_image}, step=step)
            
        except Exception as e:
            print(f"⚠️  이미지 로깅 실패: {str(e)}")
    
    def log_images(self, key: str, images: List[Union[np.ndarray, torch.Tensor]], 
                   captions: Optional[List[str]] = None, step: Optional[int] = None):
        """다중 이미지 로깅"""
        if not self.initialized or not self.run:
            return
        
        wandb_images = []
        for i, img in enumerate(images):
            caption = captions[i] if captions and i < len(captions) else f"Image {i+1}"
            
            try:
                if isinstance(img, torch.Tensor):
                    if img.dim() == 4:
                        img = img[0]
                    if img.dim() == 3 and img.shape[0] in [1, 3]:
                        img = img.permute(1, 2, 0)
                    img = img.detach().cpu().numpy()
                
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                
                wandb_images.append(wandb.Image(img, caption=caption))
                
            except Exception as e:
                print(f"⚠️  이미지 {i} 처리 실패: {str(e)}")
        
        if wandb_images:
            self.run.log({key: wandb_images}, step=step)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: Optional[List[str]] = None,
                           title: str = "Confusion Matrix", step: Optional[int] = None):
        """혼동 행렬 로깅"""
        if not self.initialized or not self.run:
            return
        
        try:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            
            # 혼동 행렬 시각화
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # 임시 파일로 저장 후 로깅
            temp_path = "temp_confusion_matrix.png"
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.run.log({
                "confusion_matrix": wandb.Image(temp_path, caption=title)
            }, step=step)
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"⚠️  혼동 행렬 로깅 실패: {str(e)}")
    
    def log_learning_curve(self, train_losses: List[float], val_losses: List[float],
                          train_accs: List[float], val_accs: List[float],
                          title: str = "Learning Curves"):
        """학습 곡선 로깅"""
        if not self.initialized or not self.run:
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 손실 곡선
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 정확도 곡선
            ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
            ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            # 임시 파일로 저장 후 로깅
            temp_path = "temp_learning_curve.png"
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.run.log({
                "learning_curves": wandb.Image(temp_path, caption=title)
            })
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"⚠️  학습 곡선 로깅 실패: {str(e)}")
    
    def log_model_architecture(self, model: torch.nn.Module, input_size: tuple = (3, 224, 224)):
        """모델 구조 로깅"""
        if not self.initialized or not self.run:
            return
        
        try:
            # 모델 요약
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # float32 기준
                "input_size": input_size
            }
            
            self.run.log({"model_info": model_info})
            
            # 모델 구조를 텍스트로 저장
            model_summary = str(model)
            with open("model_architecture.txt", "w") as f:
                f.write(f"Model Architecture\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total Parameters: {total_params:,}\n")
                f.write(f"Trainable Parameters: {trainable_params:,}\n")
                f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
                f.write("=" * 50 + "\n\n")
                f.write(model_summary)
            
            # 파일 업로드
            self.run.upload_file("model_architecture.txt")
            
            # 임시 파일 삭제
            if os.path.exists("model_architecture.txt"):
                os.remove("model_architecture.txt")
                
            print(f"📊 모델 정보 로깅 완료")
            print(f"   📊 총 파라미터: {total_params:,}")
            print(f"   📊 훈련 가능: {trainable_params:,}")
            
        except Exception as e:
            print(f"⚠️  모델 구조 로깅 실패: {str(e)}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        if self.initialized and self.run:
            self.run.config.update(hyperparams)
            print(f"⚙️  하이퍼파라미터 업데이트: {len(hyperparams)}개")
    
    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "model"):
        """아티팩트 로깅 (모델, 데이터셋 등)"""
        if not self.initialized or not self.run:
            return
        
        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            print(f"📦 아티팩트 업로드: {artifact_name}")
            
        except Exception as e:
            print(f"⚠️  아티팩트 업로드 실패: {str(e)}")
    
    def log_system_info(self):
        """시스템 정보 로깅"""
        if not self.initialized or not self.run:
            return
        
        try:
            import psutil
            import platform
            
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "pytorch_version": torch.__version__,
            }
            
            # GPU 정보
            if torch.cuda.is_available():
                system_info.update({
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
                })
            
            self.run.log({"system_info": system_info})
            print(f"💻 시스템 정보 로깅 완료")
            
        except Exception as e:
            print(f"⚠️  시스템 정보 로깅 실패: {str(e)}")
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """모델 그래디언트 추적"""
        if self.initialized and self.run:
            try:
                wandb.watch(model, log="all", log_freq=log_freq)
                print(f"👁️  모델 그래디언트 추적 시작 (log_freq={log_freq})")
            except Exception as e:
                print(f"⚠️  모델 추적 실패: {str(e)}")
    
    def finish(self):
        """WandB 세션 종료"""
        if self.initialized and self.run:
            self.run.finish()
            print(f"✅ WandB 세션 종료")
            self.initialized = False

# 전역 로거 인스턴스
_global_logger: Optional[WandBLogger] = None

def init_wandb(project_name: str, config: Optional[DictConfig] = None,
               experiment_name: Optional[str] = None, tags: Optional[List[str]] = None,
               notes: Optional[str] = None, offline: bool = False) -> WandBLogger:
    """WandB 로거 초기화 (전역)"""
    global _global_logger
    
    _global_logger = WandBLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        config=config,
        tags=tags,
        notes=notes,
        offline=offline
    )
    
    return _global_logger

def log_metrics(metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
    """메트릭 로깅 (전역 함수)"""
    if _global_logger:
        _global_logger.log_metrics(metrics, step)
    else:
        print(f"⚠️  WandB 로거가 초기화되지 않았습니다")

def log_image(key: str, image: Union[np.ndarray, torch.Tensor, str], 
              caption: Optional[str] = None, step: Optional[int] = None):
    """이미지 로깅 (전역 함수)"""
    if _global_logger:
        _global_logger.log_image(key, image, caption, step)

def log_model_info(model: torch.nn.Module, input_size: tuple = (3, 224, 224)):
    """모델 정보 로깅 (전역 함수)"""
    if _global_logger:
        _global_logger.log_model_architecture(model, input_size)
        _global_logger.log_system_info()

def finish_wandb():
    """WandB 세션 종료 (전역 함수)"""
    global _global_logger
    if _global_logger:
        _global_logger.finish()
        _global_logger = None

class LocalLogger:
    """WandB 대신 로컬 로깅"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.config_file = self.log_dir / "config.json"
        
        print(f"📁 로컬 로거 초기화: {log_dir}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """메트릭을 JSON Lines 형식으로 저장"""
        log_entry = {"step": step, **metrics}
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # 콘솔 출력
        step_str = f"Step {step}: " if step is not None else ""
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"📊 {step_str}{metric_str}")
    
    def log_config(self, config: Dict[str, Any]):
        """설정 저장"""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def save_plot(self, fig, filename: str):
        """플롯 저장"""
        plot_path = self.log_dir / filename
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📈 플롯 저장: {plot_path}")

def create_experiment_summary(project_name: str, config: DictConfig, 
                            final_metrics: Dict[str, float]) -> str:
    """실험 요약 생성"""
    import datetime
    
    summary = f"""
# 실험 요약: {project_name}
실행 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 설정
- 모델: {config.model.backbone}
- 배치 크기: {config.train.batch_size}
- 학습률: {config.train.lr}
- 에포크: {config.train.epochs}
- 증강 레벨: {config.augmentation.level}

## 최종 결과
"""
    
    for metric, value in final_metrics.items():
        summary += f"- {metric}: {value:.4f}\n"
    
    return summary

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 WandB 로거 테스트")
    
    # 가짜 설정
    from omegaconf import DictConfig
    
    test_config = DictConfig({
        "model": {"backbone": "resnet50", "num_classes": 396},
        "train": {"batch_size": 32, "lr": 0.001, "epochs": 10},
        "augmentation": {"level": "medium"}
    })
    
    # 로거 초기화 (오프라인 모드)
    logger = init_wandb(
        project_name="hecto-ai-test",
        config=test_config,
        experiment_name="test_experiment",
        tags=["test", "car_classification"],
        offline=True
    )
    
    # 테스트 로깅
    if logger and logger.initialized:
        # 메트릭 로깅
        for epoch in range(3):
            metrics = {
                "train_loss": 2.5 - epoch * 0.3,
                "train_acc": 0.3 + epoch * 0.2,
                "val_loss": 2.3 - epoch * 0.25,
                "val_acc": 0.35 + epoch * 0.18
            }
            log_metrics(metrics, step=epoch)
        
        # 시스템 정보 로깅
        logger.log_system_info()
        
        print("✅ 로거 테스트 완료")
        
        # 종료
        finish_wandb()
    else:
        print("⚠️  로거 초기화 실패 - 로컬 로깅으로 대체")
        
        # 로컬 로거 테스트
        local_logger = LocalLogger("./test_logs")
        local_logger.log_metrics({"test_metric": 0.85}, step=1)
        local_logger.log_config(OmegaConf.to_container(test_config))
    
    print("🎉 로거 모듈 테스트 완료!")