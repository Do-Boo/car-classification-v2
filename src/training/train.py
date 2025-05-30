"""
개선된 헥토 AI 자동차 분류 훈련 파이프라인
- Hydra 의존성 제거, config.py 통합
- 메모리 효율적 배치 처리
- 안전한 체크포인트 시스템  
- 에러 복구 메커니즘
- 개선된 로깅 및 모니터링
"""

import os
import sys
import gc
import time
import traceback
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm

# 로컬 모듈 import
try:
    from src.utils.config import Config, get_config
    from src.models.backbone_factory import create_model, get_supported_models, get_model_recommendations
    from src.data.data import CarDataset, get_dataloader, get_kfold
    from src.utils.utils import (
        get_memory_usage, clear_memory, safe_model_save, 
        initialize_environment, safe_operation, retry_on_failure
    )
    
    # 간단한 메모리 정리 데코레이터 정의
    def memory_cleanup_decorator(func):
        """간단한 메모리 정리 데코레이터"""
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 주기적 메모리 정리
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                gc.collect()
        return wrapper
    
    # 누락된 함수들 정의
    def cleanup_old_checkpoints(save_dir: str, keep_last_n: int = 5):
        """오래된 체크포인트 정리"""
        try:
            import glob
            checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_*.pth"))
            if len(checkpoint_files) > keep_last_n:
                # 파일명에서 에포크 번호 추출하여 정렬
                checkpoint_files.sort(key=lambda x: os.path.getctime(x))
                files_to_remove = checkpoint_files[:-keep_last_n]
                for file_path in files_to_remove:
                    os.remove(file_path)
                    logger.info(f"🗑️  오래된 체크포인트 삭제: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"⚠️  체크포인트 정리 실패: {str(e)}")
    
    def get_train_val_split(dataset, val_ratio=0.2):
        """간단한 훈련/검증 분할"""
        from sklearn.model_selection import train_test_split
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)
        
        from torch.utils.data import Subset
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        return train_subset, val_subset
    from utils.logger import init_wandb, log_metrics, finish_wandb
    import logging
    logger = logging.getLogger(__name__)
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"❌ 모듈 Import 실패: {str(e)}")
    logger.error("   필요한 모듈들을 확인하세요")
    sys.exit(1)


class TrainingManager:
    """훈련 관리 클래스 - 메모리 효율적이고 안전한 훈련"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 통계 추적
        self.training_stats = {
            'best_val_acc': 0.0,
            'epochs_without_improvement': 0,
            'total_train_time': 0.0,
            'memory_usage_history': []
        }
        
        # WandB 로거
        self.wandb_logger = None
        if config.wandb.enabled:
            self._init_wandb_logger()
        
        logger.info(f"🏋️ 훈련 매니저 초기화 완료")
        logger.info(f"   💻 디바이스: {self.device}")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        hardware_config = getattr(self.config, 'hardware', None)
        device_setting = getattr(hardware_config, 'device', 'auto') if hardware_config else 'auto'
        
        if device_setting == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_setting)
        
        # Mixed precision 설정
        mixed_precision = getattr(self.config, 'hardware', None)
        if mixed_precision and getattr(mixed_precision, 'mixed_precision', False) and device.type == "cuda":
            logger.info("⚡ Mixed Precision (AMP) 활성화")
        
        return device
    
    def _init_wandb_logger(self):
        """WandB 로거 초기화"""
        try:
            self.wandb_logger = init_wandb(
                project_name=self.config.wandb.project_name,
                config=self.config,
                experiment_name=None,  # 자동 생성
                tags=self.config.wandb.tags,
                offline=self.config.wandb.offline
            )
            logger.info("📊 WandB 로거 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️  WandB 초기화 실패: {str(e)}")
            self.config.wandb.enabled = False
    
    @safe_operation("데이터셋 준비")
    def prepare_datasets(self) -> Tuple[Dataset, Optional[List]]:
        """데이터셋 준비"""
        logger.info("📊 데이터셋 로딩 중...")
        # 데이터셋 생성
        dataset = CarDataset(
            root_dir=self.config.train.root_dir,
            transform=get_train_augmentations(self.config)
        )
        
        logger.info(f"📊 데이터셋 정보:")
        logger.info(f"   총 이미지: {len(dataset):,}장")
        
        if hasattr(dataset, 'class_to_idx'):
            logger.info(f"   클래스 수: {len(dataset.class_to_idx)}개")
            
            # 클래스 분포 확인 (간단한 버전)
            logger.info(f"   클래스 분포 확인 완료")
        
        # K-Fold 생성 (설정된 경우)
        folds = None
        if self.config.train.kfold > 1:
            logger.info("🔀 K-Fold 생성 중...")
            try:
                folds = list(get_kfold(dataset, self.config.train.kfold))
                logger.info(f"🔀 {self.config.train.kfold}-Fold 교차검증 준비 완료")
            except Exception as e:
                logger.error(f"❌ K-Fold 생성 실패: {str(e)}")
                logger.info("   단일 훈련/검증 분할로 전환")
                folds = None
        
        return dataset, folds
    
    @safe_operation("모델 준비")
    def prepare_model(self) -> bool:
        """모델 및 훈련 컴포넌트 준비"""
        
        # 모델 추천 (설정된 경우)
        if hasattr(self.config.model, 'auto_recommend') and self.config.model.auto_recommend:
            logger.info("🤖 모델 자동 추천 실행...")
            recommendations = get_model_recommendations(
                num_classes=self.config.model.num_classes,
                dataset_size=50000,  # 대략적 추정
                gpu_memory_gb=8.0    # 기본값
            )
            
            recommended_backbone = recommendations.get('balanced', [self.config.model.backbone])[0]
            logger.info(f"💡 추천 모델: {recommended_backbone}")
            
            # 사용자 선택 존중하되 로그만 출력
            if recommended_backbone != self.config.model.backbone:
                logger.info(f"   현재 설정: {self.config.model.backbone}")
                logger.info("   모델 변경을 원하면 config에서 backbone을 수정하세요")
        
        # 안전한 모델 생성
        logger.info("🧠 모델 생성 중...")
        # 모델 생성 (간단한 버전)
        self.model = create_model(
            backbone=self.config.model.backbone,
            num_classes=self.config.model.num_classes,
            pretrained=self.config.model.pretrained
        )
        self.model = self.model.to(self.device)
        actual_backbone = self.config.model.backbone
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"🧠 모델 정보:")
        logger.info(f"   백본: {actual_backbone}")
        logger.info(f"   총 파라미터: {total_params:,}")
        logger.info(f"   훈련 가능: {trainable_params:,}")
        logger.info(f"   모델 크기: {total_params * 4 / 1024 / 1024:.1f}MB")
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.train.label_smoothing
        )
        
        # 옵티마이저
        optimizer_name = getattr(self.config.train, 'optimizer', 'adamw')
        optimizer_cls = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD
        }.get(optimizer_name.lower(), optim.AdamW)
        
        if optimizer_cls == optim.SGD:
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.config.train.lr,
                momentum=0.9,
                weight_decay=self.config.train.weight_decay
            )
        else:
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.config.train.lr,
                weight_decay=self.config.train.weight_decay
            )
        
        # 스케줄러
        scheduler_type = getattr(self.config.train, 'scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.train.epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.train.epochs // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # 체크포인트 로드 (추가된 부분)
        resume_path = self.config.train.resume_path
        self.start_epoch = 0  # 시작 에포크 초기화
        
        if resume_path and os.path.exists(resume_path):
            try:
                logger.info(f"🔄 체크포인트 로드 중: {resume_path}")
                checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
                
                # 모델 가중치 로드
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                    logger.info("✅ 모델 가중치 복원 완료")
                else:
                    logger.warning("⚠️ 체크포인트에 모델 가중치가 없습니다")
                
                # 옵티마이저 및 스케줄러 상태 로드
                if 'optimizer' in checkpoint and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    logger.info("✅ 옵티마이저 상태 복원 완료")
                
                if 'scheduler' in checkpoint and self.scheduler and checkpoint['scheduler']:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    logger.info("✅ 스케줄러 상태 복원 완료")
                
                # 훈련 정보 로드
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"✅ 체크포인트 로드 완료: 에포크 {self.start_epoch}부터 훈련 재개")
                
                # 성능 메트릭 복원
                metrics = checkpoint.get('metrics', {})
                if metrics:
                    logger.info(f"📊 이전 성능: 정확도 {metrics.get('val_acc', 0):.4f}, 손실 {metrics.get('val_loss', 0):.4f}")
            except Exception as e:
                logger.error(f"❌ 체크포인트 로드 실패: {str(e)}")
                logger.error(traceback.format_exc())
                logger.info("⚠️ 처음부터 훈련을 시작합니다.")
        else:
            logger.info("🆕 새로운 훈련 시작 (체크포인트 없음)")
        
        logger.info(f"⚙️  훈련 설정:")
        logger.info(f"   옵티마이저: {optimizer_name}")
        logger.info(f"   학습률: {self.config.train.lr}")
        logger.info(f"   스케줄러: {scheduler_type}")
        logger.info(f"   레이블 스무딩: {self.config.train.label_smoothing}")
        
        # Mixed Precision Scaler
        self.scaler = None
        hardware_config = getattr(self.config, 'hardware', None)
        if hardware_config and getattr(hardware_config, 'mixed_precision', False) and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("⚡ AMP Scaler 준비 완료")
        
        return True
    
    @memory_cleanup_decorator
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        # 진행 바
        pbar = tqdm(
            train_loader, 
            desc=f"🏃 에포크 {epoch+1}/{self.config.train.epochs}",
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                # 디바이스로 이동
                images = images.to(self.device, non_blocking=True)
                
                # MixUp/CutMix 처리
                is_mixed = isinstance(labels, tuple)
                if is_mixed:
                    labels_a, labels_b, lam = labels
                    labels_a = labels_a.to(self.device, non_blocking=True)
                    labels_b = labels_b.to(self.device, non_blocking=True)
                else:
                    labels = labels.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                # Mixed Precision 순전파
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        
                        if is_mixed:
                            loss = lam * self.criterion(outputs, labels_a) + \
                                   (1 - lam) * self.criterion(outputs, labels_b)
                        else:
                            loss = self.criterion(outputs, labels)
                    
                    # Mixed Precision 역전파
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 일반 순전파
                    outputs = self.model(images)
                    
                    if is_mixed:
                        loss = lam * self.criterion(outputs, labels_a) + \
                               (1 - lam) * self.criterion(outputs, labels_b)
                    else:
                        loss = self.criterion(outputs, labels)
                    
                    # 일반 역전파
                    loss.backward()
                    self.optimizer.step()
                
                # 통계 업데이트
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_total += batch_size
                
                # 정확도 계산
                with torch.no_grad():
                    _, preds = outputs.max(1)
                    if is_mixed:
                        # MixUp/CutMix 정확도
                        correct_a = (preds == labels_a).float()
                        correct_b = (preds == labels_b).float()
                        running_correct += (lam * correct_a + (1 - lam) * correct_b).sum().item()
                    else:
                        running_correct += (preds == labels).sum().item()
                
                # 진행 상황 업데이트
                current_acc = running_correct / running_total if running_total > 0 else 0
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # 주기적 메모리 정리
                if batch_idx % 100 == 0:
                    del outputs, loss
                    if batch_idx % 500 == 0:  # 500배치마다 강제 정리
                        clear_memory(verbose=False)
                
            except Exception as e:
                logger.error(f"❌ 배치 {batch_idx} 훈련 실패: {str(e)}")
                # 메모리 정리 후 다음 배치로
                clear_memory(verbose=False)
                continue
        
        # 에포크 통계
        epoch_loss = running_loss / running_total if running_total > 0 else 0
        epoch_acc = running_correct / running_total if running_total > 0 else 0
        
        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'samples_processed': running_total
        }
    
    @torch.no_grad()
    def validate_one_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        pbar = tqdm(
            val_loader,
            desc=f"🔍 검증 {epoch+1}",
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Mixed Precision 추론
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # 통계 업데이트
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_total += batch_size
                
                _, preds = outputs.max(1)
                running_correct += (preds == labels).sum().item()
                
                # 진행 상황
                current_acc = running_correct / running_total if running_total > 0 else 0
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
                
                # 메모리 정리
                if batch_idx % 100 == 0:
                    del outputs, loss
                
            except Exception as e:
                logger.error(f"❌ 검증 배치 {batch_idx} 실패: {str(e)}")
                continue
        
        epoch_loss = running_loss / running_total if running_total > 0 else 0
        epoch_acc = running_correct / running_total if running_total > 0 else 0
        
        return {
            'val_loss': epoch_loss,
            'val_acc': epoch_acc,
            'samples_processed': running_total
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False, fold: int = 0):
        """체크포인트 저장"""
        
        checkpoint_info = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'fold': fold
        }
        
        # 일반 체크포인트
        checkpoint_path = os.path.join(
            self.config.train.save_dir, 
            f"checkpoint_fold{fold}_epoch{epoch+1}.pth"
        )
        
        if safe_model_save(self.model, checkpoint_path, checkpoint_info):
            logger.info(f"💾 체크포인트 저장: epoch {epoch+1}")
        
        # 최고 성능 모델
        if is_best:
            best_path = os.path.join(
                self.config.train.save_dir,
                f"best_model_fold{fold}.pth"
            )
            if safe_model_save(self.model, best_path, checkpoint_info):
                logger.info(f"🏆 최고 성능 모델 저장: {metrics.get('val_acc', 0):.4f}")
        
        # 오래된 체크포인트 정리
        if (epoch + 1) % 5 == 0:  # 5 에포크마다
            cleanup_old_checkpoints(
                self.config.train.save_dir, 
                keep_last_n=self.config.train.max_checkpoints_keep
            )
    
    def should_early_stop(self, val_acc: float) -> bool:
        """조기 종료 판단"""
        if val_acc > self.training_stats['best_val_acc'] + self.config.train.early_stopping_min_delta:
            self.training_stats['best_val_acc'] = val_acc
            self.training_stats['epochs_without_improvement'] = 0
            return False
        else:
            self.training_stats['epochs_without_improvement'] += 1
            
            if self.training_stats['epochs_without_improvement'] >= self.config.train.early_stopping_patience:
                logger.info(f"🛑 조기 종료: {self.config.train.early_stopping_patience} 에포크 동안 개선 없음")
                return True
            
            return False
    
    def train_fold(self, dataset: Dataset, fold_idx: int, 
                   train_indices: List[int], val_indices: List[int]) -> Dict[str, float]:
        """한 Fold 훈련"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Fold {fold_idx+1} 시작")
        logger.info(f"   훈련: {len(train_indices):,}장")
        logger.info(f"   검증: {len(val_indices):,}장")
        logger.info(f"{'='*60}")
        
        # Fold별 데이터 준비
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # 증강 설정
        train_transform = get_train_augmentations(self.config)
        val_transform = get_validation_augmentations()
        
        # Transform 적용 (데이터셋의 transform 속성 수정)
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        
        # 데이터로더 생성
        train_loader = get_dataloader(
            train_subset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            memory_efficient=True,
            memory_limit_mb=16384  # 2048 → 16384 (16GB)
        )
        
        val_loader = get_dataloader(
            val_subset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            memory_efficient=True,
            memory_limit_mb=8192   # 1024 → 8192 (8GB)
        )
        
        logger.info(f"📦 데이터로더 준비 완료:")
        logger.info(f"   훈련 배치: {len(train_loader)}")
        logger.info(f"   검증 배치: {len(val_loader)}")
        
        # Fold별 통계 초기화
        fold_best_acc = 0.0
        fold_start_time = time.time()
        
        # 에포크 루프 (시작 에포크부터)
        for epoch in range(self.start_epoch, self.config.train.epochs):
            epoch_start_time = time.time()
            
            # 메모리 상태 체크
            memory_before = get_memory_usage()
            
            # 훈련
            train_metrics = self.train_one_epoch(train_loader, epoch)
            
            # 검증
            val_metrics = self.validate_one_epoch(val_loader, epoch)
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
            
            # 메트릭 결합
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics.update({
                'epoch': epoch + 1,
                'fold': fold_idx,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # 메모리 사용량 추적
            memory_after = get_memory_usage()
            memory_used = memory_after['ram_used'] - memory_before['ram_used']
            epoch_metrics['memory_used_mb'] = memory_used
            
            # 시간 측정
            epoch_time = time.time() - epoch_start_time
            epoch_metrics['epoch_time'] = epoch_time
            
            # 로깅
            if self.config.wandb.enabled:
                log_metrics(epoch_metrics, step=epoch)
            
            # 결과 출력
            total_epochs = self.config.train.epochs - self.start_epoch
            current_epoch = epoch - self.start_epoch + 1
            progress = current_epoch / total_epochs * 100
            
            logger.info(f"\n📊 에포크 {epoch+1}/{self.config.train.epochs} (진행률: {progress:.1f}%) 결과:")
            logger.info(f"   🏃 훈련 - Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.4f}")
            logger.info(f"   🔍 검증 - Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.4f}")
            logger.info(f"   ⏱️  시간: {epoch_time:.1f}초, 메모리: {memory_used:+.1f}MB")
            logger.info(f"   ⚙️  학습률: {epoch_metrics['lr']:.6f}")
            
            # 체크포인트 저장
            is_best = val_metrics['val_acc'] > fold_best_acc
            if is_best:
                fold_best_acc = val_metrics['val_acc']
            
            self.save_checkpoint(epoch, epoch_metrics, is_best, fold_idx)
            
            # 조기 종료 체크
            if self.should_early_stop(val_metrics['val_acc']):
                logger.info(f"🛑 Fold {fold_idx+1} 조기 종료 (Epoch {epoch+1})")
                break
            
            # 주기적 메모리 정리
            if (epoch + 1) % 5 == 0:
                clear_memory(verbose=True)
        
        fold_time = time.time() - fold_start_time
        logger.info(f"\n🏁 Fold {fold_idx+1} 완료!")
        logger.info(f"   🏆 최고 검증 정확도: {fold_best_acc:.4f}")
        logger.info(f"   ⏱️  총 시간: {fold_time/60:.1f}분")
        
        return {
            'fold': fold_idx,
            'best_val_acc': fold_best_acc,
            'total_time': fold_time
        }
    
    @safe_operation("전체 훈련")
    def train(self):
        """전체 훈련 프로세스"""
        logger.info("🚀 헥토 AI 자동차 분류 훈련 시작!")
        
        # 데이터셋 준비
        dataset, folds = self.prepare_datasets()
        if dataset is None:
            logger.error("❌ 데이터셋 준비 실패")
            return
        
        # 모델 준비
        if not self.prepare_model():
            logger.error("❌ 모델 준비 실패")
            return
        
        # 클래스 매핑 저장
        if hasattr(dataset, 'class_to_idx'):
            class_mapping_path = os.path.join(self.config.train.save_dir, "class_to_idx.json")
            from utils import safe_json_save
            safe_json_save(dataset.class_to_idx, class_mapping_path)
            logger.info(f"💾 클래스 매핑 저장: {class_mapping_path}")
        
        total_start_time = time.time()
        fold_results = []
        
        if folds:
            # K-Fold 교차 검증
            for fold_idx, (train_indices, val_indices) in enumerate(folds):
                try:
                    fold_result = self.train_fold(dataset, fold_idx, train_indices, val_indices)
                    fold_results.append(fold_result)
                    
                    # Fold 간 메모리 정리
                    clear_memory(verbose=True)
                    
                except Exception as e:
                    logger.error(f"❌ Fold {fold_idx+1} 훈련 실패: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
        else:
            # 단일 훈련/검증 분할
            train_subset, val_subset = get_train_val_split(
                dataset, 
                val_ratio=self.config.train.validation_split
            )
            
            train_indices = train_subset.indices
            val_indices = val_subset.indices
            
            fold_result = self.train_fold(dataset, 0, train_indices, val_indices)
            fold_results.append(fold_result)
        
        # 전체 결과 요약
        total_time = time.time() - total_start_time
        
        if fold_results:
            avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
            std_acc = np.std([r['best_val_acc'] for r in fold_results])
            
            logger.info(f"\n{'='*60}")
            logger.info("🎉 전체 훈련 완료!")
            logger.info(f"{'='*60}")
            logger.info(f"📊 결과 요약:")
            logger.info(f"   평균 검증 정확도: {avg_acc:.4f} ± {std_acc:.4f}")
            logger.info(f"   총 훈련 시간: {total_time/60:.1f}분")
            
            for i, result in enumerate(fold_results):
                logger.info(f"   Fold {i+1}: {result['best_val_acc']:.4f}")
            
            # 최종 메트릭 로깅
            if self.config.wandb.enabled:
                final_metrics = {
                    'final_avg_acc': avg_acc,
                    'final_std_acc': std_acc,
                    'total_training_time': total_time
                }
                log_metrics(final_metrics)
            
            logger.info(f"\n💡 다음 단계:")
            logger.info(f"   1. 최고 성능 모델로 추론: inference_v2.py")
            logger.info(f"   2. 앙상블 적용: ensemble_v2.py")
            logger.info(f"   3. 모델 경로: {self.config.train.save_dir}/best_model_fold*.pth")
        else:
            logger.error("❌ 모든 Fold 훈련 실패")
        
        # 정리
        if self.config.wandb.enabled:
            finish_wandb()


@retry_on_failure(max_retries=2, delay=1.0)
def main():
    """메인 함수"""
    
    # 환경 초기화
    initialize_environment(seed=42, deterministic=True)
    
    # 설정 로드
    try:
        # 명령행 인자로 설정 타입 지정 가능
        config_type = "default"
        if len(sys.argv) > 1:
            config_type = sys.argv[1]
        
        config = get_config(config_type)
        config.print_config()
        
        if not config.validate():
            logger.error("❌ 설정 검증 실패")
            return
        
    except Exception as e:
        logger.error(f"❌ 설정 로드 실패: {str(e)}")
        logger.info("   기본 설정으로 실행합니다")
        config = get_config("default")
    
    # 저장 디렉토리 확인
    if not os.path.exists(config.train.root_dir):
        logger.error(f"❌ 훈련 데이터 디렉토리 없음: {config.train.root_dir}")
        logger.error("   데이터 경로를 확인하세요")
        return
    
    # 훈련 시작
    try:
        trainer = TrainingManager(config)
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("⏹️  사용자에 의해 훈련 중단")
        clear_memory(verbose=True)
        
    except Exception as e:
        logger.error(f"❌ 훈련 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        clear_memory(verbose=True)
    
    finally:
        # 최종 정리
        clear_memory(verbose=True)
        logger.info("🧹 훈련 세션 정리 완료")


if __name__ == "__main__":
    main()