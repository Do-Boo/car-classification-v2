"""
개선된 헥토 AI 자동차 분류 추론 파이프라인
- Hydra 의존성 제거, config.py 통합
- 메모리 효율적 TTA 구현
- 스트리밍 배치 추론
- 안전한 결과 저장
- 성능 최적화
"""

import os
import sys
import gc
import time
import traceback
from typing import Optional, Dict, List, Tuple, Generator
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 로컬 모듈 import
try:
    from src.utils.config import Config, get_config
    from src.models.backbone_factory import create_model, get_supported_models
    from src.data.data import CarDataset, get_dataloader
    from src.utils.utils import (
        get_memory_usage, clear_memory, safe_operation, retry_on_failure,
        initialize_environment
    )
    from src.utils.logger import logger
except ImportError as e:
    logger.error(f"❌ 모듈 Import 실패: {str(e)}")
    logger.error("   개선된 모듈들이 필요합니다")
    sys.exit(1)


class InferenceManager:
    """추론 관리 클래스 - 메모리 효율적이고 안전한 추론"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        
        # 성능 통계
        self.inference_stats = {
            'total_images': 0,
            'processing_time': 0.0,
            'avg_time_per_image': 0.0,
            'memory_peak': 0.0,
            'tta_times_used': 0
        }
        
        logger.info(f"🔮 추론 매니저 초기화 완료")
        logger.info(f"   💻 디바이스: {self.device}")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        if self.config.inference.force_cpu:
            device = torch.device("cpu")
            logger.info("💻 강제 CPU 모드")
        elif self.config.system.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.system.device)
        
        return device
    
    @safe_operation("모델 로드")
    def load_model(self) -> bool:
        """모델 및 클래스 매핑 로드"""
        
        # 모델 경로 확인
        if not self.config.inference.model_path:
            logger.error("❌ 모델 경로가 설정되지 않았습니다")
            return False
        
        if not os.path.exists(self.config.inference.model_path):
            logger.error(f"❌ 모델 파일 없음: {self.config.inference.model_path}")
            return False
        
        with memory_monitor("모델 로딩"):
            # 체크포인트에서 설정 로드 시도
            try:
                checkpoint = torch.load(self.config.inference.model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'config' in checkpoint:
                    saved_config = checkpoint['config']
                    # 모델 관련 설정 업데이트
                    self.config.model.backbone = saved_config.model.backbone
                    self.config.model.num_classes = saved_config.model.num_classes
                    logger.info("✅ 체크포인트에서 모델 설정 로드")
            except Exception as e:
                logger.warning(f"⚠️  체크포인트 설정 로드 실패: {str(e)}")
                logger.info("   기본 설정 사용")
            
            # 모델 생성
            self.model, actual_backbone = create_model(
                backbone=self.config.model.backbone,
                num_classes=self.config.model.num_classes,
                pretrained=False,  # 추론시에는 pretrained=False
                device=self.device
            )
            
            # 가중치 로드
            self.model = create_model(
                self.model,
                self.config.inference.model_path,
                strict=False,  # 추론시에는 관대하게
                device=self.device
            )
            
            if self.model is None:
                logger.error("❌ 모델 로딩 실패")
                return False
            
            # 평가 모드로 전환
            self.model.eval()
            
            # 모델 정보
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"🧠 모델 로드 완료:")
            logger.info(f"   백본: {actual_backbone}")
            logger.info(f"   클래스 수: {self.config.model.num_classes}")
            logger.info(f"   파라미터: {total_params:,}")
        
        # 클래스 매핑 로드
        self._load_class_mapping()
        
        return True
    
    def _load_class_mapping(self):
        """클래스 매핑 로드"""
        
        # 클래스 매핑 파일 경로 찾기
        class_mapping_paths = [
            getattr(self.config.inference, 'class_mapping_path', None),
            os.path.join(os.path.dirname(self.config.inference.model_path), "class_to_idx.json"),
            os.path.join(self.config.train.save_dir, "class_to_idx.json"),
            "./class_to_idx.json"
        ]
        
        class_mapping_path = None
        for path in class_mapping_paths:
            if path and os.path.exists(path):
                class_mapping_path = path
                break
        
        if class_mapping_path:
            self.class_to_idx = safe_json_load(class_mapping_path)
            if self.class_to_idx:
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
                logger.info(f"📂 클래스 매핑 로드: {class_mapping_path}")
                logger.info(f"   클래스 수: {len(self.class_to_idx)}")
            else:
                logger.warning(f"⚠️  클래스 매핑 로드 실패: {class_mapping_path}")
        else:
            logger.warning("⚠️  클래스 매핑 파일을 찾을 수 없습니다")
            logger.info("   클래스명을 'class_0', 'class_1', ... 형식으로 생성합니다")
        
        # 클래스명 생성 (매핑이 없는 경우)
        if not self.class_to_idx:
            self.class_names = [f'class_{i}' for i in range(self.config.model.num_classes)]
        else:
            self.class_names = [self.idx_to_class.get(i, f'class_{i}') 
                               for i in range(self.config.model.num_classes)]
    
    def get_tta_transforms(self) -> List[A.Compose]:
        """TTA 변환 리스트 생성"""
        try:
            return get_tta_augmentations()
        except:
            # Fallback TTA 변환들
            logger.warning("⚠️  TTA 변환 로드 실패, 기본 변환 사용")
            return [
                # 원본
                A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ]),
                # 수평 뒤집기
                A.Compose([
                    A.Resize(224, 224),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ]),
                # 작은 회전
                A.Compose([
                    A.Resize(224, 224),
                    A.Rotate(limit=10, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            ]
    
    @memory_cleanup_decorator
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """배치 단위 예측"""
        images = images.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            # Mixed Precision 추론 (가능한 경우)
            if self.config.system.mixed_precision and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            # 소프트맥스 적용
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu()
    
    def predict_with_tta_streaming(self, dataloader: DataLoader, 
                                  tta_times: int = 5) -> Generator[Tuple[np.ndarray, List[str]], None, None]:
        """스트리밍 방식 TTA 예측 - 메모리 효율적"""
        
        if tta_times <= 1:
            # TTA 없이 일반 추론
            logger.info("⚡ 일반 추론 모드")
            
            for batch_images, batch_ids in tqdm(dataloader, desc="추론 진행"):
                probabilities = self.predict_batch(batch_images)
                yield probabilities.numpy(), list(batch_ids)
                
                # 배치 후 메모리 정리
                del probabilities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # TTA 추론
            logger.info(f"🔄 TTA 추론 모드 (변환: {tta_times}가지)")
            
            # TTA 변환들 가져오기
            tta_transforms = self.get_tta_transforms()
            selected_transforms = tta_transforms[:min(tta_times, len(tta_transforms))]
            
            self.inference_stats['tta_times_used'] = len(selected_transforms)
            
            # 원본 데이터셋의 transform 백업
            original_transform = dataloader.dataset.transform
            
            try:
                # 각 배치에 대해 TTA 수행
                for batch_idx, (_, batch_ids) in enumerate(tqdm(dataloader, desc="TTA 추론")):
                    
                    batch_tta_results = []
                    
                    # TTA 변환별로 예측
                    for tta_idx, tta_transform in enumerate(selected_transforms):
                        # 데이터셋의 transform 변경
                        dataloader.dataset.transform = tta_transform
                        
                        # 현재 배치의 이미지들을 다시 로드
                        batch_images = []
                        for item_idx in range(len(batch_ids)):
                            dataset_idx = batch_idx * dataloader.batch_size + item_idx
                            if dataset_idx < len(dataloader.dataset):
                                image, _ = dataloader.dataset[dataset_idx]
                                batch_images.append(image)
                        
                        if batch_images:
                            batch_tensor = torch.stack(batch_images)
                            probabilities = self.predict_batch(batch_tensor)
                            batch_tta_results.append(probabilities.numpy())
                            
                            # 메모리 정리
                            del batch_tensor, probabilities
                    
                    # TTA 결과 평균
                    if batch_tta_results:
                        avg_probabilities = np.mean(batch_tta_results, axis=0)
                        yield avg_probabilities, list(batch_ids)
                    
                    # 배치별 메모리 정리
                    del batch_tta_results
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 주기적 강제 메모리 정리
                    if batch_idx % 10 == 0:
                        clear_memory(verbose=False)
            
            finally:
                # 원본 transform 복원
                dataloader.dataset.transform = original_transform
    
    @safe_operation("배치 추론")
    def inference_batch(self) -> bool:
        """메인 배치 추론 함수"""
        
        # 테스트 데이터 확인
        if not os.path.exists(self.config.inference.test_csv):
            logger.error(f"❌ 테스트 CSV 파일 없음: {self.config.inference.test_csv}")
            return False
        
        logger.info(f"📊 테스트 데이터 로딩: {self.config.inference.test_csv}")
        
        # 테스트 데이터셋 생성
        with memory_monitor("테스트 데이터셋 생성"):
            try:
                test_transform = get_test_augmentations()
            except:
                # Fallback 변환
                test_transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            
            test_dataset = TestDataset(
                test_csv=self.config.inference.test_csv,
                transform=test_transform,
                cache_images=False  # 추론시에는 캐싱 비활성화
            )
            
            logger.info(f"✅ 테스트 데이터: {len(test_dataset):,}장")
        
        # 배치 크기 자동 조정
        available_memory_gb = get_memory_usage().get('ram_used', 4000) / 1024
        if available_memory_gb > self.config.inference.max_memory_gb:
            adjusted_batch_size = max(16, self.config.inference.batch_size // 2)
            logger.warning(f"⚠️  메모리 부족으로 배치 크기 조정: {self.config.inference.batch_size} → {adjusted_batch_size}")
            self.config.inference.batch_size = adjusted_batch_size
        
        # 데이터로더 생성
        test_loader = get_dataloader(
            test_dataset,
            batch_size=self.config.inference.batch_size,
            shuffle=False,
            num_workers=self.config.inference.num_workers,
            memory_efficient=True,
            memory_limit_mb=2048
        )
        
        logger.info(f"📦 데이터로더 준비:")
        logger.info(f"   배치 크기: {self.config.inference.batch_size}")
        logger.info(f"   총 배치 수: {len(test_loader)}")
        logger.info(f"   TTA: {'Yes' if self.config.inference.use_tta else 'No'}")
        
        # 결과 저장을 위한 준비
        all_predictions = []
        all_ids = []
        
        # 추론 시작
        start_time = time.time()
        
        try:
            tta_times = self.config.inference.tta_times if self.config.inference.use_tta else 1
            
            with memory_monitor("전체 추론"):
                for batch_predictions, batch_ids in self.predict_with_tta_streaming(test_loader, tta_times):
                    all_predictions.append(batch_predictions)
                    all_ids.extend(batch_ids)
                    
                    # 진행 상황 로깅
                    processed = len(all_ids)
                    if processed % (self.config.inference.batch_size * 10) == 0:
                        logger.info(f"   진행: {processed:,}/{len(test_dataset):,} ({processed/len(test_dataset)*100:.1f}%)")
            
            # 결과 결합
            logger.info("📊 결과 처리 중...")
            
            if all_predictions:
                final_predictions = np.vstack(all_predictions)
                
                # 결과 검증
                if len(all_ids) != len(final_predictions):
                    logger.error(f"❌ 예측 결과 불일치: IDs {len(all_ids)} vs Predictions {len(final_predictions)}")
                    return False
                
                # 확률 합 검증
                prob_sums = np.sum(final_predictions, axis=1)
                if not np.allclose(prob_sums, 1.0, atol=1e-3):
                    logger.warning("⚠️  일부 예측의 확률 합이 1.0이 아닙니다")
                    # 정규화
                    final_predictions = final_predictions / prob_sums[:, np.newaxis]
                    logger.info("   확률 정규화 완료")
                
                # 제출 파일 생성
                submission = pd.DataFrame(final_predictions, columns=self.class_names)
                submission.insert(0, 'ID', all_ids)
                
                # 출력 디렉토리 생성 및 저장
                os.makedirs(os.path.dirname(self.config.inference.output_path), exist_ok=True)
                submission.to_csv(self.config.inference.output_path, index=False)
                
                # 통계 업데이트
                processing_time = time.time() - start_time
                self.inference_stats.update({
                    'total_images': len(all_ids),
                    'processing_time': processing_time,
                    'avg_time_per_image': processing_time / len(all_ids),
                    'memory_peak': get_memory_usage().get('ram_used', 0)
                })
                
                # 결과 출력
                logger.info(f"✅ 추론 완료!")
                logger.info(f"   📊 처리된 이미지: {len(all_ids):,}장")
                logger.info(f"   📊 예측 shape: {final_predictions.shape}")
                logger.info(f"   💾 저장 위치: {self.config.inference.output_path}")
                logger.info(f"   ⏱️  총 시간: {processing_time:.1f}초")
                logger.info(f"   ⚡ 이미지당 평균: {self.inference_stats['avg_time_per_image']*1000:.1f}ms")
                
                if self.config.inference.use_tta:
                    logger.info(f"   🔄 TTA 변환: {self.inference_stats['tta_times_used']}가지")
                
                # 샘플 예측 출력 (상위 3개 클래스)
                self._show_sample_predictions(final_predictions, all_ids)
                
                return True
            else:
                logger.error("❌ 예측 결과가 없습니다")
                return False
        
        except Exception as e:
            logger.error(f"❌ 추론 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        finally:
            # 메모리 정리
            clear_memory(verbose=True)
    
    def _show_sample_predictions(self, predictions: np.ndarray, ids: List[str], num_samples: int = 5):
        """샘플 예측 결과 출력"""
        logger.info(f"\n📋 샘플 예측 결과 (상위 {num_samples}개):")
        logger.info("-" * 60)
        
        for i in range(min(num_samples, len(predictions))):
            sample_pred = predictions[i]
            sample_id = ids[i]
            
            # 상위 3개 클래스
            top3_indices = np.argsort(sample_pred)[-3:][::-1]
            top3_probs = sample_pred[top3_indices]
            top3_classes = [self.class_names[idx] for idx in top3_indices]
            
            logger.info(f"ID: {sample_id}")
            for j, (cls, prob) in enumerate(zip(top3_classes, top3_probs)):
                logger.info(f"   {j+1}. {cls}: {prob:.4f}")
            logger.info("")
    
    @safe_operation("모델 최적화")
    def optimize_model_for_inference(self):
        """추론용 모델 최적화"""
        if self.model is None:
            return
        
        try:
            # TorchScript 변환 시도
            logger.info("⚡ 모델 최적화 시도 중...")
            
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                # 트레이스 가능성 확인
                traced_model = torch.jit.trace(self.model, dummy_input)
                
                # 최적화 적용
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                
                # 테스트 실행
                test_output = optimized_model(dummy_input)
                
                # 성공시 모델 교체
                self.model = optimized_model
                logger.info("✅ 모델 최적화 완료 (TorchScript)")
                
        except Exception as e:
            logger.warning(f"⚠️  모델 최적화 실패: {str(e)}")
            logger.info("   원본 모델 사용")


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
        
        # 추론 관련 설정 검증
        if not config.inference.model_path:
            logger.error("❌ 모델 경로가 설정되지 않았습니다")
            logger.info("   config.inference.model_path를 설정하세요")
            return
        
        if not config.inference.test_csv:
            logger.error("❌ 테스트 CSV 경로가 설정되지 않았습니다")
            logger.info("   config.inference.test_csv를 설정하세요")
            return
        
        logger.info("🔮 추론 설정:")
        logger.info(f"   모델: {config.inference.model_path}")
        logger.info(f"   테스트 데이터: {config.inference.test_csv}")
        logger.info(f"   출력: {config.inference.output_path}")
        logger.info(f"   배치 크기: {config.inference.batch_size}")
        logger.info(f"   TTA: {config.inference.use_tta}")
        
        if not config.validate():
            logger.error("❌ 설정 검증 실패")
            return
        
    except Exception as e:
        logger.error(f"❌ 설정 로드 실패: {str(e)}")
        return
    
    # 추론 시작
    try:
        inference_manager = InferenceManager(config)
        
        # 모델 로드
        if not inference_manager.load_model():
            logger.error("❌ 모델 로드 실패")
            return
        
        # 모델 최적화 (선택적)
        if getattr(config.inference, 'optimize_model', False):
            inference_manager.optimize_model_for_inference()
        
        # 추론 실행
        success = inference_manager.inference_batch()
        
        if success:
            logger.info("🎉 추론 성공적으로 완료!")
            logger.info(f"💡 다음 단계: 앙상블 적용 (ensemble_v2.py)")
        else:
            logger.error("❌ 추론 실패")
            
    except KeyboardInterrupt:
        logger.info("⏹️  사용자에 의해 추론 중단")
        
    except Exception as e:
        logger.error(f"❌ 추론 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
    
    finally:
        # 최종 정리
        clear_memory(verbose=True)
        logger.info("🧹 추론 세션 정리 완료")


if __name__ == "__main__":
    main()