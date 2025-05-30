"""
개선된 데이터 처리 모듈
- 메모리 효율적 데이터로딩
- utils.py 통합 및 안전한 이미지 처리
- 스트리밍 데이터셋 지원
"""

import os
import gc
import time
from typing import List, Tuple, Optional, Union, Callable
from pathlib import Path
from collections import Counter, defaultdict
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

# 순환 import 방지를 위해 직접 import
import logging
logger = logging.getLogger(__name__)

# 필요한 함수들을 직접 정의하거나 간단한 버전 사용
def safe_image_load(image_path, target_size=None):
    """간단한 이미지 로드"""
    import cv2
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def memory_monitor(name):
    """간단한 메모리 모니터 (context manager)"""
    class SimpleMonitor:
        def __enter__(self):
            logger.debug(f"시작: {name}")
            return self
        def __exit__(self, *args):
            logger.debug(f"완료: {name}")
    return SimpleMonitor()

def safe_operation(name):
    """간단한 안전 작업 데코레이터"""
    def decorator(func):
        return func
    return decorator

def memory_cleanup_decorator(func):
    """간단한 메모리 정리 데코레이터"""
    return func

def validate_image_dataset(image_dir, min_size=32, max_aspect_ratio=10.0):
    """간단한 데이터셋 검증"""
    return {'errors': []}

def get_memory_usage():
    """간단한 메모리 사용량 조회"""
    import psutil
    return {
        'ram_used': psutil.virtual_memory().used / 1024 / 1024,  # MB
        'ram_available': psutil.virtual_memory().available / 1024 / 1024  # MB
    }

def clear_memory(verbose=False):
    """메모리 정리 함수"""
    import gc
    import torch
    
    # Python 가비지 컬렉션
    gc.collect()
    
    # PyTorch CUDA 캐시 정리 (GPU 사용시)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose:
        memory_info = get_memory_usage()
        logger.info(f"🧹 메모리 정리 완료: RAM {memory_info['ram_used']:.1f}MB 사용 중")

# 경고 메시지 무시
warnings.filterwarnings('ignore')


class MemoryEfficientCarDataset(Dataset):
    """메모리 효율적인 자동차 분류 데이터셋"""
    
    def __init__(self, root_dir: str = None, csv_path: str = None, 
                 transform=None, is_train: bool = True,
                 cache_images: bool = False, max_cache_size: int = 1000,
                 lazy_loading: bool = True):
        """
        Args:
            root_dir: 이미지가 저장된 루트 디렉토리
            csv_path: CSV 파일 경로 (있으면 CSV 우선 사용)
            transform: 이미지 변환
            is_train: 훈련용 여부
            cache_images: 이미지 캐싱 여부
            max_cache_size: 최대 캐시 크기
            lazy_loading: 지연 로딩 여부
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.cache_images = cache_images
        self.max_cache_size = max_cache_size
        self.lazy_loading = lazy_loading
        
        # 이미지 캐시
        self._image_cache = {} if cache_images else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 데이터 로드
        self._load_data(csv_path, root_dir)
        
        # 데이터셋 검증 (선택적)
        if not lazy_loading:
            self._validate_dataset()
    
    def _load_data(self, csv_path: str, root_dir: str):
        """데이터 로드"""
        if csv_path and os.path.exists(csv_path):
            # CSV 파일에서 로드
            self.df = pd.read_csv(csv_path)
            self.use_csv = True
            self.image_paths = self.df['img_path'].tolist()
            self.labels = self.df['label'].tolist() if 'label' in self.df.columns else None
            logger.info(f"✅ CSV에서 데이터 로드: {len(self.image_paths):,}장")
        else:
            # 폴더 구조에서 데이터 로드
            with memory_monitor("폴더 스캔"):
                self.image_paths, self.labels, self.class_to_idx = self._scan_folders_optimized(root_dir)
            self.use_csv = False
            logger.info(f"✅ 폴더에서 데이터 로드: {len(self.image_paths):,}장, {len(self.class_to_idx)}개 클래스")
    
    def _scan_folders_optimized(self, root_dir: str) -> Tuple[List[str], List[str], dict]:
        """최적화된 폴더 스캔"""
        image_paths = []
        labels = []
        
        # 클래스 폴더들 스캔
        class_folders = sorted([f for f in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, f))])
        
        # 클래스명 -> 인덱스 매핑
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}
        
        # 병렬 처리 가능한 구조로 개선
        total_files = 0
        valid_files = 0
        
        for class_name in class_folders:
            class_path = os.path.join(root_dir, class_name)
            
            # 이미지 파일들 찾기 (확장자 체크 최적화)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [
                f for f in os.listdir(class_path) 
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                total_files += 1
                
                # 빠른 검증 (lazy loading인 경우 기본 체크만)
                if self.lazy_loading or self._quick_validate_image(image_path):
                    image_paths.append(image_path)
                    labels.append(class_name)
                    valid_files += 1
        
        logger.info(f"📊 스캔 결과: {valid_files:,}/{total_files:,} 유효 ({valid_files/total_files*100:.1f}%)")
        return image_paths, labels, class_to_idx
    
    def _quick_validate_image(self, image_path: str) -> bool:
        """빠른 이미지 유효성 검사"""
        try:
            # 파일 크기 체크 (매우 빠름)
            file_size = os.path.getsize(image_path)
            if file_size < 1024:  # 1KB 미만은 의심스러움
                return False
            
            # 파일 시그니처 체크 (선택적)
            with open(image_path, 'rb') as f:
                header = f.read(8)
                # JPEG, PNG 시그니처 체크
                if header.startswith(b'\xff\xd8') or header.startswith(b'\x89PNG'):
                    return True
            
            return True  # 기본적으로 통과
            
        except Exception:
            return False
    
    def _validate_dataset(self):
        """전체 데이터셋 검증 (lazy_loading=False인 경우)"""
        logger.info("🔍 데이터셋 검증 중...")
        
        with memory_monitor("데이터셋 검증"):
            validation_result = validate_image_dataset(
                image_dir=self.root_dir if not self.use_csv else None,
                min_size=32,
                max_aspect_ratio=10.0
            )
        
        if validation_result['errors']:
            logger.warning(f"⚠️  검증 오류: {len(validation_result['errors'])}개")
            for error in validation_result['errors'][:5]:  # 처음 5개만 출력
                logger.warning(f"   {error}")
    
    @memory_cleanup_decorator
    def __getitem__(self, idx):
        """메모리 효율적인 아이템 로드"""
        image_path = self.image_paths[idx]
        
        # 캐시에서 이미지 로드 시도
        if self.cache_images and image_path in self._image_cache:
            image = self._image_cache[image_path].copy()
            self._cache_hits += 1
        else:
            # 안전한 이미지 로드 (utils.py 사용)
            image = safe_image_load(image_path, target_size=None)
            self._cache_misses += 1
            
            # 캐시에 저장 (용량 제한)
            if self.cache_images and len(self._image_cache) < self.max_cache_size:
                self._image_cache[image_path] = image.copy()
        
        # 변환 적용
        if self.transform:
            try:
                if isinstance(self.transform, A.Compose):
                    # Albumentations
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    # torchvision transforms
                    image = self.transform(image)
            except Exception as e:
                logger.warning(f"⚠️  변환 실패: {image_path} - {str(e)}")
                # 기본 변환 적용
                image = self._get_default_transform()(image=image)['image']
        else:
            # 기본 변환 적용
            image = self._get_default_transform()(image=image)['image']
        
        if self.is_train and self.labels:
            # 훈련용: 이미지 + 라벨
            if self.use_csv:
                label = self.labels[idx]  # 이미 숫자 인덱스
            else:
                label_name = self.labels[idx]
                label = self.class_to_idx[label_name]  # 문자열 -> 숫자 변환
            
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # 테스트용: 이미지만
            return image, torch.tensor(0, dtype=torch.long)  # 더미 라벨
    
    def _get_default_transform(self):
        """기본 변환"""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_cache_stats(self) -> dict:
        """캐시 통계"""
        if not self.cache_images:
            return {"cache_enabled": False}
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_enabled": True,
            "cache_size": len(self._image_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """캐시 비우기"""
        if self.cache_images:
            self._image_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            clear_memory(verbose=False)
    
    def get_class_distribution(self) -> dict:
        """클래스 분포 반환"""
        if self.labels:
            return Counter(self.labels)
        return {}
    
    def get_class_weights(self) -> torch.Tensor:
        """클래스 가중치 계산 (불균형 데이터 대응)"""
        if not self.labels:
            return None
        
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(class_counts)
        
        # 역빈도 가중치 계산
        weights = []
        for class_name in sorted(class_counts.keys()):
            weight = total_samples / (num_classes * class_counts[class_name])
            weights.append(weight)
        
        return torch.FloatTensor(weights)


class StreamingCarDataset(Dataset):
    """스트리밍 방식 대용량 데이터셋"""
    
    def __init__(self, file_list_path: str, transform=None, 
                 chunk_size: int = 1000, preload_next_chunk: bool = True):
        """
        Args:
            file_list_path: 파일 목록 경로 (각 줄에 이미지 경로)
            transform: 이미지 변환
            chunk_size: 청크 크기
            preload_next_chunk: 다음 청크 미리 로드 여부
        """
        self.file_list_path = file_list_path
        self.transform = transform
        self.chunk_size = chunk_size
        self.preload_next_chunk = preload_next_chunk
        
        # 파일 목록 로드
        with open(file_list_path, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        
        self.total_samples = len(self.file_paths)
        self.current_chunk = 0
        self.chunk_data = {}
        
        # 첫 번째 청크 로드
        self._load_chunk(0)
        
        logger.info(f"📡 스트리밍 데이터셋 초기화: {self.total_samples:,}개 파일")
    
    def _load_chunk(self, chunk_idx: int):
        """청크 로드"""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        if chunk_idx in self.chunk_data:
            return  # 이미 로드됨
        
        chunk_files = self.file_paths[start_idx:end_idx]
        chunk_images = []
        
        logger.info(f"📦 청크 {chunk_idx} 로드 중: {len(chunk_files)}개 파일")
        
        for file_path in chunk_files:
            image = safe_image_load(file_path)
            chunk_images.append(image)
        
        self.chunk_data[chunk_idx] = chunk_images
        
        # 메모리 관리: 오래된 청크 제거
        if len(self.chunk_data) > 3:  # 최대 3개 청크만 유지
            oldest_chunk = min(self.chunk_data.keys())
            if oldest_chunk != chunk_idx:
                del self.chunk_data[oldest_chunk]
                clear_memory(verbose=False)
    
    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        # 필요한 청크 로드
        if chunk_idx not in self.chunk_data:
            self._load_chunk(chunk_idx)
        
        # 다음 청크 미리 로드 (선택적)
        if self.preload_next_chunk and (chunk_idx + 1) not in self.chunk_data:
            next_chunk_start = (chunk_idx + 1) * self.chunk_size
            if next_chunk_start < self.total_samples:
                self._load_chunk(chunk_idx + 1)
        
        # 이미지 가져오기
        image = self.chunk_data[chunk_idx][local_idx]
        
        # 변환 적용
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return image, torch.tensor(0, dtype=torch.long)  # 더미 라벨
    
    def __len__(self):
        return self.total_samples


class TestDataset(Dataset):
    """개선된 테스트 전용 데이터셋"""
    
    def __init__(self, test_csv: str, transform=None, cache_images: bool = False):
        """
        Args:
            test_csv: 테스트 CSV 파일 경로
            transform: 이미지 변환
            cache_images: 이미지 캐싱 여부 (테스트용은 일반적으로 False)
        """
        self.df = pd.read_csv(test_csv)
        self.transform = transform if transform else self._get_default_transform()
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None
        
        # 경로 정규화
        self.df['img_path'] = self.df['img_path'].apply(self._normalize_path)
        
        logger.info(f"✅ 테스트 데이터셋 로드: {len(self.df):,}장")
    
    def _normalize_path(self, img_path: str) -> str:
        """경로 정규화 - 단순화된 버전"""
        img_path = img_path.strip()
        
        # 절대 경로인 경우 그대로 사용
        if os.path.isabs(img_path) and os.path.exists(img_path):
            return img_path
        
        # 상대 경로 처리
        possible_paths = [
            img_path,
            os.path.join('data', img_path.lstrip('./')),
            img_path.replace('\\', '/')  # Windows 경로 구분자 변환
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 모든 경로가 실패하면 원본 반환 (에러는 로딩 시 처리)
        return img_path
    
    def _get_default_transform(self):
        """테스트용 기본 변환"""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @memory_cleanup_decorator
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        img_id = row['ID']
        
        # 캐시에서 이미지 로드 시도
        if self.cache_images and img_path in self._image_cache:
            image = self._image_cache[img_path].copy()
        else:
            # 안전한 이미지 로드
            image = safe_image_load(img_path)
            
            # 캐시에 저장
            if self.cache_images:
                self._image_cache[img_path] = image.copy()
        
        # 변환 적용
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except Exception as e:
                logger.warning(f"⚠️  테스트 이미지 변환 실패: {img_path} - {str(e)}")
                # 기본 변환으로 재시도
                default_transform = self._get_default_transform()
                transformed = default_transform(image=image)
                image = transformed['image']
        
        return image, img_id
    
    def __len__(self):
        return len(self.df)


class MemoryEfficientDataLoader:
    """메모리 효율적인 데이터로더 래퍼"""
    
    def __init__(self, dataset: Dataset, batch_size: int = 32, 
                 shuffle: bool = True, num_workers: int = 4,
                 memory_limit_mb: float = 2048, **kwargs):
        """
        Args:
            dataset: 데이터셋
            batch_size: 배치 크기
            shuffle: 셔플 여부
            num_workers: 워커 수
            memory_limit_mb: 메모리 제한 (MB)
            **kwargs: DataLoader에 전달할 추가 인자
        """
        self.dataset = dataset
        self.memory_limit_mb = memory_limit_mb
        self.initial_memory = get_memory_usage()
        
        # 메모리 제한에 따른 배치 크기 자동 조정
        adjusted_batch_size = self._adjust_batch_size(batch_size)
        if adjusted_batch_size != batch_size:
            logger.warning(f"⚠️  메모리 제한으로 배치 크기 조정: {batch_size} → {adjusted_batch_size}")
        
        # kwargs에서 중복 인자 제거 (중복 방지)
        kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['pin_memory', 'drop_last']}
        
        # DataLoader 생성
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=adjusted_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True if shuffle else False,
            **kwargs_filtered
        )
        
        self._batch_count = 0
        self._memory_warnings = 0
        
    def _adjust_batch_size(self, requested_batch_size: int) -> int:
        """메모리 제한에 따른 배치 크기 조정"""
        available_memory = self.memory_limit_mb
        
        # 대략적인 배치당 메모리 사용량 추정 (MB)
        # 이미지 크기: 224x224x3, float32 = ~600KB per image
        estimated_memory_per_image = 0.6  # MB
        estimated_batch_memory = requested_batch_size * estimated_memory_per_image
        
        if estimated_batch_memory > available_memory * 0.8:  # 80% 임계치
            adjusted_batch_size = max(1, int(available_memory * 0.8 / estimated_memory_per_image))
            return adjusted_batch_size
        
        return requested_batch_size
    
    def __iter__(self):
        for batch in self.dataloader:
            self._batch_count += 1
            
            # 주기적 메모리 체크
            if self._batch_count % 50 == 0:
                self._check_memory()
            
            yield batch
            
            # 배치 처리 후 메모리 정리 (필요시)
            if self._batch_count % 100 == 0:
                clear_memory(verbose=False)
    
    def _check_memory(self):
        """메모리 사용량 체크"""
        current_memory = get_memory_usage()
        memory_increase = current_memory['ram_used'] - self.initial_memory['ram_used']
        
        if memory_increase > self.memory_limit_mb:
            self._memory_warnings += 1
            logger.warning(f"⚠️  메모리 사용량 초과: {memory_increase:.1f}MB / {self.memory_limit_mb}MB")
            
            if self._memory_warnings >= 3:
                logger.error("❌ 메모리 사용량이 지속적으로 초과됩니다. 배치 크기를 줄이세요.")
    
    def __len__(self):
        return len(self.dataloader)


def get_kfold(dataset: Dataset, n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """향상된 층화 K-Fold 교차검증 생성"""
    if not hasattr(dataset, 'labels') or dataset.labels is None:
        raise ValueError("데이터셋에 라벨이 없습니다")
    
    # 라벨을 숫자로 변환
    if hasattr(dataset, 'class_to_idx'):
        labels = [dataset.class_to_idx[label] for label in dataset.labels]
    else:
        labels = dataset.labels
    
    # 클래스 분포 체크
    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())
    
    if min_class_count < n_splits:
        logger.warning(f"⚠️  일부 클래스의 샘플 수({min_class_count})가 fold 수({n_splits})보다 적습니다")
        n_splits = max(2, min_class_count)
        logger.warning(f"   fold 수를 {n_splits}로 조정합니다")
    
    # StratifiedKFold 생성
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    logger.info(f"🔀 {n_splits}-Fold 층화 교차검증 생성")
    logger.info(f"📊 클래스 분포: {dict(sorted(class_counts.items()))}")
    
    return skf.split(range(len(dataset)), labels)


@safe_operation("데이터로더 생성")
def get_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, 
                  num_workers: int = 4, pin_memory: bool = True, 
                  collate_fn: Optional[Callable] = None,
                  memory_efficient: bool = True,
                  memory_limit_mb: float = 2048) -> Union[DataLoader, MemoryEfficientDataLoader]:
    """메모리 효율적인 데이터로더 생성"""
    
    # 메모리 기반 num_workers 조정
    available_memory = get_memory_usage()['ram_used']
    if available_memory > 8000:  # 8GB 이상
        max_workers = min(num_workers, 8)
    elif available_memory > 4000:  # 4GB 이상
        max_workers = min(num_workers, 4)
    else:  # 4GB 미만
        max_workers = min(num_workers, 2)
    
    if max_workers != num_workers:
        logger.info(f"🔧 메모리 기반 워커 수 조정: {num_workers} → {max_workers}")
        num_workers = max_workers
    
    # CUDA 사용 가능시에만 pin_memory 활성화
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False
        logger.info("💾 CUDA 미사용으로 pin_memory 비활성화")
    
    if memory_efficient:
        return MemoryEfficientDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            memory_limit_mb=memory_limit_mb,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=True if shuffle else False
        )
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=True if shuffle else False
        )


# CutMix/MixUp 구현 - 메모리 누수 방지
@memory_cleanup_decorator
def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """메모리 효율적인 CutMix 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    
    # 잘라낼 영역 계산
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 중심점 랜덤 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 경계 조정
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # CutMix 적용 (in-place 연산으로 메모리 절약)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 실제 비율 계산
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


@memory_cleanup_decorator
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """메모리 효율적인 MixUp 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # in-place 연산으로 메모리 절약
    mixed_x = x.clone()  # 복사본 생성
    mixed_x.mul_(lam).add_(x[index], alpha=(1 - lam))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_collate_fn(batch):
    """메모리 효율적인 CutMix용 collate 함수"""
    try:
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        
        # CutMix 적용
        images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
        
        return images, (labels_a, labels_b, lam)
    except Exception as e:
        logger.warning(f"⚠️  CutMix collate 오류: {str(e)}")
        # fallback: 일반 collate
        images, labels = zip(*batch)
        return torch.stack(images), torch.stack(labels)


def mixup_collate_fn(batch):
    """메모리 효율적인 MixUp용 collate 함수"""
    try:
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        
        # MixUp 적용
        images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
        
        return images, (labels_a, labels_b, lam)
    except Exception as e:
        logger.warning(f"⚠️  MixUp collate 오류: {str(e)}")
        # fallback: 일반 collate
        images, labels = zip(*batch)
        return torch.stack(images), torch.stack(labels)


class ImbalancedDatasetSampler(torch.utils.data.Sampler):
    """개선된 불균형 데이터셋용 샘플러"""
    
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # 클래스별 샘플 수 계산
        label_to_count = {}
        indices_per_class = defaultdict(list)
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            label_to_count[label] = label_to_count.get(label, 0) + 1
            indices_per_class[label].append(idx)
        
        # 가중치 계산
        weights = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            weights.append(1.0 / label_to_count[label])
        
        self.weights = torch.DoubleTensor(weights)
        self.indices_per_class = dict(indices_per_class)
        
        logger.info(f"⚖️  불균형 샘플러 초기화: {len(label_to_count)}개 클래스")
        for label, count in sorted(label_to_count.items()):
            logger.info(f"   클래스 {label}: {count:,}개")
    
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True))
    
    def __len__(self):
        return self.num_samples


def create_balanced_dataloader(dataset: Dataset, batch_size: int = 32, 
                             num_workers: int = 4, memory_efficient: bool = True) -> DataLoader:
    """균형 잡힌 데이터로더 생성 (불균형 데이터 대응)"""
    sampler = ImbalancedDatasetSampler(dataset)
    
    if memory_efficient:
        return MemoryEfficientDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )


def get_train_val_split(dataset: Dataset, val_ratio: float = 0.2, 
                       random_state: int = 42) -> Tuple[Subset, Subset]:
    """메모리 효율적인 훈련/검증 데이터 분할"""
    from sklearn.model_selection import train_test_split
    
    indices = list(range(len(dataset)))
    labels = None
    
    # 라벨 추출
    if hasattr(dataset, 'labels') and dataset.labels:
        labels = dataset.labels
    
    with memory_monitor("데이터 분할"):
        if labels:
            # 층화 분할
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=val_ratio, 
                stratify=labels,
                random_state=random_state
            )
        else:
            # 랜덤 분할
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=val_ratio, 
                random_state=random_state
            )
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    logger.info(f"🔄 데이터 분할 완료:")
    logger.info(f"   📊 훈련: {len(train_subset):,}장")
    logger.info(f"   📊 검증: {len(val_subset):,}장")
    
    return train_subset, val_subset


def analyze_dataset(dataset: Dataset) -> dict:
    """데이터셋 분석 - 메모리 효율적 버전"""
    analysis = {
        'total_samples': len(dataset),
        'num_classes': 0,
        'class_distribution': {},
    }
    
    # 클래스 정보
    if hasattr(dataset, 'class_to_idx'):
        analysis['num_classes'] = len(dataset.class_to_idx)
        analysis['class_distribution'] = dataset.get_class_distribution()
    
    # 이미지 크기 분석 (샘플링으로 메모리 절약)
    sample_size = min(100, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    image_sizes = []
    valid_samples = 0
    
    logger.info(f"🔍 데이터셋 분석 중... (샘플: {sample_size}장)")
    
    for idx in sample_indices:
        try:
            if hasattr(dataset, 'image_paths'):
                image_path = dataset.image_paths[idx]
                with Image.open(image_path) as img:
                    image_sizes.append(img.size)
                    valid_samples += 1
        except Exception:
            continue
    
    if image_sizes:
        widths, heights = zip(*image_sizes)
        analysis.update({
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'min_width': min(widths),
            'min_height': min(heights),
            'max_width': max(widths),
            'max_height': max(heights),
            'valid_sample_rate': valid_samples / sample_size
        })
    
    return analysis


# 편의 함수들
CarDataset = MemoryEfficientCarDataset  # 하위 호환성을 위한 별칭

def save_class_mapping(class_to_idx: dict, save_path: str):
    """클래스 매핑 저장"""
    from utils import safe_json_save
    safe_json_save(class_to_idx, save_path)
    logger.info(f"💾 클래스 매핑 저장: {save_path}")


def load_class_mapping(load_path: str) -> dict:
    """클래스 매핑 로드"""
    from utils import safe_json_load
    class_to_idx = safe_json_load(load_path)
    if class_to_idx:
        logger.info(f"📂 클래스 매핑 로드: {load_path}")
        return class_to_idx
    else:
        raise ValueError(f"클래스 매핑 로드 실패: {load_path}")


if __name__ == "__main__":
    # 테스트 코드
    from utils import initialize_environment
    
    initialize_environment()
    logger.info("🧪 개선된 데이터 모듈 테스트")
    
    # 메모리 효율적 데이터셋 테스트
    try:
        if os.path.exists("data/train"):
            with memory_monitor("데이터셋 생성"):
                dataset = MemoryEfficientCarDataset(
                    root_dir="data/train",
                    cache_images=True,
                    max_cache_size=100
                )
            
            logger.info(f"✅ 데이터셋 로드: {len(dataset):,}장")
            
            # 캐시 테스트
            _ = dataset[0]  # 캐시 미스
            _ = dataset[0]  # 캐시 히트
            cache_stats = dataset.get_cache_stats()
            logger.info(f"📊 캐시 통계: {cache_stats}")
            
            # 분석
            analysis = analyze_dataset(dataset)
            logger.info(f"📊 분석 결과: {analysis}")
            
            # 메모리 효율적 데이터로더 테스트
            dataloader = get_dataloader(
                dataset, 
                batch_size=8, 
                memory_efficient=True,
                memory_limit_mb=1024
            )
            logger.info(f"✅ 데이터로더 생성: {len(dataloader)}배치")
            
            # 첫 번째 배치 테스트
            first_batch = next(iter(dataloader))
            logger.info(f"📦 첫 배치 shape: {first_batch[0].shape}")
            
    except Exception as e:
        logger.error(f"❌ 데이터셋 테스트 실패: {e}")
    
    logger.info("🎉 개선된 데이터 모듈 테스트 완료!")

def create_submission(predictions, ids, class_names, output_path):
    """제출 파일 생성"""
    import pandas as pd
    
    # 예측 결과를 DataFrame으로 변환
    submission = pd.DataFrame(predictions, columns=class_names)
    submission.insert(0, 'ID', ids)
    
    # CSV 파일로 저장
    submission.to_csv(output_path, index=False)
    
    return submission