#!/usr/bin/env python3
"""
개선된 데이터 유틸리티
UTF-8 인코딩, 경로 정규화, 메모리 효율적 데이터셋 포함
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from sklearn.model_selection import StratifiedKFold

class EfficientCarDataset(Dataset):
    """메모리 효율적인 자동차 데이터셋"""
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform=None,
                 cache_size: int = 1000):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트
            labels: 라벨 리스트
            transform: 이미지 변환 함수
            cache_size: 캐시할 이미지 수 (메모리 사용량 조절)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache_size = cache_size
        self.cache = {}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 캐시에서 이미지 확인
        if idx in self.cache:
            image = self.cache[idx]
        else:
            # 이미지 로드
            try:
                image = Image.open(image_path).convert('RGB')
                
                # 캐시 크기 관리
                if len(self.cache) < self.cache_size:
                    self.cache[idx] = image
                    
            except Exception as e:
                print(f"이미지 로드 실패: {image_path}, 에러: {e}")
                # 기본 이미지 생성 (검은색 224x224)
                image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data_with_encoding(csv_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """UTF-8 인코딩으로 CSV 파일 로드"""
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"✅ CSV 로드 성공: {csv_path} (인코딩: {encoding})")
        return df
    except UnicodeDecodeError:
        # UTF-8 실패 시 다른 인코딩 시도
        encodings = ['cp949', 'euc-kr', 'latin-1']
        for enc in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                print(f"✅ CSV 로드 성공: {csv_path} (인코딩: {enc})")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"CSV 파일을 읽을 수 없습니다: {csv_path}")

def normalize_paths(base_path: str, relative_paths: List[str]) -> List[str]:
    """경로 정규화"""
    base_path = Path(base_path).resolve()
    normalized_paths = []
    
    for rel_path in relative_paths:
        # 상대 경로를 절대 경로로 변환
        full_path = base_path / rel_path
        normalized_paths.append(str(full_path.resolve()))
    
    return normalized_paths

def apply_class_mapping(labels: List[str], class_mapping: Dict[str, int]) -> List[int]:
    """클래스 이름을 인덱스로 변환"""
    mapped_labels = []
    missing_classes = set()
    
    for label in labels:
        if label in class_mapping:
            mapped_labels.append(class_mapping[label])
        else:
            missing_classes.add(label)
            # 기본값으로 0 할당 (또는 에러 발생)
            mapped_labels.append(0)
    
    if missing_classes:
        print(f"⚠️ 매핑되지 않은 클래스들: {missing_classes}")
    
    return mapped_labels

def create_stratified_folds(labels: List[int], n_splits: int = 5, random_state: int = 42) -> List[Tuple[List[int], List[int]]]:
    """Stratified K-Fold 생성"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(range(len(labels)), labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    
    print(f"✅ {n_splits}-Fold 교차검증 생성 완료")
    return folds

def validate_dataset(image_paths: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    """데이터셋 유효성 검사 및 정리"""
    valid_paths = []
    valid_labels = []
    invalid_count = 0
    
    for path, label in zip(image_paths, labels):
        if os.path.exists(path):
            try:
                # 이미지 파일 유효성 검사
                with Image.open(path) as img:
                    img.verify()
                valid_paths.append(path)
                valid_labels.append(label)
            except Exception as e:
                print(f"⚠️ 유효하지 않은 이미지: {path}, 에러: {e}")
                invalid_count += 1
        else:
            print(f"⚠️ 파일이 존재하지 않음: {path}")
            invalid_count += 1
    
    print(f"✅ 데이터셋 검증 완료: 유효 {len(valid_paths)}개, 무효 {invalid_count}개")
    return valid_paths, valid_labels

def get_class_distribution(labels: List[int], class_names: Optional[List[str]] = None) -> Dict:
    """클래스 분포 분석"""
    unique, counts = np.unique(labels, return_counts=True)
    
    distribution = {}
    for class_idx, count in zip(unique, counts):
        class_name = class_names[class_idx] if class_names else f"Class_{class_idx}"
        distribution[class_name] = count
    
    # 통계 정보
    stats = {
        'total_samples': len(labels),
        'num_classes': len(unique),
        'min_samples': int(np.min(counts)),
        'max_samples': int(np.max(counts)),
        'mean_samples': float(np.mean(counts)),
        'std_samples': float(np.std(counts))
    }
    
    return {
        'distribution': distribution,
        'statistics': stats
    }

def save_data_info(data_info: Dict, save_path: str):
    """데이터 정보 저장"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, ensure_ascii=False, indent=2)
    print(f"✅ 데이터 정보 저장: {save_path}")

def load_and_prepare_data(train_csv_path: str, 
                         test_csv_path: str,
                         class_mapping_path: str,
                         data_root: str) -> Dict:
    """전체 데이터 로드 및 준비"""
    
    print("🔄 데이터 로드 및 준비 시작...")
    
    # 1. CSV 파일 로드
    train_df = load_data_with_encoding(train_csv_path)
    test_df = load_data_with_encoding(test_csv_path)
    
    # 2. 클래스 매핑 로드
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    
    # 3. 경로 정규화
    train_paths = normalize_paths(data_root, train_df['img_path'].tolist())
    test_paths = normalize_paths(data_root, test_df['img_path'].tolist())
    
    # 4. 라벨 매핑
    train_labels = apply_class_mapping(train_df['label'].tolist(), class_mapping)
    
    # 5. 데이터셋 유효성 검사
    train_paths, train_labels = validate_dataset(train_paths, train_labels)
    test_paths, _ = validate_dataset(test_paths, [0] * len(test_paths))
    
    # 6. 클래스 분포 분석
    class_names = list(class_mapping.keys())
    class_dist = get_class_distribution(train_labels, class_names)
    
    # 7. K-Fold 생성
    folds = create_stratified_folds(train_labels)
    
    # 8. 결과 정리
    data_info = {
        'train_size': len(train_paths),
        'test_size': len(test_paths),
        'num_classes': len(class_mapping),
        'class_distribution': class_dist,
        'folds': folds
    }
    
    print(f"✅ 데이터 준비 완료!")
    print(f"   - 훈련 데이터: {len(train_paths)}개")
    print(f"   - 테스트 데이터: {len(test_paths)}개") 
    print(f"   - 클래스 수: {len(class_mapping)}개")
    
    return {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'test_paths': test_paths,
        'class_mapping': class_mapping,
        'data_info': data_info
    } 