#!/usr/bin/env python3
"""
앙상블 추론 스크립트
여러 모델의 예측을 결합하여 최종 예측을 생성합니다.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.model import CarClassificationModel
from src.data.data import CarDataset
from src.utils.utils import load_config
from torch.utils.data import DataLoader
import torch.nn.functional as F

def load_model(checkpoint_path, config):
    """체크포인트에서 모델을 로드합니다."""
    model = CarClassificationModel(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def ensemble_predict(models, dataloader, device):
    """여러 모델의 예측을 앙상블합니다."""
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_predictions = []
            
            # 각 모델의 예측을 수집
            for model in models:
                model = model.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                batch_predictions.append(probs.cpu().numpy())
            
            # 평균 앙상블
            ensemble_pred = np.mean(batch_predictions, axis=0)
            all_predictions.append(ensemble_pred)
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx}/{len(dataloader)} batches")
    
    return np.vstack(all_predictions)

def main():
    # 설정 로드
    config_path = project_root / "config" / "default.yaml"
    config = load_config(str(config_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 체크포인트 경로들
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_paths = [
        checkpoint_dir / "best_model_fold0.pth",
        # 추가 체크포인트들을 여기에 추가
    ]
    
    # 모델들 로드
    models = []
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            print(f"Loading model from {checkpoint_path}")
            model = load_model(checkpoint_path, config)
            models.append(model)
        else:
            print(f"Warning: {checkpoint_path} not found")
    
    if not models:
        print("No models loaded. Exiting.")
        return
    
    print(f"Loaded {len(models)} models for ensemble")
    
    # 테스트 데이터셋 생성
    test_dataset = CarDataset(
        data_dir=str(project_root / "data" / "test"),
        csv_file=str(project_root / "data" / "test.csv"),
        transform=config['test_transform'],
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # 앙상블 예측 수행
    print("Starting ensemble inference...")
    predictions = ensemble_predict(models, test_loader, device)
    
    # 결과 저장
    test_df = pd.read_csv(project_root / "data" / "test.csv")
    
    # 클래스 매핑 로드
    class_to_idx_path = checkpoint_dir / "class_to_idx.json"
    if class_to_idx_path.exists():
        import json
        with open(class_to_idx_path, 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = [f"class_{i}" for i in range(predictions.shape[1])]
    
    # 제출 파일 생성
    submission_df = pd.DataFrame(predictions, columns=class_names)
    submission_df.insert(0, 'ID', test_df['ID'])
    
    output_path = project_root / "submission_ensemble.csv"
    submission_df.to_csv(output_path, index=False)
    
    print(f"Ensemble predictions saved to {output_path}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Max prediction confidence: {predictions.max():.4f}")
    print(f"Mean prediction confidence: {predictions.max(axis=1).mean():.4f}")

if __name__ == "__main__":
    main() 