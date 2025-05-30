#!/usr/bin/env python3
"""
앙상블 훈련 스크립트
여러 모델을 훈련하고 앙상블 예측을 생성합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def ensemble_predictions(pred_dir, output_path):
    """여러 예측 파일을 앙상블합니다."""
    
    # 예측 파일들 수집
    pred_files = glob.glob(os.path.join(pred_dir, '*.csv'))
    
    if len(pred_files) == 0:
        print(f"❌ 예측 파일을 찾을 수 없습니다: {pred_dir}")
        return False
    
    print(f"📊 발견된 예측 파일: {len(pred_files)}개")
    for file in pred_files:
        print(f"   - {os.path.basename(file)}")
    
    # 모든 예측 파일 로드
    predictions = []
    ids = None
    
    for file_path in pred_files:
        try:
            df = pd.read_csv(file_path)
            
            # ID 컬럼 확인
            if ids is None:
                ids = df['ID']
            
            # 예측 확률 추출 (ID 컬럼 제외)
            pred_values = df.drop('ID', axis=1).values
            predictions.append(pred_values)
            
            print(f"✅ 로드 완료: {os.path.basename(file_path)} - Shape: {pred_values.shape}")
            
        except Exception as e:
            print(f"❌ 파일 로드 실패: {file_path} - {str(e)}")
            continue
    
    if len(predictions) == 0:
        print("❌ 유효한 예측 파일이 없습니다")
        return False
    
    # 앙상블 (평균)
    print("🔄 앙상블 수행 중...")
    ensemble_pred = np.mean(predictions, axis=0)
    
    # 클래스명 가져오기
    class_names = pd.read_csv(pred_files[0]).columns.tolist()[1:]
    
    # 제출 파일 생성
    submission = pd.DataFrame(ensemble_pred, columns=class_names)
    submission.insert(0, 'ID', ids)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 저장
    submission.to_csv(output_path, index=False)
    
    print(f"✅ 앙상블 완료!")
    print(f"   📊 앙상블된 모델 수: {len(predictions)}")
    print(f"   📊 예측 shape: {ensemble_pred.shape}")
    print(f"   💾 저장 위치: {output_path}")
    
    # 통계 출력
    max_probs = ensemble_pred.max(axis=1)
    print(f"   📈 최대 확률 평균: {max_probs.mean():.4f}")
    print(f"   📈 높은 신뢰도(>0.8) 예측: {(max_probs > 0.8).sum()}개 ({(max_probs > 0.8).mean()*100:.1f}%)")
    
    return True

def main():
    """메인 함수"""
    
    # 기본 설정
    pred_dir = "."  # 현재 디렉토리에서 예측 파일 찾기
    output_path = "submission_ensemble.csv"
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        pred_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print("🚀 앙상블 훈련 시작!")
    print(f"   📂 예측 파일 디렉토리: {pred_dir}")
    print(f"   📄 출력 파일: {output_path}")
    
    # 앙상블 수행
    success = ensemble_predictions(pred_dir, output_path)
    
    if success:
        print("🎉 앙상블 성공!")
    else:
        print("❌ 앙상블 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()
