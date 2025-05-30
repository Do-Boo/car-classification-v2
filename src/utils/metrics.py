import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def calculate_accuracy(outputs, targets):
    """정확도 계산"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # 예측값 계산
    predictions = np.argmax(outputs, axis=1)
    
    return accuracy_score(targets, predictions)

def top1_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def topk_accuracy(y_true, y_pred, k=5):
    topk = np.argsort(y_pred, axis=1)[:, -k:]
    return np.mean([y in topk_row for y, topk_row in zip(y_true, topk)])

def calculate_metrics(outputs, targets, class_names=None):
    """종합 메트릭 계산"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    predictions = np.argmax(outputs, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'classification_report': classification_report(targets, predictions, target_names=class_names, output_dict=True),
        'confusion_matrix': confusion_matrix(targets, predictions)
    }
    
    return metrics

def calculate_f1_score(outputs, targets, average='weighted'):
    """F1 스코어 계산"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    predictions = np.argmax(outputs, axis=1)
    
    return f1_score(targets, predictions, average=average)
