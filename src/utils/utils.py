"""
공통 유틸리티 함수 모듈
- 메모리 관리, 모델 로딩, 검증, 에러 복구 등
- V2 코드의 중복 제거 및 안정성 향상
"""

import os
import gc
import json
import time
import shutil
import logging
import psutil
import traceback
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import cv2


# ================================
# 로깅 설정
# ================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger("hecto_ai")
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (옵션)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 전역 로거
logger = setup_logging()


# ================================
# 메모리 관리
# ================================

def get_memory_usage() -> Dict[str, float]:
    """현재 메모리 사용량 조회 (MB 단위)"""
    process = psutil.Process()
    memory_info = {
        'ram_used': process.memory_info().rss / 1024 / 1024,
        'ram_percent': process.memory_percent()
    }
    
    # GPU 메모리 (CUDA 사용 가능한 경우)
    if torch.cuda.is_available():
        memory_info.update({
            'gpu_allocated': torch.cuda.memory_allocated() / 1024 / 1024,
            'gpu_reserved': torch.cuda.memory_reserved() / 1024 / 1024,
            'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024
        })
    
    return memory_info


def clear_memory(verbose: bool = True):
    """메모리 정리"""
    if verbose:
        before = get_memory_usage()
    
    # Python 가비지 수집
    gc.collect()
    
    # GPU 메모리 정리 (CUDA 사용 가능한 경우)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if verbose:
        after = get_memory_usage()
        ram_freed = before['ram_used'] - after['ram_used']
        logger.info(f"🧹 메모리 정리 완료: RAM {ram_freed:.1f}MB 해제")
        
        if torch.cuda.is_available():
            gpu_freed = before['gpu_allocated'] - after['gpu_allocated']
            logger.info(f"🎮 GPU 메모리: {gpu_freed:.1f}MB 해제")


@contextmanager
def memory_monitor(operation_name: str = "Operation"):
    """메모리 사용량 모니터링 컨텍스트 매니저"""
    before = get_memory_usage()
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        after = get_memory_usage()
        
        ram_used = after['ram_used'] - before['ram_used']
        duration = end_time - start_time
        
        logger.info(f"📊 {operation_name} 완료:")
        logger.info(f"   ⏱️  소요시간: {duration:.2f}초")
        logger.info(f"   🧠 RAM 사용: {ram_used:+.1f}MB")
        
        if torch.cuda.is_available():
            gpu_used = after['gpu_allocated'] - before['gpu_allocated']
            logger.info(f"   🎮 GPU 사용: {gpu_used:+.1f}MB")


def memory_cleanup_decorator(func):
    """메모리 정리 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            clear_memory(verbose=False)
    return wrapper


# ================================
# 안전한 모델 관리
# ================================

def safe_model_load(model_path: str, model_class=None, device: str = "auto", 
                   strict: bool = True, **model_kwargs) -> Optional[nn.Module]:
    """안전한 모델 로딩 with 에러 복구"""
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 모델 파일 없음: {model_path}")
        return None
    
    # 디바이스 자동 감지
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        logger.info(f"📦 모델 로딩 시작: {model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # state_dict 추출
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 모델 생성 (클래스가 제공된 경우)
        if model_class:
            model = model_class(**model_kwargs)
            
            # state_dict 로드
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"⚠️  누락된 키: {len(missing_keys)}개")
            if unexpected_keys:
                logger.warning(f"⚠️  예상치 못한 키: {len(unexpected_keys)}개")
            
            model = model.to(device)
            logger.info(f"✅ 모델 로딩 완료")
            return model
        else:
            # state_dict만 반환
            return state_dict
            
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def safe_model_save(model: nn.Module, save_path: str, 
                   additional_info: Optional[Dict] = None,
                   create_backup: bool = True) -> bool:
    """안전한 모델 저장"""
    
    try:
        # 저장 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 백업 생성 (기존 파일이 있는 경우)
        if create_backup and os.path.exists(save_path):
            backup_path = save_path + ".backup"
            shutil.copy2(save_path, backup_path)
            logger.info(f"💾 백업 생성: {backup_path}")
        
        # 저장할 정보 구성
        save_dict = {
            'model': model.state_dict(),
            'model_class': model.__class__.__name__,
            'save_time': time.time()
        }
        
        # 추가 정보 포함
        if additional_info:
            save_dict.update(additional_info)
        
        # 실제 저장
        torch.save(save_dict, save_path)
        logger.info(f"✅ 모델 저장 완료: {save_path}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(save_path) / 1024 / 1024
        logger.info(f"📁 파일 크기: {file_size:.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 저장 실패: {str(e)}")
        
        # 백업에서 복구 시도
        if create_backup:
            backup_path = save_path + ".backup"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, save_path)
                logger.info(f"🔄 백업에서 복구 완료")
        
        return False


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3):
    """오래된 체크포인트 정리"""
    
    if not os.path.exists(checkpoint_dir):
        return
    
    # 체크포인트 파일들 찾기
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith(('.pth', '.ckpt', '.pt')):
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            checkpoint_files.append((file_path, mtime))
    
    # 수정 시간 기준 정렬 (최신순)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # 오래된 파일들 삭제
    files_to_delete = checkpoint_files[keep_last_n:]
    for file_path, _ in files_to_delete:
        try:
            os.remove(file_path)
            logger.info(f"🗑️  오래된 체크포인트 삭제: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"⚠️  파일 삭제 실패: {file_path} - {str(e)}")


# ================================
# 파일 시스템 유틸리티
# ================================

def safe_file_read(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """안전한 파일 읽기"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ 파일 읽기 실패: {file_path} - {str(e)}")
        return None


def safe_json_load(json_path: str) -> Optional[Dict]:
    """안전한 JSON 로드"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ JSON 로드 실패: {json_path} - {str(e)}")
        return None


def safe_json_save(data: Dict, json_path: str) -> bool:
    """안전한 JSON 저장"""
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ JSON 저장 실패: {json_path} - {str(e)}")
        return False


def ensure_dir_exists(dir_path: str) -> bool:
    """디렉토리 존재 확인 및 생성"""
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"❌ 디렉토리 생성 실패: {dir_path} - {str(e)}")
        return False


def get_dir_size(dir_path: str) -> float:
    """디렉토리 크기 계산 (MB)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, FileNotFoundError):
                continue
    
    return total_size / 1024 / 1024  # MB


def cleanup_temp_files(temp_dir: str = "./temp", max_age_hours: int = 24):
    """임시 파일 정리"""
    if not os.path.exists(temp_dir):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    logger.debug(f"🧹 임시 파일 삭제: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️  임시 파일 삭제 실패: {file_path} - {str(e)}")


# ================================
# 이미지 처리 유틸리티
# ================================

def safe_image_load(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """안전한 이미지 로딩"""
    try:
        # OpenCV로 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"⚠️  OpenCV 로드 실패, PIL로 재시도: {image_path}")
            
            # PIL로 재시도
            pil_image = Image.open(image_path).convert('RGB')
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # BGR -> RGB 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 크기 조정 (요청된 경우)
        if target_size:
            image = cv2.resize(image, target_size)
        
        return image
        
    except Exception as e:
        logger.error(f"❌ 이미지 로딩 실패: {image_path} - {str(e)}")
        
        # 더미 이미지 반환
        if target_size:
            return np.zeros((*target_size[::-1], 3), dtype=np.uint8)
        else:
            return np.zeros((224, 224, 3), dtype=np.uint8)


def validate_image_dataset(image_dir: str, min_size: int = 32, 
                          max_aspect_ratio: float = 10.0) -> Dict[str, Any]:
    """이미지 데이터셋 검증"""
    
    results = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'errors': []
    }
    
    if not os.path.exists(image_dir):
        results['errors'].append(f"디렉토리 없음: {image_dir}")
        return results
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                results['total_images'] += 1
                image_path = os.path.join(root, file)
                
                try:
                    image = safe_image_load(image_path)
                    if image is not None:
                        h, w = image.shape[:2]
                        
                        # 크기 검증
                        if min(h, w) < min_size:
                            results['errors'].append(f"크기 부족: {image_path} ({w}x{h})")
                            results['invalid_images'] += 1
                            continue
                        
                        # 종횡비 검증
                        aspect_ratio = max(h, w) / min(h, w)
                        if aspect_ratio > max_aspect_ratio:
                            results['errors'].append(f"종횡비 초과: {image_path} ({aspect_ratio:.2f})")
                            results['invalid_images'] += 1
                            continue
                        
                        results['valid_images'] += 1
                    else:
                        results['invalid_images'] += 1
                        
                except Exception as e:
                    results['invalid_images'] += 1
                    results['errors'].append(f"검증 오류: {image_path} - {str(e)}")
    
    return results


# ================================
# 성능 모니터링
# ================================

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """타이머 시작"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """타이머 종료 및 시간 반환"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            
            return duration
        else:
            logger.warning(f"⚠️  타이머가 시작되지 않음: {operation}")
            return 0.0
    
    @contextmanager
    def timer(self, operation: str):
        """컨텍스트 매니저 타이머"""
        self.start_timer(operation)
        try:
            yield
        finally:
            duration = self.end_timer(operation)
            logger.info(f"⏱️  {operation}: {duration:.3f}초")
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """성능 통계 반환"""
        stats = {}
        
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'std': np.std(times)
                }
        
        return stats
    
    def print_statistics(self):
        """성능 통계 출력"""
        stats = self.get_statistics()
        
        if not stats:
            print("📊 성능 통계 없음")
            return
        
        print("\n📊 성능 통계:")
        print("-" * 60)
        print(f"{'Operation':<20} {'Count':<8} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10}")
        print("-" * 60)
        
        for operation, stat in stats.items():
            print(f"{operation:<20} {stat['count']:<8} {stat['average']:<10.3f} "
                  f"{stat['min']:<10.3f} {stat['max']:<10.3f}")
        
        print("-" * 60)


# 전역 성능 모니터
performance_monitor = PerformanceMonitor()


# ================================
# 에러 복구 유틸리티
# ================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0):
    """실패시 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"❌ {func.__name__} 최종 실패 after {max_retries} retries: {str(e)}")
                        raise
                    else:
                        logger.warning(f"⚠️  {func.__name__} 실패 (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
        return wrapper
    return decorator


def safe_operation(operation_name: str = "Operation"):
    """안전한 작업 실행 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"🚀 {operation_name} 시작")
                result = func(*args, **kwargs)
                logger.info(f"✅ {operation_name} 완료")
                return result
            except Exception as e:
                logger.error(f"❌ {operation_name} 실패: {str(e)}")
                logger.error(traceback.format_exc())
                return None
        return wrapper
    return decorator


# ================================
# 시스템 정보
# ================================

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 수집"""
    info = {
        'python_version': '.'.join(map(str, __import__('sys').version_info[:3])),
        'platform': __import__('platform').platform(),
        'cpu_count': psutil.cpu_count(),
        'ram_total_gb': psutil.virtual_memory().total / 1024**3,
        'ram_available_gb': psutil.virtual_memory().available / 1024**3,
    }
    
    # PyTorch 정보
    try:
        info.update({
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        })
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
    except:
        pass
    
    return info


def print_system_info():
    """시스템 정보 출력"""
    info = get_system_info()
    
    print("\n💻 시스템 정보:")
    print("-" * 40)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("-" * 40)


# ================================
# 전역 초기화
# ================================

def initialize_environment(seed: int = 42, deterministic: bool = True):
    """환경 초기화"""
    
    # 재현성 설정
    if deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # 로깅 설정
    logger.info("🔧 환경 초기화 완료")
    
    # 시스템 정보 출력
    if logger.isEnabledFor(logging.INFO):
        print_system_info()
    
    # 임시 파일 정리
    cleanup_temp_files()
    
    # 메모리 정리
    clear_memory(verbose=False)


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 유틸리티 모듈 테스트")
    
    # 환경 초기화
    initialize_environment()
    
    # 메모리 모니터링 테스트
    with memory_monitor("테스트 작업"):
        # 더미 텐서 생성
        dummy_tensor = torch.randn(1000, 1000)
        time.sleep(0.1)
    
    # 성능 모니터링 테스트
    with performance_monitor.timer("더미 연산"):
        time.sleep(0.05)
    
    performance_monitor.print_statistics()
    
    # 메모리 정리
    clear_memory()
    
    print("✅ 유틸리티 모듈 테스트 완료")