import os
from dataclasses import dataclass


@dataclass
class Config:
    """Training configuration with validation"""

    # 데이터 경로
    train_root: str = "./train"
    test_root: str = "./test"

    # 모델 설정
    image_size: int = 384  # 224는 너무 작을 수 있음
    batch_size: int = 16  # GPU 메모리에 따라 조정

    # 훈련 설정
    num_epochs: int = 30  # 10은 너무 적음
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4  # 정규화 추가

    # 스케줄러 설정
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5

    # 데이터 증강
    use_augmentation: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    # 기타
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    def __post_init__(self):
        """설정 검증"""
        assert self.image_size > 0, "image_size must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"

        # 경로 존재 확인
        if not os.path.exists(self.train_root):
            raise FileNotFoundError(f"Train directory not found: {self.train_root}")
