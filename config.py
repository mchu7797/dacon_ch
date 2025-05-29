from dataclasses import dataclass
from typing import List, Dict, Any
import os


@dataclass
class Config:
    """앙상블 훈련 및 추론 설정"""

    # 기본 설정 (기존 Config 상속)
    train_root: str = "./train"
    test_root: str = "./test"
    image_size: int = 384
    batch_size: int = 12  # 큰 모델들이므로 배치 크기 감소

    # 훈련 설정
    num_epochs: int = 25
    learning_rate: float = 8e-5  # 큰 모델들이므로 학습률 약간 감소
    weight_decay: float = 1e-4

    # 앙상블 모델 구성
    ensemble_models: List[Dict[str, Any]] = None

    # 기타 설정
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    patience: int = 8
    min_delta: float = 1e-4

    # K-Fold 설정
    n_folds: int = 5

    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = [
                {
                    "name": "convnext_large",
                    "model_name": "convnext_large",
                    "pretrained": True,
                    "dropout_rate": 0.3,
                    "weight": 1.0,  # 앙상블 가중치
                },
                {
                    "name": "efficientnet_v2_large",
                    "model_name": "tf_efficientnetv2_l",
                    "pretrained": True,
                    "dropout_rate": 0.3,
                    "weight": 1.0,
                },
                {
                    "name": "regnet_y_128gf",
                    "model_name": "regnety_128",
                    "pretrained": True,
                    "dropout_rate": 0.2,
                    "weight": 1.0,
                },
            ]

        # 설정 검증
        assert self.image_size > 0, "image_size must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert len(self.ensemble_models) > 0, "At least one model required"

        # 경로 존재 확인
        if not os.path.exists(self.train_root):
            raise FileNotFoundError(f"Train directory not found: {self.train_root}")
