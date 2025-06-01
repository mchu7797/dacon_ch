from dataclasses import dataclass


@dataclass
class Config:
    train_directory: str = "train"
    test_directory: str = "test"
    model_directory: str = "models"

    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_minimum_delta: float = 0.0002

    use_tta: bool = True
    tta_batch_size_divisor: int = 4

    image_size: int = 224
    batch_size: int = 128
    epochs: int = 25
    learning_rate: float = 3e-4
    seed: int = 42

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


def get_config():
    return Config()
