from dataclasses import dataclass


@dataclass
class Config:
    train_directory: str = "train"
    test_directory: str = "test"

    image_size: int = 224
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 0.001
    seed: int = 42

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


def get_config():
    return Config()
