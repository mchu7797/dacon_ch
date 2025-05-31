from torchvision import transforms
from typing import Tuple


def get_train_transform(
    image_size: int, mean: Tuple[int], std: Tuple[int]
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_val_transform(
    image_size: int, mean: Tuple[int], std: Tuple[int]
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_tta_transforms(
    image_size: int,
    mean: Tuple[int],
    std: Tuple[int],
) -> list[transforms.Compose]:
    tta_transforms = [
        # 원본
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        # 좌우 반전
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),  # 항상 적용
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        # 약간의 회전 (5도)
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=5, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        # 약간의 회전 (-5도)
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=-5, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    ]
    return tta_transforms
