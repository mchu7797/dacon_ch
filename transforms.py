import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


def get_train_transform(
    image_size: int, mean: Tuple[float], std: Tuple[float]
) -> A.Compose:
    return A.Compose(
        [
            # 1. 이미지 비율을 유지하면서 가장 긴 쪽을 image_size에 맞춤
            A.LongestMaxSize(max_size=image_size),
            # 2. 부족한 부분을 채워 image_size x image_size 정사각형으로 만듦
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),

            # 3. 고급 데이터 증강 기법들
            A.HorizontalFlip(p=0.5), # 좌우 반전
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), # 이동, 스케일, 회전
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # 밝기, 대비 조절
            A.CoarseDropout(max_holes=8, max_height=image_size//10, max_width=image_size//10, min_holes=1, fill_value=0, p=0.5), # 이미지 일부에 구멍(Cutout)

            # 4. 정규화 및 텐서 변환
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transform(
    image_size: int, mean: Tuple[float], std: Tuple[float]
) -> A.Compose:
    return A.Compose(
        [
            # 검증/테스트 시에는 비율 유지 리사이즈와 정규화만 적용
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(
    image_size: int,
    mean: Tuple[float],
    std: Tuple[float],
) -> list[A.Compose]:
    tta_transforms = [
        # TTA 1: 원본 (비율 유지 리사이즈)
        A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]),
        # TTA 2: 좌우 반전
        A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]),
    ]
    return tta_transforms