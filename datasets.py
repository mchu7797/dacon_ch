# datasets.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np # numpy를 임포트합니다.

class CarDataset(Dataset):
    def __init__(
        self,
        root_directory,
        *,
        is_test: bool = False,
        transform=None,
        tta_transforms=None,
    ):
        self.root_directory = root_directory
        self.is_test = is_test
        self.transform = transform
        self.tta_transforms = tta_transforms
        self.data = []

        if is_test:
            for filename in sorted(os.listdir(root_directory)):
                if filename.lower().endswith(".jpg"):
                    self.data.append(os.path.join(root_directory, filename))
        else:
            self.classes = sorted(os.listdir(root_directory))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            for class_name in self.classes:
                class_dir = os.path.join(root_directory, class_name)
                for filename in sorted(os.listdir(class_dir)):
                    if filename.lower().endswith(".jpg"):
                        self.data.append(
                            (
                                os.path.join(class_dir, filename),
                                self.class_to_idx[class_name],
                            )
                        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_test:
            image_path = self.data[idx]
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image) # 1. 이미지를 numpy 배열로 변환

            if self.tta_transforms is not None:
                tta_images = []
                for transform in self.tta_transforms:
                    # 2. image= 키워드 인자를 사용하여 변환 적용
                    transformed = transform(image=image_np)
                    tta_images.append(transformed["image"]) # 3. 결과 딕셔너리에서 'image' 키로 텐서를 가져옴
                return torch.stack(tta_images)

            if self.transform:
                transformed = self.transform(image=image_np)
                image = transformed["image"]

            return image
        else:
            image_path, label = self.data[idx]
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image) # 1. 이미지를 numpy 배열로 변환

            if self.transform:
                # 2. image= 키워드 인자를 사용하여 변환 적용
                transformed = self.transform(image=image_np)
                image = transformed["image"] # 3. 결과 딕셔너리에서 'image' 키로 텐서를 가져옴
            return image, label