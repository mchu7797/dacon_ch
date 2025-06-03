import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle


class CarDataset(Dataset):
    def __init__(
        self,
        root_directory,
        brand_predictions_path,
        brand_info_path,
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

        self.brand_classes = None
        if brand_info_path and os.path.exists(brand_info_path):
            with open(brand_info_path, "rb") as f:
                brand_info = pickle.load(f)
            self.brand_classes = brand_info["brands"]

        self.brand_predictions = None
        if brand_predictions_path and os.path.exists(brand_predictions_path):
            with open(brand_predictions_path, "rb") as f:
                self.brand_predictions = pickle.load(f)

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
            filename = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")

            brand_logits = None
            if self.brand_predictions and filename in self.brand_predictions:
                brand_logits = torch.tensor(
                    self.brand_predictions[filename], dtype=torch.float32
                )

            if self.tta_transforms is not None:
                tta_images = []
                for transform in self.tta_transforms:
                    tta_images.append(transform(image))
                if brand_logits is not None:
                    return torch.stack(tta_images), brand_logits
                else:
                    return torch.stack(tta_images)

            if self.transform:
                image = self.transform(image)

            return (image, brand_logits) if brand_logits is not None else image
        else:
            image_path, label = self.data[idx]
            filename = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")

            brand_logits = None
            if self.brand_predictions and filename in self.brand_predictions:
                brand_logits = torch.tensor(
                    self.brand_predictions[filename], dtype=torch.float32
                )

            if self.transform:
                image = self.transform(image)
            return (
                (image, brand_logits, label)
                if brand_logits is not None
                else (image, label)
            )
