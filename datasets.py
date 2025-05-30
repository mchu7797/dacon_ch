import os
from torch.utils.data import Dataset
from PIL import Image


class CarDataset(Dataset):
    def __init__(self, root_directory, *, is_test: bool = False, transform=None):
        self.root_directory = root_directory
        self.is_test = is_test
        self.transform = transform
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
            if self.transform:
                image = self.transform(image)
            return image
        else:
            image_path, label = self.data[idx]
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
