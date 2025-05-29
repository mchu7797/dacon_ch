import os
from PIL import Image
from torch.utils.data import Dataset


class CarDataset(Dataset):
    """Enhanced car dataset with better error handling"""

    def __init__(
        self,
        root_directory: str,
        *,
        transform=None,
        is_test: bool = False,
    ):
        self.root_directory = root_directory
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        # 지원하는 이미지 확장자
        self.valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        if not os.path.exists(root_directory):
            raise FileNotFoundError(f"Directory not found: {root_directory}")

        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {root_directory}")

    def _load_samples(self):
        """샘플 로딩 로직"""
        if self.is_test:
            self._load_test_samples()
        else:
            self._load_train_samples()

    def _load_test_samples(self):
        """테스트 샘플 로딩"""
        for filename in sorted(os.listdir(self.root_directory)):
            if self._is_valid_image(filename):
                file_path = os.path.join(self.root_directory, filename)
                self.samples.append((file_path, filename))

    def _load_train_samples(self):
        """훈련 샘플 로딩"""
        self.classes = sorted(
            [
                d
                for d in os.listdir(self.root_directory)
                if os.path.isdir(os.path.join(self.root_directory, d))
            ]
        )

        if not self.classes:
            raise ValueError("No class directories found")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_name in self.classes:
            class_directory = os.path.join(self.root_directory, class_name)
            class_samples = 0

            for filename in sorted(os.listdir(class_directory)):
                if self._is_valid_image(filename):
                    file_path = os.path.join(class_directory, filename)
                    self.samples.append((file_path, self.class_to_idx[class_name]))
                    class_samples += 1

            print(f"Class '{class_name}': {class_samples} samples")

    def _is_valid_image(self, filename: str) -> bool:
        """이미지 파일 검증"""
        return any(filename.lower().endswith(ext) for ext in self.valid_extensions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            if self.is_test:
                image_path, filename = self.samples[idx]
                image = self._load_image(image_path)
                if self.transform:
                    image = self.transform(image)
                return image, filename
            else:
                image_path, label = self.samples[idx]
                image = self._load_image(image_path)
                if self.transform:
                    image = self.transform(image)
                return image, label
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # 기본 이미지 반환 또는 다음 샘플로 건너뛰기
            raise

    def _load_image(self, image_path: str) -> Image.Image:
        """안전한 이미지 로딩"""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            # 기본 검은색 이미지 생성
            return Image.new("RGB", (224, 224), color="black")
