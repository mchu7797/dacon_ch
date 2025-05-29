import os
import time
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
                # 파일 접근 가능성 확인
                if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
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

            if not os.path.exists(class_directory):
                print(f"⚠️ Warning: Class directory not found: {class_directory}")
                continue

            for filename in sorted(os.listdir(class_directory)):
                if self._is_valid_image(filename):
                    file_path = os.path.join(class_directory, filename)
                    # 파일 접근 가능성 확인
                    if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                        self.samples.append((file_path, self.class_to_idx[class_name]))
                        class_samples += 1

            print(f"Class '{class_name}': {class_samples} samples")

    def _is_valid_image(self, filename: str) -> bool:
        """이미지 파일 검증"""
        return any(filename.lower().endswith(ext) for ext in self.valid_extensions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self.is_test:
                    image_path, filename = self.samples[idx]
                    image = self._load_image_safely(image_path)
                    if self.transform:
                        image = self.transform(image)
                    return image, filename
                else:
                    image_path, label = self.samples[idx]
                    image = self._load_image_safely(image_path)
                    if self.transform:
                        image = self.transform(image)
                    return image, label

            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"❌ Failed to load sample {idx} after {max_retries} attempts: {e}"
                    )
                    # 대체 샘플 반환
                    return self._get_fallback_sample()
                else:
                    print(f"⚠️ Retry {attempt + 1}/{max_retries} for sample {idx}")
                    time.sleep(0.1)

    def _load_image_safely(self, image_path: str) -> Image.Image:
        """안전한 이미지 로딩 with 재시도"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                image = Image.open(image_path).convert("RGB")
                # 이미지 유효성 검증
                image.verify()
                # verify() 후에는 이미지를 다시 열어야 함
                image = Image.open(image_path).convert("RGB")
                return image

            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"❌ Failed to load image {image_path} after {max_retries} attempts: {e}"
                    )
                    # 기본 검은색 이미지 생성
                    return Image.new("RGB", (224, 224), color="black")
                else:
                    time.sleep(0.1)  # 짧은 대기 후 재시도

    def _get_fallback_sample(self):
        """오류 시 대체 샘플 반환"""
        fallback_image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            fallback_image = self.transform(fallback_image)

        if self.is_test:
            return fallback_image, "fallback.jpg"
        else:
            return fallback_image, 0  # 첫 번째 클래스로 설정
