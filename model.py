import torch
import torch.nn as nn
import timm
import numpy as np
from typing import List, Dict, Any
import gc


class Model(nn.Module):
    """앙상블 모델 클래스"""

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        img_size: int = 384,
    ):
        super().__init__()

        # TIMM 모델 생성
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # feature extractor로 사용
            global_pool="",
        )

        # Feature dimension 안전한 계산
        self.feature_dim = self._get_feature_dim_safely(img_size)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(1024, num_classes),
        )

    def _get_feature_dim_safely(self, img_size: int) -> int:
        """GPU 메모리 절약을 위한 안전한 feature dimension 계산"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)

            try:
                # GPU에서 시도
                if torch.cuda.is_available():
                    features = self.backbone(dummy_input.cuda())
                    feature_dim = (
                        features.shape[1]
                        if len(features.shape) == 4
                        else features.shape[1]
                    )
                    # 즉시 메모리 정리
                    del features, dummy_input
                    torch.cuda.empty_cache()
                else:
                    features = self.backbone(dummy_input)
                    feature_dim = (
                        features.shape[1]
                        if len(features.shape) == 4
                        else features.shape[1]
                    )
                    del features, dummy_input

                return feature_dim

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠️ GPU OOM during feature dim calculation, using CPU")
                    torch.cuda.empty_cache()
                    # CPU에서 계산
                    features = self.backbone(dummy_input.cpu())
                    feature_dim = (
                        features.shape[1]
                        if len(features.shape) == 4
                        else features.shape[1]
                    )
                    del features, dummy_input
                    gc.collect()
                    return feature_dim
                else:
                    raise e

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # Global pooling if needed
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)

        # Classification
        output = self.classifier(features)
        return output


class Ensemble:
    """Soft Voting 앙상블 클래스"""

    def __init__(self, model_configs: List[Dict[str, Any]], device: torch.device):
        self.model_configs = model_configs
        self.device = device
        self.models = []
        self.weights = []

    def load_models(self, model_paths: List[str], num_classes: int):
        """저장된 모델들을 로드"""
        assert len(model_paths) == len(self.model_configs), (
            "모델 경로와 설정 수가 일치하지 않습니다"
        )

        for i, (config, model_path) in enumerate(zip(self.model_configs, model_paths)):
            print(f"Loading model {i + 1}: {config['name']}")

            try:
                # 모델 생성
                model = Model(
                    num_classes=num_classes,
                    model_name=config["model_name"],
                    pretrained=False,  # 체크포인트에서 로드하므로 False
                    dropout_rate=config["dropout_rate"],
                )

                # 체크포인트 로드
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.models.append(model)
                self.weights.append(config["weight"])

                print(f"✅ {config['name']} loaded successfully")

                # 메모리 정리
                del checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Failed to load {config['name']}: {e}")
                raise e

        # 가중치 정규화
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        print(f"📊 Ensemble weights: {[f'{w:.3f}' for w in self.weights]}")

    def predict(self, dataloader) -> np.ndarray:
        """앙상블 예측 수행"""
        all_predictions = []

        print("🔮 Performing ensemble prediction...")

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                try:
                    images = images.to(self.device)
                    batch_predictions = []

                    # 각 모델의 예측 수집
                    for model in self.models:
                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1)
                        batch_predictions.append(probs.cpu().numpy())

                    # 가중 평균으로 앙상블
                    weighted_pred = np.zeros_like(batch_predictions[0])
                    for pred, weight in zip(batch_predictions, self.weights):
                        weighted_pred += pred * weight

                    all_predictions.append(weighted_pred)

                    # 메모리 정리
                    del images, batch_predictions, weighted_pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if batch_idx % 10 == 0:
                        print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(
                            f"⚠️ OOM at batch {batch_idx}, clearing cache and retrying..."
                        )
                        torch.cuda.empty_cache()
                        gc.collect()
                        # 재시도
                        continue
                    else:
                        raise e

        return np.vstack(all_predictions)

    def predict_single_model(self, dataloader, model_idx: int) -> np.ndarray:
        """단일 모델 예측 (개별 성능 확인용)"""
        model = self.models[model_idx]
        all_predictions = []

        with torch.no_grad():
            for images, _ in dataloader:
                try:
                    images = images.to(self.device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    all_predictions.append(probs.cpu().numpy())

                    # 메모리 정리
                    del images, outputs, probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("⚠️ OOM in single model prediction, clearing cache...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

        return np.vstack(all_predictions)
