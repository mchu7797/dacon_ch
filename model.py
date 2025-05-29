import torch
import torch.nn as nn
import timm


class ImprovedModel(nn.Module):
    """Enhanced model with modern architecture"""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # 최신 모델 사용
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # feature extractor로 사용
            global_pool="",  # 커스텀 pooling 사용
        )

        # Feature dimension 자동 계산
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:  # [B, C, H, W]
                self.feature_dim = features.shape[1]
            else:  # [B, C]
                self.feature_dim = features.shape[1]

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

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


# 기존 모델 유지 (호환성)
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.feature_dim = 2048  # ResNet50 feature dimension
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
