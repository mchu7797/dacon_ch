import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedModel(nn.Module):
    def __init__(self, base_model: nn.Module, brand_classes: int):
        super(AdvancedModel, self).__init__()
        self.base_model = base_model
        self.num_classes = base_model.num_classes

        if hasattr(base_model, "head"):
            feature_dim = base_model.head.in_features
            base_model.head = nn.Identity()
        elif hasattr(base_model, "classifier"):
            feature_dim = base_model.classifier.in_features
            base_model.classifier = nn.Identity()

        self.brand_fc = nn.Linear(brand_classes, 256)

        combined_feature_dim = feature_dim + 256
        self.final_classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, combined_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_feature_dim // 2, combined_feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(combined_feature_dim // 4, self.num_classes),
        )

    def forward(self, x: torch.Tensor, brand_logits: torch.Tensor) -> torch.Tensor:
        image_features = self.base_model(x)

        if len(image_features.shape) == 4:
            image_features = F.adaptive_avg_pool2d(image_features, (1, 1))
            image_features = image_features.view(image_features.size(0), -1)

        brand_features = F.relu(self.brand_fc(brand_logits))

        combined_features = torch.cat((image_features, brand_features), dim=1)
        output = self.final_classifier(combined_features)
        return output


class BrandModel(nn.Module):
    def __init__(self, num_brands):
        super(BrandModel, self).__init__()
        self.model = timm.create_model(
            "timm/convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=num_brands,
        )

    def forward(self, x):
        return self.model(x)


def get_models(*, num_classes: int, brand_classes: int | None) -> list:
    base_models = [
        timm.create_model(
            "timm/convnextv2_large.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=num_classes,
        ),
        timm.create_model(
            "timm/convnext_large.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        ),
        timm.create_model(
            "timm/tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        ),
    ]

    if brand_classes is not None:
        return [AdvancedModel(base_model, brand_classes) for base_model in base_models]
    else:
        return base_models
