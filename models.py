import timm


def get_models(*, num_classes: int) -> list:
    return [
        timm.create_model(
            "timm/vit_large_patch16_224.augreg_in21k_ft_in1k",
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
