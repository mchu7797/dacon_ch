import timm


def get_models(*, num_classes: int) -> list:
    return [
        timm.create_model(
            "timm/maxvit_large_tf_384.in21k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        ),
        timm.create_model(
            "timm/convnextv2_base.fcmae_ft_in22k_in1k_384",
            pretrained=True,
            num_classes=num_classes,
        ),
        timm.create_model(
            "timm/tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        ),
    ]
