import timm


def get_models(*, num_classes: int) -> list:
    return [
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
        timm.create_model(
            "timm/regnety_120.sw_in12k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        ),
        timm.create_model(
            "timm/vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        ),
        timm.create_model(
            "timm/vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=num_classes,
        )
    ]
