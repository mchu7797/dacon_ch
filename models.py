import timm


def get_models(*, num_classes: int):
    models = []

    models.append(
        timm.create_model(
            "timm/convnext_large.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
    )
    models.append(
        timm.create_model(
            "timm/tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
    )
    models.append(
        timm.create_model(
            "timm/regnety_120.sw_in12k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
    )
