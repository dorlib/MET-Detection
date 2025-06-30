from monai.networks.nets import UNETR

def unetr_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    """
    Initialize UNETR model from MONAI.
    """
    model = UNETR(
        in_channels=IMG_CHANNELS,
        out_channels=num_classes,
        img_size=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        dropout_rate=0.0,
        norm_name="instance",
    )
    return model
