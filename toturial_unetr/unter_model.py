from monai.networks.nets import UNETR

def unetr_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS=1, num_classes=4):
    """
    Initialize UNETR model from MONAI.
    
    Parameters:
    -----------
    IMG_HEIGHT : int
        Height of the input volume
    IMG_WIDTH : int
        Width of the input volume
    IMG_DEPTH : int
        Depth of the input volume
    IMG_CHANNELS : int, default=1
        Number of input channels (1 for t1c-only)
    num_classes : int, default=4
        Number of segmentation classes
    
    Returns:
    --------
    model : UNETR
        UNETR model instance
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
