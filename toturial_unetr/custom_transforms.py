import numpy as np
from monai.transforms import MapTransform

class LoadNpy(MapTransform):
    """
    Custom MONAI transform to load .npy files.
    """
    def __init__(self, keys, dtype=np.float32):
        super().__init__(keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                filename = d[key]
                if isinstance(filename, str):
                    d[key] = np.load(filename, allow_pickle=True).astype(self.dtype)
        return d
