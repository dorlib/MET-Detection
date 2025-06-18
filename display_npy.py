
import glasbey
import numpy as np
import napari


glas = glasbey.create_palette(256)

class Viewer:
    def __init__(self, image_path: str, mask_path: str, mode_3d: bool = True):
        self.image = np.load(image_path)
        self.mask = np.load(mask_path).astype(np.uint8)
        self.mode_3d = mode_3d
        self.viewer = napari.Viewer(ndisplay=3 if mode_3d else 2)
        self.label_layer = None
        if self.mask.ndim == 4:
            self.mask  = np.argmax(self.mask , axis=-1)
        if self.image.ndim == 4:
            self.image  = np.argmax(self.image , axis=-1)


    def show_image(self,show_all = True, opacity=0.3, colormap="gray",  mask_opacity = 0.7, mask_colormap = None):
        if show_all:
            self.show_all_labels(mask_opacity, mask_colormap)
        self.viewer.add_image(
            self.image,
            name="MRI",
            colormap=colormap,
            contrast_limits=[self.image.min(), self.image.max()],
            opacity=opacity,
        )


    def show_all_labels(self, opacity=0.5, color_map=None):
        if color_map is None:
            # Default color map: auto assign up to 10 labels
            unique_labels = np.unique(self.mask)
            color_map = {label: (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)
                        for label in unique_labels}
            color_map[0] = (0, 0, 0, 0)  # background fully transparent
        self.label_layer = self.viewer.add_labels(self.mask, name="Segmentation", opacity=opacity)
        self.label_layer.color = color_map


    def show_single_label(self, label_id: int, color=(1, 0, 0, 1), opacity=0.8):
        single_label = np.where(self.mask == label_id, label_id, 0)
        self.label_layer = self.viewer.add_labels(single_label, name=f"Label {label_id}", opacity=opacity)
        self.label_layer.color = {
            0: (0, 0, 0, 0),
            label_id: color
        }
     
    def visualize_prediction(self, predicted_path: str, label_colors=None, opacity=0.5):
        """
        Add predicted mask alongside the loaded image and ground truth mask.
        """
        pred_mask = np.load(predicted_path)
        # Convert one-hot or probability to label index
        if pred_mask.ndim == 4 and pred_mask.shape[-1] > 1:
            pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask.astype(np.uint8)

        # Only use the default colormap if none was provided
        if label_colors is None:
            label_colors = {
                0: (0.0, 0.0, 0.0, 0.0),  # transparent background
                1: (1.0, 0.0, 0.0, 0.5),  # red (metastasis)
                2: (0.6, 0.7, 0.0, 0.5),  # yellow (edema)
                3: (0.0, 1.0, 1.0, 0.5),  # blue (core)
            }

        # Add prediction layer
        pred_layer = self.viewer.add_labels(pred_mask, name="Prediction", opacity=opacity, colormap =glas)


    def set_slice(self, z: int):
        self.viewer.dims.set_point(0, z)

    def run(self):
        napari.run()
        
       








