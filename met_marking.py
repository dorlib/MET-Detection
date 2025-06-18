import numpy as np
from scipy import ndimage
from display_npy import Viewer

def analyze_lesions(
    image_path: str,
    mask_path: str,
    prediction = None, 
    voxel_spacing=(1.0, 1.0, 1.0)
):
    # --- load data ---
    image = np.load(image_path)               # raw brain volume
    mask  = np.load(mask_path)
    mask  = np.argmax(mask, axis=-1)          # collapse one-hot → 3D labels
    image  = np.argmax(image, axis=-1)          # collapse one-hot → 3D labels
    # physical volume of one voxel
    voxel_vol_mm3 = np.prod(voxel_spacing)

    # find all distinct class‐labels
    labels = np.unique(mask)
    print("Classes in mask:", labels)

    name_map = {
        1: "Metastasis",
        2: "Edema",
        3: "Tumor core"
    }
    # set up napari
    viewer = Viewer(image_path=image_path, mask_path=mask_path, mode_3d=True)
    
    # add the raw brain image underneath everything
    viewer.viewer.add_image(
        image,
        name="Raw Image",
        opacity=.5
    )

    # define colors for each class (RGBA)
    color_map = {
        1: (1.0, 0.0, 0.0, 0.6),   # red
        2: (0.0, 1.0, 0.0, 0.6),   # green
        3: (0.0, 0.0, 1.0, 0.6),   # blue
        # etc.
    }

    for class_label in labels:
        if class_label == 0:
            continue  # skip background

        # binary mask for this class
        binary = (mask == class_label)

        # split into connected components
        cc_map, num_instances = ndimage.label(binary)
        
        
        cmap = np.zeros((2, 4), float)
        col = color_map.get(class_label, (1.0, 1.0, 0.0, 0.6))
        cmap[1] = col
        print(f"Class {class_label}: {num_instances} instance(s)")
        viewer.viewer.add_labels(
                binary,
                name=f"{name_map[class_label]}_total",
                colormap = cmap,
                opacity=0.6        
            )
        for inst_idx in range(1, num_instances + 1):
            lesion_mask = (cc_map == inst_idx).astype(np.uint16)

            # compute volume if you like
            vox = int(lesion_mask.sum())
            mm3 = vox * voxel_vol_mm3
            print(f"  • label_{class_label}_{inst_idx}: {vox} voxels ({mm3:.2f} mm³)")




            # add the layer with our custom colormap
            viewer.viewer.add_labels(
                lesion_mask,
                name=f"{name_map[class_label]}_{inst_idx}",
                colormap=cmap,      
                opacity=0.6, 

            )
    if prediction != None: 
    	viewer.visualize_prediction(prediction)
    for layer in viewer.viewer.layers:
        if layer.name != "Raw Image" and layer.name !="Prediction" and not "total" in layer.name:
            layer.visible = False

    viewer.run()


analyze_lesions(

    image_path="MET_samples/images/image_0.npy",
    mask_path="MET_samples/masks/mask_0.npy",
    prediction="MET_samples/masks/prediction_0.npy",
    #target_label=3
)
