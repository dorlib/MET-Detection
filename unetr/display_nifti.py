import os
import nibabel as nib
import matplotlib.pyplot as plt

def nii_to_png(
    input_dir: str,
    output_dir: str,
    slice_axis: int = 2,
    slice_index: int = None,
    cmap: str = 'gray',
    interpolation: str = 'nearest'
):
    """
    Convert all .nii/.nii.gz volumes in input_dir to 2D PNG slices.

    Parameters:
    - input_dir:      folder containing .nii or .nii.gz files
    - output_dir:     folder where .png files will be saved
    - slice_axis:     which axis to slice (0, 1 or 2; default=2 for axial)
    - slice_index:    which slice index to extract (default=None → middle slice)
    - cmap:           matplotlib colormap (default='gray')
    - interpolation:  matplotlib interpolation (default='nearest')
    """
    os.makedirs(output_dir, exist_ok=True)

    # find all NIfTI files
    nii_files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ]

    for fname in nii_files:
        fpath = os.path.join(input_dir, fname)
        img = nib.load(fpath)
        data = img.get_fdata()

        # determine which slice to take
        if slice_index is None:
            slice_index_use = data.shape[slice_axis] // 2
        else:
            slice_index_use = slice_index

        # extract the 2D slice
        if slice_axis == 0:
            slc = data[slice_index_use, :, :]
        elif slice_axis == 1:
            slc = data[:, slice_index_use, :]
        else:
            slc = data[:, :, slice_index_use]

        # plot and save
        plt.figure(figsize=(6, 6))
        plt.imshow(
            slc.T, 
            cmap=cmap, 
            origin='lower', 
            interpolation=interpolation
        )
        plt.axis('off')
        out_name = fname.replace('.nii.gz', '').replace('.nii', '')
        out_path = os.path.join(
            output_dir, 
            f"{out_name}_axis{slice_axis}_slice{slice_index_use}.png"
        )
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    # --- Edit these paths before running ---
    input_folder  = "../Data/training/training/BraTS-MET-00007-000"       # where your .nii/.nii.gz live
    output_folder = "." # where you want the .png files

    # Run conversion (middle axial slice, nearest interp)
    nii_to_png(
        input_folder,
        output_folder,
        slice_axis=2,    # 0=coronal, 1=sagittal, 2=axial
        slice_index=None,# default→middle slice
        cmap='gray',
        interpolation='nearest'
    )

