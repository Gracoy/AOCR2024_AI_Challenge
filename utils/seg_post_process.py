import glob
import os
import nibabel as nib
import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label


"""
This file is used to optimize the segmentation results, there are two steps.
Step 1: Remove any label that appear at border
Step 2: Keep the largest component
"""


def clear_border_and_keep_largest_component(result_folder):
    gz_files = glob.glob(result_folder + "\\*")

    for gz in gz_files:
        file_name = gz.split("\\")[-1].replace(".nii.gz", "")
        if not file_name.startswith("Appendix"):
            continue

        # read original segmentation result
        mask_file = nib.load(gz)
        mask_affine = mask_file.affine
        mask_meta_data = mask_file.header
        origin_mask = mask_file.get_fdata().astype(np.int16)
        h, w, d = origin_mask.shape

        # append two sliced at start and end of 3D array
        mask_clear_border = np.zeros((h, w, d + 2), dtype=np.int16)
        mask_clear_border[:, :, 1:d+1] = np.copy(origin_mask)

        # remove any label that appear at border
        mask_clear_border = clear_border(mask_clear_border)
        mask_clear_border = mask_clear_border[:, :, 1:d+1]

        # keep the largest component
        mask = np.zeros_like(mask_clear_border, dtype=bool)
        mask |= mask_clear_border > 0
        mask_keep = remove_all_but_largest_component(mask)
        ret = np.copy(mask_clear_border)
        ret[mask & ~mask_keep] = 0

        # save as new nii file
        new_mask = nib.Nifti1Image(ret, mask_affine, mask_meta_data)
        nib.save(new_mask, f"{result_folder}\\{file_name}_pp.nii.gz")
        os.remove(gz)


def remove_all_but_largest_component(binary_image: np.ndarray, connectivity: int = None) -> np.ndarray:
    """
    Removes all but the largest component in binary_image. Replaces pixels that don't belong to it with background_label
    """
    return generic_filter_components(binary_image, connectivity)


def label_with_component_sizes(binary_image: np.ndarray, connectivity: int = None):
    labeled_image, num_components = label(binary_image, return_num=True, connectivity=connectivity)
    component_sizes = {i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])}
    return labeled_image, component_sizes


def generic_filter_components(binary_image: np.ndarray, connectivity: int = None):
    """
    filter_fn MUST return the component ids that should be KEPT!
    filter_fn will be called as: filter_fn(component_ids, component_sizes) and is expected to return a List of int

    returns a binary array that is True where the filtered components are
    """
    labeled_image, component_sizes = label_with_component_sizes(binary_image, connectivity)
    component_ids = list(component_sizes.keys())
    component_sizes = list(component_sizes.values())
    keep = [i for i, j in zip(component_ids, component_sizes) if j == max(component_sizes)]

    # if there are more than one largest components, then we consider this scan as label 0
    if len(keep) > 1:
        return np.zeros_like(labeled_image, dtype=bool)
    return np.in1d(labeled_image.ravel(), keep).reshape(labeled_image.shape)


if __name__ == "__main__":
    seg_res_folder = "D:\\Dataset779_Appendix3labels_seg_result_tv1000_pp"
    clear_border_and_keep_largest_component(seg_res_folder)
    pass
