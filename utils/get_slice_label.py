import pandas as pd
import glob
import nibabel as nib
import numpy as np


"""
This file is used to extract slice level labels from segmentation results.
"""


def get_slice_label(seg_result_dir):
    """
    This program gets the slice label from segmentation result,
    if there is label 1 in the slice, then we consider it as label 1.
    """
    # directory of segmentation result
    gz_files = glob.glob(seg_result_dir + "\\*")

    # Important !!! Because we cropped z-direction from slice 8, so we must add the offset (k0 = 8) back
    k0 = 8

    # restore the file name we recorded during formatting step
    scan_name_df = pd.read_csv("D:\\aocr2024\\nnU_Net\\Dataset779_Appendix3labels\\scan_name_map_test_2.csv")
    scan_name_map = {}
    for index, data in scan_name_df.iterrows():
        scan_name_map[data["sample_id"]] = data["scan_id"]

    res = []
    for idx, gz in enumerate(gz_files):
        if (idx + 1) % 50 == 0:
            print(f"Handling no.{idx + 1} file.")
        mask_nii = nib.load(gz)
        mask_pixels = mask_nii.get_fdata()
        h, w, d = mask_pixels.shape
        file_name = gz.split("\\")[-1].replace("_pp.nii.gz", "")
        scan_name = scan_name_map[file_name]
        for k in range(d):
            frame = mask_pixels[:, :, k]
            uniques = np.unique(frame)
            label_1, label_2 = 0, 0
            if 1 in uniques: label_1 = 1
            if 2 in uniques: label_2 = 1
            res.append([f"{scan_name}_{k + k0}", label_1, label_2])

    label_df = pd.DataFrame(res, columns=["slice_id", "appendix", "fat_stranding"])
    label_df.to_csv("stage_2_slice_label_test200.csv", index=False)


if __name__ == "__main__":
    result_dir = "D:\\seg_result_ts_2_pp"
    get_slice_label(result_dir)
    pass

