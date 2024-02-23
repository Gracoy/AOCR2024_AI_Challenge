import numpy as np
import pandas as pd
import glob
import nibabel as nib


"""
These two programs are pre-process step for segmentation model and classification model respectively.
"""


def extract_appendix_bounding_box_from_gt():
    """
    Get the 3D bounding box of appendix from ground truth label.
    Use the coordinates to further crop the CT nii file to reduce usage
    of computing and storage resources when training segmentation model.
    """

    # directory of ground truth labels
    gt_mask_folder = "D:\\aocr2024\\mask_files"

    gz_files = glob.glob(gt_mask_folder + "\\*")
    appendix_window = []
    for idx, gz in enumerate(gz_files):
        if (idx + 1) % 50 == 0:
            print(f"Handling no.{idx + 1} file")
        scan_name = gz.split("\\")[-1].replace("_label_1.nii.gz", "")   # remove suffix and extend of the file name

        # read data and initialize the coordinates
        start_i, end_i = 1000, -1
        start_j, end_j = 1000, -1
        start_k, end_k = 1000, -1
        nii_file = nib.load(gz)
        pixels = nii_file.get_fdata()
        w, h, d = pixels.shape
        if np.sum(pixels) == 0:
            raise ValueError(f"Scan {scan_name} has no label.")

        # iterate through the 3D array
        for k in range(d):
            frame = pixels[:, :, k]
            if np.sum(frame) == 0:
                continue
            start_k = min(start_k, k)
            end_k = max(end_k, k)
            for i in range(h):
                if np.sum(frame[i]) == 0:
                    continue
                start_i = min(start_i, i)
                end_i = max(end_i, i)
                for j in range(w):
                    if frame[i][j] != 0:
                        start_j = min(start_j, j)
                        end_j = max(end_j, j)

        # after iterating the 3D array, record it's scan name and bounding box coordinates
        appendix_window.append([scan_name,
                                start_i, end_i,
                                start_j, end_j,
                                start_k, end_k])

    # save to csv for the subsequent cropping step
    df_slice = pd.DataFrame(appendix_window, columns=["scan_id",
                                                      "start_i", "end_i",
                                                      "start_j", "end_j",
                                                      "start_k", "end_k"])
    df_slice.to_csv("label_bounding_box_coordinates_from_gt.csv", index=False)


def extract_appendix_bounding_box_from_seg():
    """
    Get the 3D bounding box of appendix from segmentation results.
    Use the coordinates to crop the segmented nii label file for classification.
    If there is no label 1 (appendix) in the segmentation result, then we
    record the coordinates as -1 and consider it as normal scan (scan label 0)
    """

    # directory of post-processed segmented results
    seg_result_folder = "D:\\aocr2024\\Dataset779_Appendix_seg_result_pp"

    # due to we change the file name to fit the format required for nnU-Net
    # we need to recover to the original file name corresponding the
    # csv file we store during the file sorting step
    scan_name_df = pd.read_csv("D:\\aocr2024\\nnU_Net\\scan_sample_name_map.csv")
    scan_name_map = {}
    for index, data in scan_name_df.iterrows():
        scan_name_map[data["sample_id"]] = data["scan_id"]

    gz_files = glob.glob(seg_result_folder + "\\*")
    appendix_window = []
    for idx, gz in enumerate(gz_files):
        if idx > 0 and (idx + 1) % 50 == 0:
            print(f"Handling no.{idx + 1} file")
        seg_file_name = gz.split("\\")[-1].replace("_pp.nii.gz", "")      # remove suffix and extend of file name
        scan_name = scan_name_map[seg_file_name]                          # get the corresponding original name

        # read data and initialize the coordinates
        start_i, end_i = 1000, -1
        start_j, end_j = 1000, -1
        start_k, end_k = 1000, -1
        nii_file = nib.load(gz)
        pixels = nii_file.get_fdata().astype(np.int16)
        uniques = np.unique(pixels)

        # if there is no appendix (label 1) in the segmented result
        # then we consider this file has no acute appendicitis (scan label 0)
        if 1 not in uniques:
            appendix_window.append([scan_name, -1, -1, -1, -1, -1, -1])
            continue

        # iterate through the 3D array
        w, h, d = pixels.shape
        for k in range(d):
            frame = pixels[:, :, k]
            if np.sum(frame) == 0:
                continue
            start_k = min(start_k, k)
            end_k = max(end_k, k)
            for i in range(h):
                if np.sum(frame[i]) == 0:
                    continue
                start_i = min(start_i, i)
                end_i = max(end_i, i)
                for j in range(w):
                    if frame[i][j] != 0:
                        start_j = min(start_j, j)
                        end_j = max(end_j, j)

        # this branch might be redundant (?)
        if end_k == -1:
            appendix_window.append([scan_name, -1, -1, -1, -1, -1, -1])
        # record it's scan name and bounding box coordinates
        else:
            appendix_window.append([scan_name,
                                    start_i, end_i,
                                    start_j, end_j,
                                    start_k, end_k])

    df_slice = pd.DataFrame(appendix_window, columns=["scan_id",
                                                      "start_i", "end_i",
                                                      "start_j", "end_j",
                                                      "start_k", "end_k"])
    df_slice.to_csv("appendix_bounding_box_coordinates_from_seg.csv", index=False)


if __name__ == "__main__":
    # extract_appendix_bounding_box_from_gt()
    # extract_appendix_bounding_box_from_seg()
    pass
