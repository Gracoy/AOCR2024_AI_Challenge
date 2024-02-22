import pandas as pd
import glob, os
import nibabel as nib
import numpy as np


"""
This file is used to crop the 3D nii files corresponding to
the 3D bounding box we obtained from "extract_appendix_window.py"

Important !!! The bounding box coordinates of tasks come from:
Segmentation:   from ground truth labels
Classification: from segmentation results
"""


def crop_for_segmentation_train_val():
    """
    Crop the nii file corresponding to bounding box we obtained and save as new file.
    """
    # directory of train and label data
    gz_files = glob.glob("D:\\aocr2024\\aaa\\*.nii.gz")
    save_path   = "D:\\aocr2024\\nnU_Net\\cropped_3labels"

    # these coordinates were obtained from "extract_appendix_window.py -- extract_appendix_bounding_box_from_gt()"
    i0, i1, j0, j1, k0, k1 = 36, 356, 98, 418, 8, 84
    file_counter = 0

    for gz in gz_files:
        file_counter += 1
        if file_counter % 50 == 0:
            print(f"Handling no.{file_counter} file.")
        scan_name = gz.split("\\")[-1]

        nii_file = nib.load(gz)
        nii_affine = nii_file.affine
        nii_meta_data = nii_file.header
        nii_data = nii_file.get_fdata().astype(np.int16)
        nii_data = nii_data[i0:i1, j0:j1, k0:k1]

        # just to confirm every file is correctly cropped
        if np.sum(nii_data) == 0:
            print(scan_name, nii_data.shape, np.sum(nii_data))

        cropped_nii  = nib.Nifti1Image(nii_data,  nii_affine,  nii_meta_data)
        nib.save(cropped_nii,  f"{save_path}\\{scan_name}")


def crop_for_segmentation_test():
    """
    Same function but used for test dataset
    """
    # directory of test data
    gz_files = glob.glob("D:\\aocr2024\\Test2_Image\\test_stage_2")
    save_path   = "D:\\aocr2024\\nnU_Net\\Dataset779_Appendix3labels\\imagesTs_2"

    i0, i1, j0, j1, k0, k1 = 36, 356, 98, 418, 8, 84
    file_counter = 0

    for gz in gz_files:
        file_counter += 1
        if file_counter % 50 == 0:
            print(f"Handling no.{file_counter} file.")
        scan_name = gz.split("\\")[-1].replace(".nii.gz", "")
        nii_file = nib.load(gz)
        nii_affine = nii_file.affine
        nii_meta_data = nii_file.header
        nii_data = nii_file.get_fdata().astype(np.int16)
        nii_data = nii_data[i0:i1, j0:j1, k0:k1]

        scan_file_name = f"{scan_name}.nii.gz"
        cropped_nii  = nib.Nifti1Image(nii_data,  nii_affine,  nii_meta_data)

        nib.save(cropped_nii,  f"{save_path}\\{scan_file_name}")


def crop_for_classification_train_val():
    """
    Keep the region of the segmentation results where has labels.
    In other words, remove background.
    """
    # directory of post-processed segmentation results
    folder = "D:\\Dataset779_Appendix3labels_seg_result_tv1000_pp_cropped"

    # bounding box coordinates we obtained from "extract_appendix_window.py -- extract_appendix_bounding_box_from_seg()"
    appendix_window_df = pd.read_csv("appendix_box_coordinates_tv1000.csv")
    appendix_window_d = {}
    for idx, data in appendix_window_df.iterrows():
        appendix_window_d[data["scan_id"]] = [data["start_i"], data["end_i"],
                                              data["start_j"], data["end_j"],
                                              data["start_k"], data["end_k"]]

    broader = 5
    file_counter = 0
    count_no_window = 0

    gz_files = glob.glob(folder + "\\*")
    for gz in gz_files:
        file_counter += 1
        if file_counter % 50 == 0:
            print(f"Handling no.{file_counter} file.")

        scan_name = gz.split("\\")[-1].replace(".nii.gz", "")
        i0, i1, j0, j1, k0, k1 = appendix_window_d[scan_name]
        if k0 == -1:
            count_no_window += 1
            continue

        i0, i1 = i0 - broader, i1 + broader
        j0, j1 = j0 - broader, j1 + broader
        k0, k1 = k0 - 1,       k1 + 1

        nii_file = nib.load(gz)
        nii_affine = nii_file.affine
        nii_meta_data = nii_file.header
        nii_data = nii_file.get_fdata().astype(np.int16)
        nii_data = nii_data[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]

        scan_file_name = f"{scan_name}_cropped.nii.gz"
        cropped_nii  = nib.Nifti1Image(nii_data,  nii_affine,  nii_meta_data)

        nib.save(cropped_nii, f"{folder}\\{scan_file_name}")
        os.remove(gz)

    # just to make sure the number of scans that has no label 1 in segmentation result
    print(f"File counts with now window: {count_no_window}")


def crop_for_classification_test():
    """
    Same function but used for test dataset
    """
    folder = "D:\\seg_result_ts_2_pp_cropped"

    appendix_window_df = pd.read_csv("appendix_box_coordinates_test200_stage_2.csv")
    appendix_window_d = {}
    for idx, data in appendix_window_df.iterrows():
        appendix_window_d[data["scan_id"]] = [data["start_i"], data["end_i"],
                                              data["start_j"], data["end_j"],
                                              data["start_k"], data["end_k"]]

    broader = 5
    file_counter = 0
    count_no_window = 0

    gz_files = glob.glob(folder + "\\*")
    for gz in gz_files:
        file_counter += 1
        if file_counter % 50 == 0:
            print(f"Handling no.{file_counter} file.")

        scan_name = gz.split("\\")[-1].replace(".nii.gz", "")
        i0, i1, j0, j1, k0, k1 = appendix_window_d[scan_name]
        if k0 == -1:
            count_no_window += 1
            continue

        i0, i1 = i0 - broader, i1 + broader
        j0, j1 = j0 - broader, j1 + broader
        k0, k1 = k0 - 1,       k1 + 1

        nii_file = nib.load(gz)
        nii_affine = nii_file.affine
        nii_meta_data = nii_file.header
        nii_data = nii_file.get_fdata().astype(np.int16)
        nii_data = nii_data[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]

        scan_file_name = f"{scan_name}_cropped.nii.gz"
        cropped_nii  = nib.Nifti1Image(nii_data,  nii_affine,  nii_meta_data)

        nib.save(cropped_nii, f"{folder}\\{scan_file_name}")
        os.remove(gz)

    print(f"File counts with now window: {count_no_window}")


if __name__ == "__main__":
    # nnU_Net_nii_cropper_train_val()
    # nnU_Net_nii_cropper_test()
    # crop_for_classification_train_val()
    # crop_for_classification_test
    pass
