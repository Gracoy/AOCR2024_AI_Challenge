import shutil
import glob
import os
import pandas as pd
import json


"""
This file is used to format the file names and create json file for requirement 
of nnU-Net and record the original file name to csv for further usage.
"""


def prepare_data():
    base_directory = "D:\\aocr2024\\nnU_Net"
    copy_and_rename(base_directory)
    create_nnU_Net_json()


def copy_and_rename(base_dir):
    """
    This program is used to copy and rename the train and label file
    to meet the format requirement of nnU-Net and store file names to csv.
    """
    os.chdir(base_dir)
    image_path, label_path = "imagesTr", "labelsTr"
    maybe_mkdir(image_path)
    maybe_mkdir(label_path)

    # directory of training dataset, the train data and label data are put in same directory
    source_folder = "cropped_train_data"
    gz_files = glob.glob(source_folder + "\\*")
    scan_name_map = {}
    # you can name the task name whatever you want
    task_name = "Appendix"

    file_counter = 0
    for gz in gz_files:
        # if we meet label file, skip it
        if gz.endswith("_label.nii.gz"):
            continue

        file_counter += 1
        if file_counter % 20 == 0:
            print(f"Handling no.{file_counter} file.")

        # replace suffix of nii file to handle the label nii file
        # Ex: train file name --> abc123456.nii.gz, then it's corresponding label file --> abc123456_label.nii.gz
        mask = gz.replace(".nii.gz", "_label.nii.gz")

        # here we store the map of new name and original name
        scan_name = gz.split("\\")[-1].replace(".nii.gz", "")
        new_name = f"{task_name}_{str(file_counter).zfill(3)}"
        scan_name_map[new_name] = scan_name

        # copy train and label files and rename to corresponding directories
        shutil.copy(gz,   f"{image_path}\\{new_name}_0000.nii.gz")
        shutil.copy(mask, f"{label_path}\\{new_name}.nii.gz")

    # save the file name map we store to csv file for further recovery
    df_scan_name = pd.DataFrame.from_dict(scan_name_map, orient="index", columns=["sample_id"])
    df_scan_name.index.rename("scan_id", inplace=True)
    df_scan_name.to_csv(f"scan_sample_name_map.csv")


def create_nnU_Net_json():
    """
    Create the json file for nnU-Net
    """
    json_name = "dataset.json"
    json_d = {}
    json_d["channel_names"] = {"0": "CT"}
    json_d["labels"] = {"background": 0, "appendix": 1, "fat_stranding": 2}
    json_d["numTraining"] = len(os.listdir("imagesTr"))
    json_d["file_ending"] = ".nii.gz"
    json_d["overwrite_image_reader_writer"] = "SimpleITKIO"

    with open(json_name, 'w') as f:
        json.dump(json_d, f, indent=4, sort_keys=False)


def maybe_mkdir(dir_path):
    """
    Just a utility to create folder
    """
    if os.path.exists(dir_path):
        print(f"Directory {dir_path} already exits.")
    else:
        os.mkdir(dir_path)
        print(f"Create {dir_path} success.")


def rename_test_file():
    """
    This program is used to copy and rename the test dataset for inference
    to meet the format requirement of nnU-Net and store file names to csv.
    """
    # directory of test dataset
    base_directory = "D:\\aocr2024\\nnU_Net\\cropped_test_data"
    gz_files = glob.glob(base_directory + "\\*")
    # you can name the task name whatever you want
    task_name = "Appendix"

    file_counter = 0
    scan_name_map = {}
    for gz in gz_files:
        file_counter += 1
        scan_name = gz.split("\\")[-1].replace(".nii.gz", "")
        new_name = f"{task_name}_{str(file_counter).zfill(3)}"
        scan_name_map[scan_name] = new_name
        os.rename(gz, f"{base_directory}\\{new_name}_0000.nii.gz")

    df_scan_name = pd.DataFrame.from_dict(scan_name_map, orient="index", columns=["sample_id"])
    df_scan_name.index.rename("scan_id", inplace=True)
    df_scan_name.to_csv(f"scan_test_name_map.csv")


def recover_to_original_name():
    """
    This program recover the test data to original name for subsequent classification task.
    """
    target_directory = "D:\\Dataset779_Appendix3labels_seg_result_pp_cropped"
    gz_files = glob.glob(target_directory + "\\*")

    scan_name_df = pd.read_csv("D:\\aocr2024\\nnU_Net\\scan_sample_name_map.csv")
    scan_name_map = {}
    for index, data in scan_name_df.iterrows():
        scan_name_map[data["sample_id"]] = data["scan_id"]

    for idx, gz in enumerate(gz_files):
        sample_name = gz.split("\\")[-1].replace("_pp.nii.gz", "")
        scan_name   = scan_name_map[sample_name] + ".nii.gz"
        os.rename(gz, f"{target_directory}\\{scan_name}")


if __name__ == "__main__":
    prepare_data()
    pass
