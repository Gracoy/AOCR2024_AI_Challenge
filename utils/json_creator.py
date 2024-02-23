import glob
import json
import pandas as pd

"""
Create json file for data loader of classification model.
"""


def create_json_train_val():
    train_dataset_folder = "D:\\aocr2024\\ResNext3D\\train_classification_v6"
    val_dataset_folder = "D:\\aocr2024\\ResNext3D\\val_classification_v6"
    split_name = "train_val_classification_v6"

    # Create hashmap of scan_name: label
    label_csv = "D:\\aocr2024\\TrainValid_split.csv"
    start_end_df = pd.read_csv(label_csv)
    scan_label = {}
    for index, data in start_end_df.iterrows():
        scan_label[data["id"]] = data["scan-level label"]

    # Handle train dataset
    train_data = []
    train_gz_files = glob.glob(train_dataset_folder + "\\*.gz")
    train_sub_folder = train_dataset_folder.split("\\")[-1]
    for gz in train_gz_files:
        file_name = gz.split("\\")[-1]
        scan_name = file_name.replace(".nii.gz", "")
        train_data.append({"image": f"{train_sub_folder}/{file_name}",
                           "label": scan_label[scan_name]})

    # Handle val dataset
    val_data = []
    val_gz_files = glob.glob(val_dataset_folder + "\\*.gz")
    val_sub_folder = val_dataset_folder.split("\\")[-1]
    for gz in val_gz_files:
        file_name = gz.split("\\")[-1]
        scan_name = file_name.replace(".nii.gz", "")
        val_data.append({"image": f"{val_sub_folder}/{file_name}",
                         "label": scan_label[scan_name]})

    res = {"train": train_data, "val": val_data}
    with open(f"{split_name}.json", "w") as f:
        json.dump(res, f, sort_keys=True, indent=4)


def create_json_test():
    test_dataset_folder = "D:\\aocr2024\\ResNext3D\\test_classification_v6"
    split_name = "test_classification_v6"

    test_data = []
    test_gz_files = glob.glob(test_dataset_folder + "\\*.gz")
    sub_folder = test_dataset_folder.split("\\")[-1]
    for gz in test_gz_files:
        file_name = gz.split("\\")[-1]
        test_data.append({"image": f"{sub_folder}/{file_name}"})

    res = {"test": test_data}
    with open(f"{split_name}.json", "w") as f:
        json.dump(res, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    # create_json_train_val()
    # create_json_test()
    pass
