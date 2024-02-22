# AOCR2024_AI_Challenge
Colab notebooks and utilities for AOCR2024 AI Challenge

# How to use
1. Use "extract_appendix_bounding_box_from_gt()" in "extract_appendix_window.py" to obatin label 3D bounding box coordinates and record as csv file.
2. Crop the train data and ground truth label corresponding to the 3D bounding box by "crop_for_segmentation_train_val()" in "nii_cropper.py".
3. Change the file format and create json file to meet requirement of nnU-Net by "prepare_data()" in "file_sorter.py"
4. Run note book of nnU-Net to train segmentation model and inference.
5. 
