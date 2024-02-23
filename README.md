# AOCR2024_AI_Challenge
Colab notebooks and data pre-processing utilities of AOCR2024 AI Challenge.

# Pipeline :
Use nnU-Net to find appendix location and extract the slice-level labels,
then use the segmentation results to train ResNeXt3D to inference scan-level labels.

# How to use :
## Segmentation task
1. Use "extract_appendix_window.py" to obatin 3D bounding box coordinates of ground truth labels.
2. Crop the train data, train label and test files corresponding to the 3D bounding box by "nii_cropper.py".
3. Change the file format and create json file to meet requirement of nnU-Net by "file_sorter.py"
4. Run notebook "nnU-Net.ipynb" to train nnU-Net segmentation model and inference test data.
5. Use "seg_post_process.py" to optimize segmentation results of nnU-Net.
6. Run "get_slice_label.py" to iterate through post-processed segmentation results and obtain slice-level labels.

## Calssification task
7. Use "extract_appendix_window.py" to obatin 3D bounding box coordinates of segmentation results.
8. Crop the segmentation results corresponding to the 3D bounding box by "nii_cropper.py".
9. Create json files for train/val and test split respectively.
10. Run notebook "ResNeXt_3D.ipynb" to train RexNext3D classification models.
11. Run notebook "ResNeXt_3D_inference.ipynb" to inference test data and obtain scan-lavel labels.
12. Combine slice-level labels obtained from step.6 and scan-level labels obtained from step.11 for final results.
