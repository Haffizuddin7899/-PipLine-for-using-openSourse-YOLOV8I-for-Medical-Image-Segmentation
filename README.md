# -PipLine-for-using-openSourse-YOLOV8I-for-Medical-Image-Segmentation
# Introduction:
Medical image segmentation is a critical task in healthcare that enables precise identification and extraction of regions of interest, such as tumors, tissues, or organs. The segmentation of nuclei in histopathological images can provide essential insights into various diseases, including cancer. This article explores the use of the YOLOv8i model from the Ultralytics library, originally developed for object detection, to perform high-quality instance segmentation in medical images.

In this guide, we will walk through a complete pipeline for applying YOLOv8i to the segmentation of nuclei in tissue images from the NuInsSeg dataset. This pipeline covers dataset preparation, model training, inference, and result visualization. Through fine-tuning, the YOLOv8i model can deliver robust segmentation results, making it a promising tool for medical imaging tasks.

# Dataset Introduction:
The NuInsSeg dataset is an extensive collection of manually segmented nuclei extracted from histopathological images of human and mouse tissues. The dataset includes over 30,000 segmented nuclei across 31 organs and 665 image patches. Each image in the dataset is stained using Hematoxylin and Eosin (H&E) to highlight cellular structures, making it ideal for segmentation tasks that require precision in delineating cell boundaries.

Human Organs: brain (cerebellum, cerebrum), colon, epiglottis, kidney, liver, lung, pancreas, spleen, stomach, tongue, and more.
Mouse Organs: brain, colon, lung, stomach, testis, and more.

This dataset provides an excellent foundation for training medical image segmentation models, and its use in this pipeline will showcase the application of deep learning models, such as YOLOv8i, in medical research.

# Step-by-Step Explanation of the Pipeline:

# Step 1: Install Necessary Libraries
We begin by installing all the necessary libraries, including Ultralytics for YOLOv8, pycocotools for COCO-style datasets, and scikit-learn for data splitting.

!pip install -q ultralytics pycocotools scikit-learn matplotlib
# Step 2: Downloading and Unzipping the Dataset
We provide a function to download the dataset from a given URL and unzip it into a working directory. This dataset contains raw images and corresponding labeled masks for medical image segmentation.

import os
import requests
from zipfile import ZipFile

def download_and_unzip(url, save_path, extract_dir):
    print("Downloading assets...")
    file = requests.get(url)
    open(save_path, "wb").write(file.content)
    print("Download completed.")
    if save_path.endswith(".zip"):
        with ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction Done")

DATASET_URL = r"https://www.dropbox.com/scl/fi/u83yuoea9hu7xhhhsgp4d/Nuclei_Instance_Seg.zip?rlkey=7tw3vs4xh7rych4yq1xizd8mh&dl=1"
DATASET_DIR = "Nuclei-Instance-Dataset"
DATASET_ZIP_PATH = os.path.join(os.getcwd(), f"{DATASET_DIR}.zip")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR, exist_ok=True)
    download_and_unzip(DATASET_URL, DATASET_ZIP_PATH, DATASET_DIR)
    os.remove(DATASET_ZIP_PATH)
Step 3: Dataset Pruning and Preparation
The downloaded dataset contains multiple subdirectories, but we only need images and labeled masks. This function removes unnecessary subdirectories, keeping only essential data.

import shutil

def prune_subdirectories(base_dir, keep_dirs):
    for root_dir in os.listdir(base_dir):
        root_path = os.path.join(base_dir, root_dir)
        if os.path.isdir(root_path):
            for sub_dir in os.listdir(root_path):
                sub_path = os.path.join(root_path, sub_dir)
                if os.path.isdir(sub_path) and sub_dir not in keep_dirs:
                    shutil.rmtree(sub_path)

directories_to_keep = ['tissue images', 'mask binary without border', 'label masks modify']
prune_subdirectories(DATASET_DIR, directories_to_keep)
# Step 4: Pairing Images with Corresponding Masks
Next, we pair tissue images with their corresponding masks to prepare them for training and validation.

def get_image_mask_pairs(data_dir, limit=100):
    image_paths = []
    mask_paths = []
    for root, _, files in os.walk(data_dir):
        if 'tissue images' in root:
            for file in files:
                if file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
                    mask_paths.append(os.path.join(root.replace('tissue images', 'label masks modify'), file.replace('.png', '.tif')))
                if len(image_paths) >= limit:
                    break
            if len(image_paths) >= limit:
                break
    return image_paths, mask_paths

image_paths, mask_paths = get_image_mask_pairs(DATASET_DIR, limit=1200)
# Step 5: Processing and Converting Data into YOLO Format
The following function processes the image and mask pairs, converting them into YOLO's polygon format for instance segmentation.

import cv2
import numpy as np

def mask_to_polygons(mask, epsilon=1.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
            poly = contour.reshape(-1).tolist()
            if len(poly) > 4:
                polygons.append(poly)
    return polygons

def process_data(image_paths, mask_paths, output_images_dir, output_labels_dir):
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        shutil.copy(img_path, os.path.join(output_images_dir, os.path.basename(img_path)))

        unique_values = np.unique(mask)
        for value in unique_values:
            if value == 0:
                continue
            object_mask = (mask == value).astype(np.uint8) * 255
            polygons = mask_to_polygons(object_mask)

            with open(os.path.join(output_labels_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt'), 'w') as file:
                for poly in polygons:
                    normalized_poly = [str(coord) for coord in poly]
                    file.write(" ".join(normalized_poly) + "\n")
# Step 6: Training the YOLOv8i Model
Now, we use the processed dataset to train the YOLOv8i model.

from ultralytics import YOLO

model = YOLO("yolov8l-seg.yaml")
model.train(
    data="path_to_your_dataset_yaml",
    epochs=20,
    batch=4,
    project="yolov8_medical",
    name="yolov8_medical_experiment",
)
# Step 7: Visualizing Training Results

The training process generates various output files, including a results image that summarizes model performance.

This section displays these visual outputs, such as the results summary.

from IPython.display import Image

# Display the image
Image("/content/yolov8_l_dataset/results/20_epochs_limited/results.png")



# Results Summary
# Step 7: Inference on Validation Images
After training, the model can be used to make predictions on validation images.

inference_model = YOLO("best.pt")
inference_img_path = "/content/yolov8_l_dataset/val/images/sample_image.png"
inference_result = inference_model.predict(inference_img_path, conf=0.7, save=True)

# Visualizing result
inference_result_array = inference_result[0].plot()
plt.imshow(inference_result_array)
plt.show()



# Example Output
Conclusion:
This article presents a comprehensive pipeline for medical image segmentation using the YOLOv8i model. By adapting YOLOv8i to the NuInsSeg dataset, we demonstrate the power of deep learning in medical image analysis, specifically for nuclei segmentation tasks. The pipeline can be applied to various other medical image segmentation challenges, making it a flexible and robust solution for research and clinical applications.

# References:
Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.
Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention.
