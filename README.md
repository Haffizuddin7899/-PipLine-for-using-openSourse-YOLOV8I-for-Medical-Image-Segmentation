# YOLOv8i Medical Image Segmentation
# Overview
This repository demonstrates the use of the YOLOv8i model for nuclei segmentation in histopathological tissue images. By fine-tuning the YOLOv8i instance segmentation model from the Ultralytics library, the project provides an efficient pipeline for medical image segmentation tasks. The dataset used is NuInsSeg, consisting of manually segmented nuclei from human and mouse tissue samples stained with Hematoxylin and Eosin (H&E).

The pipeline includes dataset preparation, model training, inference, and visualization of results, making it a comprehensive solution for researchers and developers working in the field of medical imaging.

# Features
1. Dataset downloading and preparation from the NuInsSeg dataset.
2. Conversion of images and masks into YOLO-compatible polygon format for instance segmentation.
3. Fine-tuning of the YOLOv8i model for accurate medical image segmentation.
4. Visualization of results for both training and inference stages.
# Dataset
The NuInsSeg dataset used in this project consists of over 30,000 segmented nuclei across 31 human and mouse organs, providing a comprehensive foundation for training medical segmentation models. The dataset contains both tissue images and corresponding labeled masks in TIFF format.

# Some organs included:

Human organs: Brain (cerebellum, cerebrum), colon, kidney, liver, pancreas, spleen, lung, and more.
Mouse organs: Brain, colon, lung, testis, stomach, and more.
Requirements
Python 3.x
PyTorch
Ultralytics YOLO
OpenCV
Scikit-learn
Matplotlib
Pycocotools
Installation
# Clone the repository:
git clone https://github.com/your-username/YOLOv8i-Medical-Segmentation.git
cd YOLOv8i-Medical-Segmentation
# Install the required Python packages:
pip install -r requirements.txt
Download and unzip the dataset:

The dataset will be automatically downloaded and unzipped when running the pipeline.
DATASET_URL = r"https://www.dropbox.com/scl/fi/u83yuoea9hu7xhhhsgp4d/Nuclei_Instance_Seg.zip?rlkey=7tw3vs4xh7rych4yq1xizd8mh&dl=1"
Pipeline
# 1. Dataset Preparation
The dataset consists of tissue images and corresponding masks. To start, the script will:

Download the dataset and extract the necessary files.
Prune unnecessary subdirectories to keep only the essential images and masks.
Pair images with corresponding masks for the next stage.
# 2. Data Processing
The images and masks are converted into YOLO format by extracting polygon annotations from masks and saving them into YOLO-compatible text files. This prepares the data for training.

# 3. Model Training
You can train the YOLOv8i model on the processed dataset:


from ultralytics import YOLO

model = YOLO("yolov8l-seg.yaml")
model.train(
    data="path_to_your_dataset_yaml",
    epochs=20,
    batch=4,
    project="yolov8_medical",
    name="yolov8_medical_experiment"
)
# 4. Inference and Visualization
After training, the model can be used for inference on new images. The results are visualized by plotting the segmented regions.


inference_model = YOLO("best.pt")
inference_img_path = "/path/to/validation/image.png"
inference_result = inference_model.predict(inference_img_path, conf=0.7, save=True)
# Results
The training process generates output files summarizing the model's performance. You can visualize the training and inference results using matplotlib or directly view the result images from the model output directory.

# Example Output
Training visualization:

# Inference output: Example output showing the segmented regions on a tissue image.

# Conclusion
This project demonstrates the effective use of the YOLOv8i model for nuclei segmentation in histopathological images. The pipeline can be extended and adapted to other medical image segmentation challenges, providing a robust and flexible framework for both research and clinical applications.

# References
Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.
Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
