
# Person Segmentation

This demo shows how to train and run a Semantic Segmentation network for the "Person vs. Background" task.
The model is trained on a subset of the COCO segmentation dataset (http://cocodataset.org/), containing "person" class.
The network architecture is made from scratch and is inspired be U-Net and DeepLab v3 (ASPP) architectures

## Jupyter notebooks

1. `person-segmentation.ipynb`
    * All in one notebook: load data, create model, run training, run inference
2. `post-processing.ipynb`
    * Playground for segmentation post processing (applications)
    * Emulating DOF effect and background switch
2. `kaggle_submission.ipynb`
    * Run all test images through a model
    * Generate a kaggle submission file (convers all segmentation maps to CSV using RLE)

## Python scripts

The scripts contain approximatelly the same code, as in `person-segmentation.ipynb`

1. `data.py`
    * Dataset classes for preparation fo training and validation data pipelines
2. `model.py`
    * Definition of the Model class, defining the network architecture
3. `train.py`
    * Execute this script to run the training procedure
4. `inference.py`
    * Execute this script to run the inference on a trained model
