# DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution in PyTorch

## Project Overview
DeepLabv3+ is a semantic segmentation model based on Atrous Separable Convolution, designed to capture multi-scale features effectively while maintaining an efficient decoding module. The model enhances the segmentation performance by combining deep feature extraction with a powerful decoder, which recovers high spatial resolution for accurate pixel-level classification.

## File Description
### 1. `add_modules`
This directory contains the additional modules introduced in the paper to enhance the model's performance, particularly focusing on improving the feature extraction and attention mechanisms.

### 2. `attention`
Contains modules for implementing attention mechanisms added to the base model to improve the model's ability to focus on important regions of the input image.

### 3. `logs/`
Stores model weights and training logs generated during the training process, including the best weights and the final weights at the end of the training.

### 4. `model_data/`
Contains pre-trained model files that can be used for initializing the network to speed up training or directly used for inference.

### 5. `nets/`
- **deeplab.py**: This is the main implementation file for DeepLabv3+, including model definition and forward propagation logic.
- **mobilenetv2.py** and **xception.py**: These files implement two backbone networks used for feature extraction.
- **aspp.py**: Implements the Atrous Spatial Pyramid Pooling (ASPP) module, used for multi-scale feature extraction.
- **decoder.py**: Implements the decoder module responsible for recovering the spatial resolution of the features.

### 6. `utils/`
- **metrics.py**: Provides implementations for evaluation metrics used in semantic segmentation tasks.
- **lr_scheduler.py**: Implements dynamic learning rate adjustment strategies during training.
- **data_loader.py**: Defines the dataset loading pipeline and data augmentation operations.
- **visualizer.py**: Includes methods to visualize predictions and segmentation results.

### 7. `VOCdevkit/`
- **VOC2007**: Contains images and labels required for training and testing, following the VOC format.
  - **ImageSets/**: Contains files that define the dataset splits (in `.txt` format).
  - **JPEGImages/**: Stores raw image files.
  - **SegmentationClass/**: Contains the label files for segmentation tasks (in PNG format, with each pixel value corresponding to a class).

### 8. `train.py`
This file is used for model training and supports multiple optimizers and learning rate strategies. You need to configure parameters such as `num_classes` and `model_path` before running the training.

### 9. `predict.py`
Used for inference with a trained model. It supports predicting a single image, a folder of images, FPS testing, and video detection.

### 10. `get_miou.py`
This script calculates evaluation metrics for segmentation models, including mIoU (Mean Intersection over Union), Precision, and others. Ensure to configure the class names and class count properly before running.

### 11. `voc_annotation.py`
Converts the VOC dataset into the required format for training, generating the necessary `.txt` files for training.

### 12. `requirements.txt`
Lists all the Python libraries and their versions that are required to ensure a consistent environment configuration for running the project.

## Usage Instructions

### Training Process
1. Set up the VOC dataset by placing the images in `VOCdevkit/VOC2007/JPEGImages/` and the labels in `VOCdevkit/VOC2007/SegmentationClass/`.
2. Run `voc_annotation.py` to generate the corresponding training `.txt` files.
3. Modify `train.py` to configure parameters such as `num_classes`, `model_path`, and the desired backbone model (e.g., MobileNetV2 or Xception).
4. Run `train.py` to begin the training process.

### Inference Process
1. Modify the `predict.py` script by specifying the correct `model_path`, `num_classes`, and the backbone model to be used.
2. Run `predict.py` to make predictions by inputting the image path. The script can predict a single image or batch process a folder of images.

### Evaluation Process
1. Modify the `get_miou.py` script to specify the correct `num_classes` and class names.
2. Run `get_miou.py` to calculate the evaluation metrics, which include mIoU, Precision, and others.

## References
- [Pytorch Segmentation](https://github.com/ggyyzm/pytorch_segmentation)
- [Keras DeepLabv3+](https://github.com/bonlime/keras-deeplab-v3-plus)

---

This version of the README now contains the full English translation and expanded explanations while maintaining the structure and key information.
