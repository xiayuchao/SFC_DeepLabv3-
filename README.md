# DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution in PyTorch

---

## 目录
1. [项目简介](#项目简介)
2. [文件说明](#文件说明)
3. [使用说明](#使用说明)
4. [参考资料](#参考资料)

---

## 项目简介
DeepLabv3+ 是一种基于 Atrous Separable Convolution 的语义分割模型，具有强大的多尺度特征提取能力和高效的解码器模块。

---
## 文件说明
### 1. `add_modules`
论文改进模型所添加的模块

### 2. `attention`
论文改进模型所添加的注意力模块

### 3. `logs/`
存储训练过程中生成的模型权重文件和日志文件，包括最佳权重、最后一轮权重。

### 4. `model_data/`
存放预训练模型文件，用于加载权重加速训练或直接用于预测。

### 5. `nets/`
- **deeplab.py**：DeepLabv3+的整体实现文件，包括模型定义和前向传播逻辑。
- **mobilenetv2.py** 和 **xception.py**：实现两种主干网络。
- **aspp.py**：实现ASPP模块，用于多尺度特征提取。
- **decoder.py**：实现解码器模块，用于恢复空间分辨率。

### 6. `utils/`
- **metrics.py**：提供语义分割评估指标的实现。
- **lr_scheduler.py**：实现学习率动态调整策略。
- **data_loader.py**：定义数据集加载流程和数据增强操作。
- **visualizer.py**：可视化预测结果和分割图像。

### 7. `VOCdevkit/`
- **VOC2007**：存储训练和测试所需的图片和标签，需符合VOC格式。
  - **ImageSets/**：存储数据划分的文件（.txt格式）
  - **JPEGImages/**：存储原始图片文件。
  - **SegmentationClass/**：存储分割任务的标签文件（PNG格式，每个像素值表示类别）。

### 8. `train.py`
用于模型训练，支持多种优化器和学习率调整策略，需配置`num_classes`和`model_path`。

### 9. `predict.py`
用于加载训练好的模型进行预测，支持单张图片预测、文件夹批量预测、FPS测试和视频检测。

### 10. `get_miou.py`
计算分割模型的评估指标，包括mIoU、Precision等，需提前配置类别名称和类别数量。

### 11. `voc_annotation.py`
将VOC格式数据集转换为训练所需的格式，生成用于训练的txt文件。

### 12. `requirements.txt`
列出所有依赖的Python库和版本信息，确保环境配置一致。

---

## 使用说明

### 训练步骤
1. 配置VOC数据集：将图片存放于`VOCdevkit/VOC2007/JPEGImages/`，标签存放于`VOCdevkit/VOC2007/SegmentationClass/`。
2. 运行`voc_annotation.py`生成对应的训练txt文件。
3. 修改`train.py`中的`num_classes`、`model_path`和`backbone`为对应值。
4. 运行`train.py`开始训练。

### 预测步骤
1. 修改`predict.py`中的`model_path`、`num_classes`和`backbone`。
2. 运行`predict.py`，输入图片路径进行预测。

### 评估步骤
1. 修改`get_miou.py`中的`num_classes`和类别名称。
2. 运行`get_miou.py`计算评估指标。

---

## 参考资料
- [Pytorch Segmentation](https://github.com/ggyyzm/pytorch_segmentation)
- [Keras DeepLabv3+](https://github.com/bonlime/keras-deeplab-v3-plus)

