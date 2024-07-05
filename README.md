# brain_tumor_detection

Click here to launch the model
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb)

https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb

---
## Overview

This project is focused on detecting brain tumors from MRI images using deep learning techniques. The main objective is to develop a reliable model that can assist in the early diagnosis of brain tumors, improving patient outcomes.

## Project Structure

- **Data**: MRI images of brain tumors, preprocessed and split into training and testing datasets.
- **Model**: A Convolutional Neural Network (CNN) designed to classify images as either having a brain tumor or being healthy.
- **Scripts**: Python scripts for data preprocessing, model training, evaluation, and prediction.
- **Notebooks**: Jupyter notebooks that document the exploratory data analysis (EDA) and model development process.
- **Results**: Evaluation metrics and model performance visualizations.

## Dataset

The dataset used in this project contains MRI images of brain tumors. The images are preprocessed and labeled for training the model. 
1 yes- This folder contains the MRI scans that have a tumor
2 no- This folder contains the MRI scans that do not have a tumor.
3 pred- This folder contains unlabelled MRI scans for testing purposes.

## Machine Learning Model

The project utilizes a Convolutional Neural Network (CNN); RESNET-50, for image classification. The CNN architecture is designed to effectively capture spatial features in the MRI images, making it suitable for detecting tumors. Key components of the model include:
- Convolutional layers for feature extraction
- Max pooling layers for down-sampling
- Fully connected layers for classification
- Softmax activation for output probabilities

## Tools and Technologies

- **TensorFlow** and **Keras**: For building and training the deep learning model.
- **OpenCV**: For image preprocessing and augmentation.
- **NumPy** and **Pandas**: For data manipulation and analysis.
- **Matplotlib** and **Seaborn**: For data visualization and plotting model performance.
- **Jupyter Notebook**: For interactive development and documentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MOAZ47/brain_tumor_detection.git
   cd brain_tumor_detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from here:
   ```bash
   https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri
   ```

## Usage

1. Preprocess the data:
   ```bash
   python preprocess_data.py
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

3. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```

4. Make predictions on new images:
   ```bash
   python predict.py --image path/to/image.jpg
   ```

## Results

The trained model achieves high accuracy in detecting brain tumors from MRI images. Below are some sample results:

- **Accuracy**: 96.3%
- **Precision**: High
- **Recall**: High
- **F1 Score**: High

## Results Comparison

### Model 1: [VGG16-based Brain Tumor Detection](https://github.com/MOAZ47/brain_tumor_detection/blob/master/Brain_Tumor.ipynb)

**Training Results:**
- **Epochs**: 10
- **Training Accuracy**:
  - Initial epoch: 72.80%
  - Final epoch: 97.00%
- **Validation Accuracy**:
  - Initial epoch: 85.50%
  - Final epoch: 92.80%
- **Training Loss**:
  - Initial epoch: 0.5821
  - Final epoch: 0.1096
- **Validation Loss**:
  - Initial epoch: 0.3325
  - Final epoch: 0.1802

### Model 2: [ResNet50-based Brain Tumor Detection](https://github.com/MOAZ47/brain_tumor_detection/blob/master/Brain_Tumor_ResNet.ipynb)

**Training Results:**
- **Epochs**: Up to 7 (Early Stopping enabled)
- **Training Accuracy**:
  - Initial epoch: 69.39%
  - Final epoch: 98.51%
- **Validation Accuracy**:
  - Initial epoch: 91.20%
  - Final epoch: 95.07%
- **Training Loss**:
  - Initial epoch: 0.5910
  - Final epoch: 0.0700
- **Validation Loss**:
  - Initial epoch: 0.2388
  - Final epoch: 0.1348

### Analysis

- **Training and Validation Performance**:
  - The ResNet50-based model shows better training and validation performance overall.
  - Early stopping in the second model helps in preventing overfitting, indicated by better validation accuracy and lower validation loss.

- **Efficiency**:
  - The ResNet50-based model converges faster, reaching high accuracy within 7 epochs, compared to 10 epochs for the VGG16-based model.
  - This indicates that the second approach is more efficient in terms of training time and achieving higher performance.

- **Model Architecture**:
  - ResNet50, known for its deeper architecture and residual connections, performs better than VGG16 in this context.
  - Using a single Dense layer on top of ResNet50 leverages the power of pre-trained weights, contributing to improved performance.

### Conclusion

The ResNet50-based approach with automated data handling and early stopping provides better results in terms of both training and validation metrics. It is also more efficient in achieving these results within fewer epochs. This approach is more automated and robust, leading to a streamlined workflow and enhanced model performance.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs or feature requests.

## Contact

For any questions or inquiries, please contact [Moaz](mailto:moazhusain47@gmail.com).
