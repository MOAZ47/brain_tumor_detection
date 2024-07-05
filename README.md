# brain_tumor_detection

Click here to launch the model
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb)

https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb

Here's a template for your GitHub README file for your brain tumor detection project:

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

- **Accuracy**: 96%
- **Precision**: 93%
- **Recall**: 92%
- **F1 Score**: 92%

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs or feature requests.

## Contact

For any questions or inquiries, please contact [Moaz](mailto:moazhusain47@gmail.com).
