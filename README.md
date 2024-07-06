# Brain_Tumor_Detection

Click here to launch the model
<!--[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb)-->

<!--https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb-->

https://braintumordetection-kjgrvpu7gcwrhxen3zu43d.streamlit.app/

---
## Overview

This project is focused on detecting brain tumors from MRI images using deep learning techniques. The main objective is to develop a reliable model that can assist in the early diagnosis of brain tumors, improving patient outcomes.

I have also worked on the comparison between VGG16 and RESNET-50 architecture. The results are shown below

## Project Structure

- **Data**: MRI images of brain tumors, preprocessed and split into training and testing datasets.
- **Model**: A Convolutional Neural Network (CNN) designed to classify images as either having a brain tumor or being healthy.
- **Notebooks**: Jupyter notebooks that document the exploratory data analysis (EDA) and model development process. Here is the breakdown of the files:
   - **Brain_Tumor_Comparison.ipynb**: Notebook developed for comparison of VGG16 model and Resnet 50.
   - **Brain_Tumor_web_app.ipynb**: Developed to turn notebook into a web application.
- **Results**: Evaluation metrics and model performance comparison.

## Dataset

The dataset used in this project contains MRI images of brain tumors. The images are preprocessed and labeled for training the model. Here is the folder structure of the data:
1. yes- This folder contains the MRI scans that have a tumor
2. no- This folder contains the MRI scans that do not have a tumor.
3. pred- This folder contains unlabelled MRI scans for testing purposes.

Data can be found [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

### Models

**Augmentations**:
- Resizing --> 256x256
- RandomResizedCrop --> 224
- RandomHorizontalFlip
- Normalize

**VGG16-based Model**

The VGG16 model serves as the foundation, enhanced with additional layers for fine-tuning. Key steps include:

- **Model Architecture:** Utilizes the VGG16 base with additional Dense layers tailored for classification.
- **Fine tuning:** Froze all except the last two layers.
- **Training:** Trained for 25 epochs with early stopping and model checkpointing.

**ResNet50-based Model**

The ResNet50 model is chosen for its deep architecture featuring residual connections for improved gradient flow. Key steps include:

- **Model Architecture:** ResNet50 forms the core architecture, supplemented with a Dense layer for classification.
- **Fine tuning:** Froze all except the last two layers.
- **Training:** Trained with early stopping, achieving faster convergence for 25 epochs.

### Tools and Technologies

- **Library:** PyTorch for model development and training.
- **Image Processing:** PyTorch torchvision module for image preprocessing and augmentation.
- **Data Handling:** NumPy and Pandas for efficient data manipulation and analysis.
- **Visualization:** Matplotlib and Seaborn for visualizing data and plotting model performance.
- **Development Environment:** Jupyter Notebook for interactive development and documentation.

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

## Results

The trained model achieves high accuracy in detecting brain tumors from MRI images. Below are some sample results:

- **Accuracy for RESNET 50**: 60.31%
- **Accuracy for VGG16**: 57.94%

## Model Evaluation Comparison

### Model Comparison Table

| Metric                      | VGG16                           | ResNet50                        |
|-----------------------------|---------------------------------|---------------------------------|
|**Epochs to Early Stopping** | 19                              | 17                              |
|**Training Loss**            | 0.7932                          | 0.8290                         |
|**Validation Loss**          | 1.1328                          | 1.1147                          |
|**Training Accuracy**        | 0.6827 (69.27%)                 | 0.7183 (71.83%)                 |
| **Validation Accuracy**     | 0.5794 (57.94%)                 | 0.6031 (60.31%)                 |
| **Precision**               | 0.7047 (70.47%)                 | 0.7372 (73.72%)                 |
| **Recall**                  | 0.7054 (70.54%)                 | 0.7379 (73.79%)                 |
| **F1 Score**                | 0.6904 (69.04%)                 | 0.7204 (72.04%)                 |


### Interpretation:

- ResNet achieved slightly better validation accuracy (0.6031) compared to VGG (0.5794). This suggests that ResNet might generalize better to unseen data.
- VGG trained a bit faster, reaching early stopping in 19 epochs compared to ResNet's 17 epochs. This could be due to the simpler architecture of VGG.
- Both models show a similar trend of decreasing training loss and increasing training accuracy over epochs. This indicates successful learning from the training data.

### Implications:

- ResNet might be a better choice if validation accuracy is the primary concern.
- If training time is a constraint, VGG might be a viable alternative with a slight trade-off in accuracy.


### Conclusion

- ResNet outperformed VGG in terms of validation accuracy, suggesting better generalization potential.
- ResNet appears to have achieved better validation accuracy (around 0.7) compared to VGG (around 0.58) based on the peaks of their validation accuracy curves. ResNet's training loss seems to have plateaued at a slightly higher value than VGG's, potentially indicating slightly less optimal learning, but this could be due to factors like different learning rates used in training.
- VGG trained slightly faster, potentially due to its simpler architecture.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs or feature requests.

## Contact

For any questions or inquiries, please contact [Moaz](mailto:moazhusain47@gmail.com).
