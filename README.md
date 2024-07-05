# Brain_Tumor_Detection

Click here to launch the model
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb)

https://mybinder.org/v2/gh/MOAZ47/brain_tumor_detection/HEAD?urlpath=%2Fvoila%2Frender%2FBrain_Tumor_web_app.ipynb

---
## Overview

This project is focused on detecting brain tumors from MRI images using deep learning techniques. The main objective is to develop a reliable model that can assist in the early diagnosis of brain tumors, improving patient outcomes.

I have also worked on the comparison between VGG16 and RESNET-50 architecture. The results are shown below

## Project Structure

- **Data**: MRI images of brain tumors, preprocessed and split into training and testing datasets.
- **Model**: A Convolutional Neural Network (CNN) designed to classify images as either having a brain tumor or being healthy.
- **Notebooks**: Jupyter notebooks that document the exploratory data analysis (EDA) and model development process. Here is the breakdown of the files:
   - **Brain_Tumor_Comparison6.ipynb**: Notebook developed for comparison of VGG16 model and Resnet 50.
   - **Brain_Tumor_web_app.ipynb**: Developed to turn notebook into a web application.
- **Results**: Evaluation metrics and model performance comparison.

## Dataset

The dataset used in this project contains MRI images of brain tumors. The images are preprocessed and labeled for training the model. Here is the folder structure of the data:
1. yes- This folder contains the MRI scans that have a tumor
2. no- This folder contains the MRI scans that do not have a tumor.
3. pred- This folder contains unlabelled MRI scans for testing purposes.

Data can be found [here](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri)

### Models

**VGG16-based Model**

The VGG16 model serves as the foundation, enhanced with additional layers for fine-tuning. Key steps include:

- **Data Preprocessing:** Images are resized to 256x256 and augmented with scaling, rotation, and lighting adjustments.
- **Model Architecture:** Utilizes the VGG16 base with additional Dense layers tailored for classification.
- **Training:** Trained for 10 epochs with early stopping and model checkpointing.

**ResNet50-based Model**

The ResNet50 model is chosen for its deep architecture featuring residual connections for improved gradient flow. Key steps include:

- **Data Preprocessing:** Similar preprocessing steps as VGG16, including image resizing and augmentation.
- **Model Architecture:** ResNet50 forms the core architecture, supplemented with a Dense layer for classification.
- **Training:** Trained with early stopping, achieving faster convergence.

### Tools and Technologies

- **Frameworks:** TensorFlow, Fast.ai, and Keras for model development and training.
- **Image Processing:** OpenCV for image preprocessing and augmentation.
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

- **Accuracy for RESNET 50**: 77.0%
- **Accuracy for VGG16**: 99.0%

## Model Evaluation Comparison

### Model Comparison Table

| Metric          | VGG16                           | ResNet50                        |
|-----------------|---------------------------------|---------------------------------|
| **Validation Loss** | 0.2414                         | 9.8260                          |
| **Accuracy**    | 0.9900 (99.00%)                 | 0.7700 (77.00%)                 |
| **Precision**   | 0.9899 (98.99%)                 | 0.7747 (77.47%)                 |
| **Recall**      | 0.9899 (98.99%)                 | 0.7592 (75.92%)                 |
| **F1 Score**    | 0.9899 (98.99%)                 | 0.7669 (76.69%)                 |

### Interpretation

- **Validation Loss**: The VGG16 model has a much lower validation loss compared to the ResNet50 model, indicating better performance in minimizing loss on the validation set.
- **Accuracy**: The VGG16 model achieves significantly higher accuracy than the ResNet50 model, suggesting better performance in correctly classifying the validation samples.
- **Precision**: The VGG16 model shows higher precision, indicating better identification of the positive class.
- **Recall**: The VGG16 model also has higher recall, meaning it captures more true positives.
- **F1 Score**: The VGG16 model has a better balance between precision and recall compared to the ResNet50 model.


### Conclusion

Based on the evaluation metrics, the VGG16 model outperforms the ResNet50 model across all measured criteria. The VGG16 model achieves a significantly lower validation loss of 0.2414 compared to 9.8260 for ResNet50, indicating superior performance in minimizing prediction errors on unseen data. Moreover, VGG16 achieves a higher accuracy of 99.00% versus 77.00% for ResNet50, demonstrating its effectiveness in correctly classifying brain tumor images. The precision, recall, and F1 score metrics further confirm VGG16's superiority, with values of approximately 98.99% each, compared to ResNet50's metrics ranging around 76-77%.

In conclusion, the VGG16 model, with its deeper architecture and effective feature extraction capabilities, proves to be the preferred choice for brain tumor classification tasks based on this evaluation. Its robust performance underscores its suitability for medical imaging applications where high accuracy and reliable predictions are paramount.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs or feature requests.

## Contact

For any questions or inquiries, please contact [Moaz](mailto:moazhusain47@gmail.com).
