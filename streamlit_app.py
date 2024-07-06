import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import warnings

warnings.filterwarnings('ignore')

num_classes = 4

# Load the trained ResNet-50 model
@st.cache(allow_output_mutation = True)
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
    model.eval()
    return model

# Define the image transformations
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Make a prediction
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item(), torch.nn.functional.softmax(outputs, dim=1).max().item()

# Load class names
@st.cache
def load_class_names(json_path):
    with open(json_path) as f:
        class_names = json.load(f)
    return class_names

# Main function to run the Streamlit app
def main():
    st.title("Image Classification with ResNet-50")

    model_path = 'resnet_best_model.pth'  # Path to your saved model
    json_path = 'class_names.json'  # Path to the JSON file containing class names

    # Load the model and class names
    model = load_model(model_path)
    class_names = load_class_names(json_path)

    st.header("Upload an image to classify")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        image = transform_image(image)
        predicted_class, confidence = predict(model, image)

        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
