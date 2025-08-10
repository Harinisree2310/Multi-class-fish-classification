# src/app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
# Import the specific preprocessing functions needed
# CORRECTED: Added vgg16 and resnet50 to the import statement
from tensorflow.keras.applications import vgg16, resnet50, mobilenet_v2, inception_v3, efficientnet

# --- Configuration ---
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.txt")
EVAL_SUMMARY_PATH = os.path.join(MODELS_DIR, "evaluation_summary.json")

# --- Model & Preprocessing Loading ---

# Define a map of model names to their preprocessing functions
# This removes the need to import from the training script.
PREPROCESS_MAP = {
    "VGG16": vgg16.preprocess_input,
    "ResNet50": resnet50.preprocess_input,
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "InceptionV3": inception_v3.preprocess_input,
    "EfficientNetB0": efficientnet.preprocess_input
}

@st.cache_resource
def load_model_and_preprocessing():
    """
    Loads the best model and determines its required preprocessing function.
    """
    if not os.path.exists(EVAL_SUMMARY_PATH):
        st.error(f"Evaluation summary not found at {EVAL_SUMMARY_PATH}. Please run evaluation.")
        st.stop()
    
    # Find the best model's filename from the evaluation summary
    with open(EVAL_SUMMARY_PATH) as f:
        summary = json.load(f)
    
    # Find the model with the highest f1_macro score
    best_model_filename = max(summary, key=lambda k: summary[k]['f1_macro'])
    backbone_name = best_model_filename.replace("_best.h5", "")
    
    st.sidebar.info(f"Using best model: **{best_model_filename}**")

    # Determine the correct preprocessing function
    preprocess_function = None
    if backbone_name in PREPROCESS_MAP:
        preprocess_function = PREPROCESS_MAP[backbone_name]
        print(f"[INFO] Using preprocessing for: {backbone_name}")
    elif backbone_name == 'cnn':
        # For the custom CNN, we define the simple rescale function
        def cnn_preprocess(x):
            return x / 255.0
        preprocess_function = cnn_preprocess
        print("[INFO] Using default rescale preprocessing for custom CNN.")
    else:
        st.error(f"Could not determine preprocessing for model {backbone_name}. Aborting.")
        st.stop()

    # Load the actual model file
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    return model, preprocess_function

@st.cache_data
def load_class_names():
    """Loads the class names from the text file."""
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Load resources
try:
    model, preprocess_input = load_model_and_preprocessing()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Failed to load model or class names. Please ensure training and evaluation have been run. Error: {e}")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Fish Image Classifier", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish and the model will predict its class.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image using the correct function for the loaded model
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) # Create a batch
        
        # Apply the correct preprocessing
        preprocessed_img = preprocess_input(img_array)

        # Prediction
        with st.spinner('Classifying...'):
            predictions = model.predict(preprocessed_img)
        
        score = tf.nn.softmax(predictions[0])
        predicted_index = np.argmax(score)
        predicted_class = class_names[predicted_index]
        confidence = np.max(score) * 100

        st.success(f"**Predicted Species:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
