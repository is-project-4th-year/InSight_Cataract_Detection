# model_utils.py (FINAL - EfficientNetV2 + Grad-CAM)
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import models
import numpy as np
import cv2 # OpenCV is needed for Grad-CAM

# --- CONSTANTS ---
IMG_SIZE = (224, 224)
MODEL_PATH = "cataract_model_v3.keras" # Use the new V3 model
LAST_CONV_LAYER_NAME = "top_activation" # Target for EfficientNetV2

# --- FUNCTION DEFINITIONS ---

@st.cache_resource
def load_model(path):
    """
    Loads the trained Keras model (v3). Load without compiling.
    Includes debug summaries.
    """
    print(f"Loading model from {path}...")
    model = tf.keras.models.load_model(path, compile=False)
    print("Model loaded successfully.")

    # --- DEBUGGING: Print model summaries ---
    print("\n--- Full Model Summary (EfficientNetV2) ---")
    model.summary(print_fn=lambda x: print(x))
    try:
        base_model_name = "efficientnetv2-b0" 
        base_model = model.get_layer(base_model_name)
        print(f"\n--- Base Model Summary ({base_model_name}) ---")
        base_model.summary(print_fn=lambda x: print(x))
        try:
             target_layer = base_model.get_layer(LAST_CONV_LAYER_NAME)
             print(f"\nSuccessfully found Grad-CAM target layer: {LAST_CONV_LAYER_NAME}")
        except ValueError:
             print(f"\nError: Could not find target Grad-CAM layer '{LAST_CONV_LAYER_NAME}'")
    except ValueError:
        print(f"\nError: Could not find base model layer named '{base_model_name}'.")
    # --- END DEBUGGING ---

    return model

def preprocess_image_for_prediction(image_bytes, img_size=IMG_SIZE):
    """
    Prepares image for EfficientNetV2.
    The model itself handles the 0-255 scaling.
    """
    image = tf.image.decode_image(image_bytes, channels=3)
    if image is None:
        raise ValueError("Could not decode image bytes.")

    image = tf.image.resize(image, img_size)
    original_resized_img = np.array(image).astype(np.uint8)
    image_for_model = tf.expand_dims(image, axis=0) 
    
    return image_for_model, original_resized_img

def get_grad_cam_overlay(model, preprocessed_img_tensor, original_resized_img, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    """
    Generates the Grad-CAM heatmap for EfficientNetV2.
    """
    
    try:
        base_model = model.get_layer("efficientnetv2-b0") 
        last_conv_layer = base_model.get_layer(last_conv_layer_name)

        grad_model = models.Model(
            inputs=model.inputs, 
            outputs=[last_conv_layer.output, model.output]
        )
    except Exception as e:
        raise ValueError(f"Error creating grad_model. Check layer names. Error: {e}")

    with tf.GradientTape() as tape:
        try:
            last_conv_output, preds = grad_model(preprocessed_img_tensor, training=False)
        except Exception as e:
             raise RuntimeError(f"Error during grad_model execution: {e}.")

        if preds is None: raise ValueError("Model prediction (preds) is None.")
        
        class_channel = preds[0][0] # preds shape is (1, 1)

    grads = tape.gradient(class_channel, last_conv_output)
    if grads is None: raise ValueError("Gradient calculation failed (grads is None).")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]

    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    max_heatmap = tf.math.reduce_max(heatmap)
    if max_heatmap <= 1e-5:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = tf.maximum(heatmap, 0) / max_heatmap

    heatmap = heatmap.numpy()

    # --- Overlaying ---
    try:
        heatmap_resized = cv2.resize(heatmap, (original_resized_img.shape[1], original_resized_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_resized_img, 0.7, heatmap_color, 0.3, 0)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error during heatmap processing: {e}.")

    return overlay, heatmap_color, float(preds[0][0])