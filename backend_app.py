import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import base64
import os

app = FastAPI()

# Allow connection from Streamlit/React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model Architecture (ResNet18)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
# CRITICAL: This must match your trained model (1 output)
model.fc = nn.Linear(in_features, 1) 

# Load Weights
model_path = "fundus_pytorch_model.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: {model_path} not found.")

model = model.to(device)
model.eval()

# Transforms (Must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- GRAD-CAM HELPER ---
def get_gradcam(model, img_tensor, orig_img):
    gradients = []
    activations = []
    
    def b_hook(m, gi, go): gradients.append(go[0])
    def f_hook(m, i, o): activations.append(o)
    
    # Ensure we are hooking into the correct layer for ResNet18
    layer = model.layer4[-1].conv2
    h1 = layer.register_forward_hook(f_hook)
    h2 = layer.register_full_backward_hook(b_hook)
    
    # Forward
    out = model(img_tensor)
    
    # Backward
    loss = out[0]
    model.zero_grad()
    loss.backward()
    
    h1.remove()
    h2.remove()
    
    # Safety check for empty gradients
    if not gradients or not activations:
        return np.array(orig_img)

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (orig_img.size[0], orig_img.size[1]))
    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig_bgr = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.5, heatmap, 0.5, 0)
    
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Initialize variables to None to prevent UnboundLocalError
        prob = 0.0
        gradcam_base64 = None

        # 1. Prediction
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
        
        # 2. Grad-CAM Generation
        # We removed the premature 'return' statement that was here.
        try:
            with torch.enable_grad():
                gradcam_img = get_gradcam(model, img_tensor, image)
            
            # Encode to base64
            _, buffer = cv2.imencode(".png", cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
            gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as cam_error:
            print(f"Grad-CAM Error: {cam_error}")
            # If Grad-CAM fails, we can still return the prediction without crashing
            gradcam_base64 = None

        # 3. Final Return
        return {
            "prediction_probability": prob, 
            "gradcam_image_base64": gradcam_base64
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    