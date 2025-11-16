import os
import io
import uuid
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from PIL import Image
import numpy as np
import cv2
import uvicorn
import base64 # <-- ADD THIS IMPORT
from dotenv import load_dotenv

# --- 1. Load Config ---
load_dotenv() # Load from .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
IMAGE_BUCKET = os.getenv("IMAGE_BUCKET", "images")
MODEL_PATH = os.getenv("MODEL_PATH", "fundus_pytorch_model.pt")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env file.")

# --- 2. FastAPI App Setup ---
app = FastAPI(title="Insight API (PyTorch + Fundus)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows Streamlit (http://localhost:8501)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Supabase & Model ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Backend] Using device: {device}")

# Load the ResNet18 model structure
model = resnet18() 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1) # Match the training script (output 1 logit)

# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval() # Set to evaluation mode
print(f"[Backend] PyTorch model loaded: {MODEL_PATH}")

# Image transforms (MUST match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. Grad-CAM Helper ---
def generate_gradcam(model, image_tensor, orig_image_pil):
    gradients = []
    activations = []

    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    target_layer = model.layer4[-1].conv2 # Target for ResNet18
    fwd_hook = target_layer.register_forward_hook(forward_hook)
    bwd_hook = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    loss = output[0, 0] # Backpropagate the single output neuron
    loss.backward(retain_graph=True) 
    fwd_hook.remove()
    bwd_hook.remove()

    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()
    weights = np.mean(grads, axis=(2, 3))[0, :]
    cam = np.zeros(acts.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights): cam += w * acts[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (orig_image_pil.width, orig_image_pil.height))
    if np.max(cam) > 0: cam = (cam - np.min(cam)) / np.max(cam)
    else: cam = np.zeros_like(cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay_img_cv = cv2.cvtColor(np.array(orig_image_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(overlay_img_cv, 0.7, heatmap, 0.3, 0)

    is_success, buffer = cv2.imencode(".png", overlay)
    if not is_success: raise RuntimeError("Failed to encode Grad-CAM.")
    # Return both the image bytes AND the probability
    return buffer.tobytes(), float(torch.sigmoid(output).item())

# --- 5. API Endpoints ---
@app.get("/")
def root():
    return {"message": "Insight Backend is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image_pil).unsqueeze(0).to(device)

        # Generate Grad-CAM and get prediction
        gradcam_bytes, probability = generate_gradcam(model, img_tensor, image_pil)
        
        # --- ENCODE BYTES TO BASE64 STRING ---
        gradcam_base64 = base64.b64encode(gradcam_bytes).decode('utf-8')
        
        return {
            "prediction_probability": probability,
            "gradcam_image_base64": gradcam_base64  # <-- Send base64 string
        }
    except Exception as e:
        print(f"[Backend] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 6. Main entry point to run the server
if __name__ == "__main__":
    print("[Backend] Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)