import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# --- DEVICE CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Backend running on: {device}")

# --- 1. LOAD THE "GATEKEEPER" (MobileNetV2) ---
print("[INFO] Loading Validator Model...")
try:
    weights_validator = MobileNet_V2_Weights.DEFAULT
    validator = mobilenet_v2(weights=weights_validator).to(device)
    validator.eval()
    validator_classes = weights_validator.meta["categories"]
    print("[SUCCESS] Validator Loaded!")
except Exception as e:
    print(f"[ERROR] Validator init failed: {e}")

# --- 2. LOAD THE "SPECIALIST" (Your Cataract Model) ---
print("[INFO] Loading Cataract Model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1) 

model_path = "fundus_pytorch_model.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[SUCCESS] Cataract Model Loaded!")
else:
    # EMERGENCY FALLBACK for testing if model is missing
    print(f"[WARNING] {model_path} not found. Using random weights for testing.")

model = model.to(device)
model.eval()

# Transforms
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
    
    # Ensure we hook the right layer
    layer = model.layer4[-1].conv2
    h1 = layer.register_forward_hook(f_hook)
    h2 = layer.register_full_backward_hook(b_hook)
    
    out = model(img_tensor)
    loss = out[0]
    model.zero_grad()
    loss.backward()
    
    h1.remove()
    h2.remove()
    
    if not gradients or not activations: return np.array(orig_img)

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (orig_img.size[0], orig_img.size[1]))
    cam = cam - np.min(cam)
    if np.max(cam) > 0: cam = cam / np.max(cam)
    
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

        # --- STEP 1: THE "CAR TEST" (Stricter Validation) ---
        with torch.no_grad():
            val_out = validator(img_tensor)
            probabilities = torch.nn.functional.softmax(val_out[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            detected_object = validator_classes[top_catid].lower()
            
            print(f"[INFO] Validator sees: '{detected_object}' with {top_prob.item():.2f} confidence.")

            # BROADER BLOCKLIST for demo safety
            forbidden = [
                'car', 'vehicle', 'truck', 'bus', 'racer', 'wheel', 
                'dog', 'cat', 'animal', 'bird', 'fish',
                'person', 'man', 'woman', 'face',
                'house', 'building', 'toy', 'food', 'fruit',
                'illustration', 'cartoon', 'art', 'sketch'
            ]
            
            # Lowered confidence threshold to 20% (0.2) to be safer
            if top_prob > 0.2 and any(x in detected_object for x in forbidden):
                print(f"[REJECTED] Image detected as '{detected_object}'")
                return JSONResponse(
                    status_code=400, 
                    content={"error": "Wrong input; please upload an eye fundus image."}
                )

        # --- STEP 2: CATARACT DIAGNOSIS (Only runs if Step 1 passes) ---
        prob = 0.0
        gradcam_base64 = None

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
        
        try:
            with torch.enable_grad():
                gradcam_img = get_gradcam(model, img_tensor, image)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
            gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as cam_error:
            print(f"[ERROR] Grad-CAM failed: {cam_error}")

        return {
            "prediction_probability": prob, 
            "gradcam_image_base64": gradcam_base64
        }

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})