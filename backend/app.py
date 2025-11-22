# backend/app.py
import os
import io
import uuid
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
IMAGE_BUCKET = os.getenv("IMAGE_BUCKET", "images")
MODEL_PATH = os.getenv("MODEL_PATH", "backend/saved_model/cataract_cnn.pt")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("❌ Missing Supabase credentials. Check your .env file.")

# =========================
# FastAPI setup
# =========================
app = FastAPI(title="InSight Cataract Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ TODO: restrict to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Supabase client
# =========================
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# =========================
# Load trained model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_labels = ["No Cataract", "Cataract"]

# =========================
# Helper: Upload to Supabase Storage
# =========================
def upload_image_to_supabase(file_bytes: bytes, filename: str) -> str:
    """Uploads an image to Supabase Storage and returns the public URL"""
    file_path = f"{uuid.uuid4()}_{filename}"
    supabase.storage.from_(IMAGE_BUCKET).upload(file_path, file_bytes, {"content-type": "image/png"})
    return supabase.storage.from_(IMAGE_BUCKET).get_public_url(file_path)

# =========================
# Helper: Grad-CAM
# =========================
def generate_gradcam(model, image_tensor, target_class, orig_image):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks on last conv layer
    layer = model.layer4[-1].conv2
    fwd_hook = layer.register_forward_hook(forward_hook)
    bwd_hook = layer.register_backward_hook(backward_hook)

    # Forward + Backward
    output = model(image_tensor)
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    # Get hooked values
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()

    weights = np.mean(grads, axis=(2, 3))[0, :]
    cam = np.zeros(acts.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (orig_image.size[0], orig_image.size[1]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Overlay heatmap on original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = np.array(orig_image)[:, :, ::-1]  # PIL to OpenCV (RGB->BGR)
    overlay = cv2.addWeighted(overlay, 0.5, heatmap, 0.5, 0)

    # Save to buffer
    _, buffer = cv2.imencode(".png", overlay)
    return buffer.tobytes()

# =========================
# API endpoint: Health check
# =========================
@app.get("/")
def root():
    return {"message": "Backend is running ✅"}

# =========================
# API endpoint: Predict + Save
# =========================
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    medical_history: str = Form(...),
    symptoms: str = Form(...),
    user_id: str = Form(...)  # doctor/nurse id
):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

        # Upload original image
        image_url = upload_image_to_supabase(contents, file.filename)

        # Grad-CAM
        gradcam_bytes = generate_gradcam(model, img_tensor, predicted_class, image)
        gradcam_filename = f"gradcam_{uuid.uuid4()}.png"
        gradcam_url = upload_image_to_supabase(gradcam_bytes, gradcam_filename)

        #Insert patient record
        patient_data = {
            "name": patient_name,
            "age": age,
            "gender": gender,
            "relevant_medical_history": medical_history,
            "self_reported_symptoms": symptoms,
            "image_url": image_url,
            "created_by": user_id
        }
        patient_res = supabase.table("patients").insert(patient_data).execute()
        patient_id = patient_res.data[0]["id"]

        #Insert prediction record
        prediction_data = {
            "patient_id": patient_id,
            "created_by": user_id,
            "prediction": class_labels[predicted_class],
            "confidence": float(probabilities[predicted_class].item()),
            "gradcam_url": gradcam_url
        }
        supabase.table("predictions").insert(prediction_data).execute()

        # Return result
        return {
            "class": class_labels[predicted_class],
            "confidence": float(probabilities[predicted_class].item()),
            "all_probabilities": {
                "No Cataract": float(probabilities[0].item()),
                "Cataract": float(probabilities[1].item())
            },
            "patient_id": patient_id,
            "image_url": image_url,
            "gradcam_url": gradcam_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
