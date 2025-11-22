import streamlit as st
import pandas as pd
from PIL import Image
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from supabase import create_client, Client
from datetime import datetime
import plotly.express as px
from fpdf import FPDF
import base64
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="InSight: Cataract Screening", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    html, body, [class*="st-"], [class*="css-"] { font-family: 'Source Sans Pro', sans-serif; }
    .stButton>button { margin-top: 10px; margin-bottom: 10px; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- SUPABASE CONNECTION ---
@st.cache_resource
def init_supabase_client():
    # Try loading from Streamlit Secrets (Cloud) first, then environment variables (Local)
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
    if not url or not key:
        st.error("Supabase credentials not found. Please check .streamlit/secrets.toml")
        return None
    return create_client(url, key)

supabase = init_supabase_client()

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model Architecture (ResNet18)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2) # Binary classification

    # Load Weights
    model_path = "fundus_pytorch_model.pt" 
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

model, device = load_model()

# --- PREPROCESSING TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_labels = ["No Cataract", "Cataract"]

# --- GRAD-CAM HELPER ---
def generate_gradcam(model, image_tensor, target_class, orig_image):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook the last convolutional layer
    layer = model.layer4[-1].conv2
    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_full_backward_hook(backward_hook)

    # Forward & Backward
    output = model(image_tensor)
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # Process Gradients
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()
    weights = np.mean(grads, axis=(2, 3))[0, :]
    cam = np.zeros(acts.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (orig_image.size[0], orig_image.size[1]))
    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    # Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = np.array(orig_image)[:, :, ::-1] # RGB to BGR
    overlay = cv2.addWeighted(overlay, 0.5, heatmap, 0.5, 0)
    
    # Convert back to RGB for display/saving
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- PDF HELPER FUNCTION ---
def create_screening_report(patient_id, prediction, confidence, recommendation, screened_by, original_image_bytes, overlay_img, original_image_type_mime):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Times", "B", 20)
    pdf.cell(0, 10, "InSight Ocular Screening Report", 0, 1, "C")
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Info
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Patient Information", 0, 1, "L")
    pdf.set_font("Times", "", 12)
    pdf.cell(50, 8, "Patient Identifier:", 0, 0, "L")
    pdf.cell(0, 8, f"{patient_id}", 0, 1, "L")
    pdf.cell(50, 8, "Screening Date:", 0, 0, "L")
    pdf.cell(0, 8, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "L")
    pdf.cell(50, 8, "Screened By:", 0, 0, "L")
    pdf.cell(0, 8, f"{screened_by}", 0, 1, "L")
    pdf.ln(10)

    # Results
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Ocular Screening Analysis", 0, 1, "L")
    pdf.set_font("Times", "B", 12)
    pdf.cell(50, 8, "Assessment:", 0, 0, "L")
    pdf.set_text_color(220, 53, 69) if prediction == "Cataract Detected" else pdf.set_text_color(25, 135, 84)
    pdf.cell(0, 8, f"{prediction}", 0, 1, "L")
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font("Times", "", 12)
    conf_text = f"{confidence:.1%}" if prediction == "Cataract Detected" else f"{1-confidence:.1%}"
    pdf.cell(50, 8, "Confidence Level:", 0, 0, "L")
    pdf.cell(0, 8, conf_text, 0, 1, "L")
    
    pdf.set_font("Times", "B", 12)
    pdf.cell(50, 8, "Recommendation:", 0, 0, "L")
    pdf.set_font("Times", "", 12)
    pdf.cell(0, 8, recommendation, 0, 1, "L")
    pdf.ln(10)

    # Images
    y_images = pdf.get_y()
    
    # Original
    try:
        img_stream = io.BytesIO(original_image_bytes)
        img_type = original_image_type_mime.split('/')[-1].upper()
        if img_type == "JPEG": img_type = "JPG"
        pdf.image(img_stream, x=10, y=y_images, w=90, type=img_type)
    except: pass

    # Grad-CAM
    try:
        # Convert numpy array to image bytes
        gc_pil = Image.fromarray(overlay_img)
        gc_stream = io.BytesIO()
        gc_pil.save(gc_stream, format="PNG")
        pdf.image(gc_stream, x=110, y=y_images, w=90, type="PNG")
    except: pass
    
    pdf.set_y(-35)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Times", "I", 9)
    pdf.set_text_color(108, 117, 125)
    pdf.multi_cell(0, 5, "DISCLAIMER: AI-generated screening report. Not a medical diagnosis.", 0, "C")
    
    return bytes(pdf.output())

# --- AUTHENTICATION ---
def main_auth_page():
    st.title("InSight Cataract Screening")
    st.write("Please log in or sign up to continue.")
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                try:
                    session = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state["session"] = session
                    st.session_state["logged_in"] = True
                    
                    # Fetch Role
                    user_id = session.user.id
                    role_data = supabase.table('user_roles').select('role').eq('user_id', user_id).single().execute()
                    user_role = "Nurse"
                    if role_data.data and 'role' in role_data.data: user_role = role_data.data['role']
                    st.session_state["user_role"] = user_role
                    st.session_state["user_name"] = session.user.email
                    st.rerun()
                except Exception as e: st.error(f"Login failed: {e}")

    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ("Nurse", "Doctor"))
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                try:
                    session = supabase.auth.sign_up({"email": email, "password": password})
                    if session.user:
                        supabase.table('user_roles').insert({'user_id': session.user.id, 'role': role}).execute()
                        st.success("Signup successful! Check email.")
                    else: st.error("Signup failed.")
                except Exception as e: st.error(f"Error: {e}")

def logout():
    supabase.auth.sign_out()
    st.session_state["logged_in"] = False
    st.session_state.pop("session", None)
    st.rerun()

# --- SCREENING PAGE ---
def screening_page():
    st.title("New Patient Screening")
    
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])
    patient_id = st.text_input("Patient Identifier")

    if uploaded_file and patient_id:
        col1, col2 = st.columns(2)
        image_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        with col1:
            st.image(pil_image, caption="Original Image", use_container_width=True)

        if st.button("Analyze Image"):
            if model is None:
                st.error("Model failed to load.")
            else:
                with col2:
                    with st.spinner("Running AI Analysis..."):
                        # 1. Preprocess
                        img_tensor = transform(pil_image).unsqueeze(0).to(device)
                        
                        # 2. Inference
                        outputs = model(img_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0]
                        predicted_class_idx = torch.argmax(probabilities).item()
                        pred_prob = probabilities[predicted_class_idx].item()
                        
                        # 3. Grad-CAM
                        gradcam_img = generate_gradcam(model, img_tensor, predicted_class_idx, pil_image)

                        # 4. Logic
                        if predicted_class_idx == 1: # Cataract
                            label = "Cataract Detected"
                            recommendation = "Urgent: Refer to Ophthalmologist"
                            st.error(f"**{label}** ({pred_prob:.1%})")
                        else:
                            label = "No Cataract Detected"
                            recommendation = "Routine Check-up"
                            st.success(f"**{label}** ({1-pred_prob:.1%})")
                        
                        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

                        # Store results in session state to allow saving/exporting
                        st.session_state["last_result"] = {
                            "patient_id": patient_id,
                            "label": label,
                            "prob": pred_prob,
                            "rec": recommendation,
                            "img_bytes": image_bytes,
                            "gc_img": gradcam_img,
                            "mime": uploaded_file.type
                        }

        # Buttons for Save/Export (Only show if result exists)
        if "last_result" in st.session_state and st.session_state["last_result"]["patient_id"] == patient_id:
            res = st.session_state["last_result"]
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("Save to Database"):
                    try:
                        data = {
                            "patient_identifier": res["patient_id"],
                            "image_filename": uploaded_file.name,
                            "predicted_label": res["label"],
                            "prediction_probability": res["prob"],
                            "recommendation": res["rec"],
                            "screened_by_user": st.session_state["user_name"]
                        }
                        supabase.table('screenings').insert(data).execute()
                        st.success("Saved successfully!")
                    except Exception as e: st.error(f"Save failed: {e}")

            with btn_col2:
                pdf_bytes = create_screening_report(
                    res["patient_id"], res["label"], res["prob"], res["rec"],
                    st.session_state["user_name"], res["img_bytes"], res["gc_img"], res["mime"]
                )
                st.download_button("Export PDF", pdf_bytes, "report.pdf", "application/pdf")
                
            st.caption("DISCLAIMER: AI-assisted screening only. Not a clinical diagnosis.")


# --- HISTORY & ANALYTICS PAGES (Simplified) ---
def patient_history_page():
    st.title("Patient History")
    search_id = st.text_input("Search Patient ID")
    if st.button("Search"):
        data = supabase.table('screenings').select("*").eq('patient_identifier', search_id).execute()
        if data.data:
            df = pd.DataFrame(data.data)
            st.dataframe(df[['created_at', 'predicted_label', 'prediction_probability', 'screened_by_user']], use_container_width=True)
        else:
            st.info("No records found.")

def analytics_page():
    st.title("Analytics Dashboard")
    data = supabase.table('screenings').select("*").execute()
    if data.data:
        df = pd.DataFrame(data.data)
        col1, col2 = st.columns(2)
        col1.metric("Total Screenings", len(df))
        cataract_count = len(df[df['predicted_label'] == 'Cataract Detected'])
        col2.metric("Cataracts Detected", cataract_count)
        
        fig = px.pie(df, names='predicted_label', title='Detection Distribution')
        st.plotly_chart(fig)
    else:
        st.info("No data available for analytics.")

# --- MAIN ROUTER ---
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    main_auth_page()
else:
    st.sidebar.title(f"Welcome, {st.session_state['user_role']}")
    page = st.sidebar.radio("Navigate", ["Screening", "History", "Analytics"] if st.session_state["user_role"] == "Doctor" else ["Screening", "History"])
    if st.sidebar.button("Logout"): logout()
    
    if page == "Screening": screening_page()
    elif page == "History": patient_history_page()
    elif page == "Analytics": analytics_page()