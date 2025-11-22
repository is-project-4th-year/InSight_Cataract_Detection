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
    h1 { font-size: 2.5em; margin-bottom: 0.5em; }
    h2, h3 { margin-top: 1.5em; margin-bottom: 0.8em; }
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
        # Fallback for local testing if secrets.toml isn't used
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
    if not url or not key:
        st.error("Supabase credentials not found. Please check .streamlit/secrets.toml or Cloud Settings.")
        return None
    return create_client(url, key)

supabase = init_supabase_client()

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model Architecture (ResNet18)
    # We use the exact same architecture used during training
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1) 

    # Load Weights
    model_path = "fundus_pytorch_model.pt" 
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file '{model_path}' not found in the repository.")
        return None, None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, device = load_model()

# --- PREPROCESSING TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_labels = ["No Cataract", "Cataract Detected"]

# --- GRAD-CAM HELPER ---
def generate_gradcam(model, image_tensor, target_class, orig_image):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook the last convolutional layer of ResNet18
    layer = model.layer4[-1].conv2
    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_full_backward_hook(backward_hook)

    # Forward & Backward
    output = model(image_tensor)
    # Ensure target_class is within bounds (0 or 1)
    if target_class >= output.shape[1]:
        target_class = torch.argmax(output).item()
        
    loss = output[0]
    model.zero_grad()
    loss.backward()

    # Remove hooks to prevent memory leaks
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

# --- PROFESSIONAL PDF HELPER FUNCTION ---
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
    
    if prediction == "Cataract Detected":
        pdf.set_text_color(220, 53, 69) # Red
    else:
        pdf.set_text_color(25, 135, 84) # Green
        
    pdf.cell(0, 8, f"{prediction}", 0, 1, "L")
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font("Times", "", 12)
    if prediction == "Cataract Detected":
        conf_text = f"{confidence:.1%}"
    else:
        conf_text = f"{1-confidence:.1%}"
        
    pdf.cell(50, 8, "Confidence Level:", 0, 0, "L")
    pdf.cell(0, 8, conf_text, 0, 1, "L")
    
    pdf.set_font("Times", "B", 12)
    pdf.cell(50, 8, "Recommendation:", 0, 0, "L")
    pdf.set_font("Times", "", 12)
    pdf.cell(0, 8, recommendation, 0, 1, "L")
    pdf.ln(10)

    # Images
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Analysis Visualization", 0, 1, "L")
    
    y_images = pdf.get_y()
    
    # Original
    try:
        img_stream = io.BytesIO(original_image_bytes)
        img_type = original_image_type_mime.split('/')[-1].upper()
        if img_type == "JPEG": img_type = "JPG"
        pdf.image(img_stream, x=10, y=y_images, w=90, type=img_type)
        pdf.set_xy(10, y_images + 90)
        pdf.set_font("Times", "", 10)
        pdf.cell(90, 5, "Original Image", 0, 0, "C")
    except: pass

    # Grad-CAM
    try:
        # Convert numpy array to image bytes for PDF
        gc_pil = Image.fromarray(overlay_img)
        gc_stream = io.BytesIO()
        gc_pil.save(gc_stream, format="PNG")
        pdf.image(gc_stream, x=110, y=y_images, w=90, type="PNG")
        pdf.set_xy(110, y_images + 90)
        pdf.cell(90, 5, "AI Visualization (Grad-CAM)", 0, 0, "C")
    except: pass
    
    # Disclaimer
    pdf.set_y(-35)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Times", "I", 9)
    pdf.set_text_color(108, 117, 125)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an AI-assisted screening tool (InSight) and is intended for informational purposes only. It is not a medical diagnosis.", 0, "C")
    
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
            role = st.selectbox("Select your role", ("Nurse", "Doctor"))
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                try:
                    session = supabase.auth.sign_up({"email": email, "password": password})
                    if session.user:
                        supabase.table('user_roles').insert({'user_id': session.user.id, 'email': email, 'role': role}).execute()
                        st.success("Signup successful! Please check your email.")
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
    st.write("Upload a fundus photograph to screen for cataracts.")
    
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])
    patient_id = st.text_input("Patient Identifier")

    if uploaded_file and patient_id:
        st.write("")
        col1, col2 = st.columns(2)
        
        # Prepare images
        image_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        with col1:
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            if model is None:
                st.error("AI Model failed to load. Please check server logs.")
            else:
                with col2:
                    with st.spinner("Running AI Analysis..."):
                        # 1. Preprocess
                        img_tensor = transform(pil_image).unsqueeze(0).to(device)
                        
                        # NEW CODE (Use this)
                        outputs = model(img_tensor)
                        # Use sigmoid for binary classification (1 output node)
                        pred_prob = torch.sigmoid(outputs).item()

                        if pred_prob > 0.5:
                            predicted_class_idx = 1 # Cataract
                            # pred_prob is already the probability of cataract
                        else:
                            predicted_class_idx = 0 # No Cataract
                            # Invert probability so it represents confidence in "No Cataract"
                            pred_prob = 1 - pred_prob
                        
                        # 3. Generate Grad-CAM locally
                        gradcam_img = generate_gradcam(model, img_tensor, predicted_class_idx, pil_image)

                        # 4. Logic
                        if predicted_class_idx == 1: # Cataract
                            label = "Cataract Detected"
                            recommendation = "Urgent: Refer to Ophthalmologist"
                            st.error(f"**Result: {label}** (Confidence: {pred_prob:.1%})")
                        else:
                            label = "No Cataract Detected"
                            recommendation = "Routine: Next annual check-up"
                            st.success(f"**Result: {label}** (Confidence: {pred_prob:.1%})")
                        
                        st.warning(f"**Recommendation: {recommendation}**")
                        
                        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

                        # Store results in session state to allow saving/exporting after interaction
                        st.session_state["last_result"] = {
                            "patient_id": patient_id,
                            "label": label,
                            "prob": pred_prob,
                            "rec": recommendation,
                            "img_bytes": image_bytes,
                            "gc_img": gradcam_img,
                            "mime": uploaded_file.type
                        }

        # Buttons for Save/Export (Only show if a result exists for THIS patient)
        if "last_result" in st.session_state and st.session_state["last_result"]["patient_id"] == patient_id:
            res = st.session_state["last_result"]
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("Save Results", use_container_width=True):
                    try:
                        data_to_insert = {
                            "patient_identifier": res["patient_id"],
                            "image_filename": uploaded_file.name,
                            "predicted_label": res["label"],
                            "prediction_probability": res["prob"],
                            "recommendation": res["rec"],
                            "screened_by_user": st.session_state["user_name"]
                        }
                        supabase.table('screenings').insert(data_to_insert).execute()
                        st.success("Saved results successfully!")
                    except Exception as e: st.error(f"Error saving to database: {e}")

            with btn_col2:
                pdf_bytes = create_screening_report(
                    res["patient_id"], res["label"], res["prob"], res["rec"],
                    st.session_state["user_name"], res["img_bytes"], res["gc_img"], res["mime"]
                )
                st.download_button(
                    label="Export PDF Report",
                    data=pdf_bytes,
                    file_name=f"InSight_Report_{res['patient_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
            with btn_col3:
                # Convert numpy array to bytes for download
                gc_pil = Image.fromarray(res["gc_img"])
                buf = io.BytesIO()
                gc_pil.save(buf, format="PNG")
                st.download_button(
                    label="Export Grad-CAM",
                    data=buf.getvalue(),
                    file_name=f"{res['patient_id']}_gradcam.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            st.caption("DISCLAIMER: This report is generated by an AI-assisted screening tool (InSight) and is intended for informational purposes only. It is not a substitute for a comprehensive examination and diagnosis by a qualified ophthalmologist.")


# --- HISTORY PAGE ---
def patient_history_page():
    st.title("Patient Screening History")
    st.write("Enter a Patient Identifier to view their past screening results.")
    search_id = st.text_input("Patient Identifier")
    
    if st.button("Search History"):
        if not search_id: st.warning("Please enter a Patient Identifier."); return
        with st.spinner(f"Searching for patient {search_id}..."):
            try:
                query = supabase.table('screenings').select("*").eq('patient_identifier', search_id).order('created_at', desc=True).execute()
                if query.data:
                    df = pd.DataFrame(query.data)
                    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Cleanup columns for display
                    display_df = df[['created_at', 'predicted_label', 'prediction_probability', 'recommendation', 'screened_by_user', 'image_filename']].rename(columns={'created_at': 'Screening Date', 'predicted_label': 'Prediction', 'prediction_probability': 'Confidence', 'recommendation': 'Recommendation', 'screened_by_user': 'Screened By', 'image_filename': 'Original Image'})
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No records found for this patient.")
            except Exception as e:
                st.error(f"Error fetching history: {e}")

# --- ANALYTICS PAGE ---
def analytics_page():
    st.title("Analytics Dashboard")
    st.write("Overview of screening metrics. (Only Doctors can see this page)")
    try:
        data = supabase.table('screenings').select("*").execute()
        df = pd.DataFrame(data.data)
    except Exception as e: st.error(f"Error fetching data: {e}"); df = pd.DataFrame()
    
    if df.empty:
        st.warning("No screening data found to display analytics.")
        return
        
    total_screened = len(df)
    total_cataract = len(df[df['predicted_label'] == 'Cataract Detected'])
    percent_cataract = (total_cataract / total_screened) * 100 if total_screened > 0 else 0
    
    col1, col2 = st.columns(2)
    col1.metric("Total Patients Screened", total_screened)
    col2.metric("Cataracts Detected", total_cataract, f"{percent_cataract:.1f}% of total")
    
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Screenings by Recommendation")
        recommend_counts = df['recommendation'].value_counts()
        fig1 = px.bar(recommend_counts, x=recommend_counts.index, y=recommend_counts.values, labels={'x': 'Recommendation', 'y': 'Count'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with col4:
        st.subheader("Detections by Type")
        label_counts = df['predicted_label'].value_counts()
        fig2 = px.pie(label_counts, names=label_counts.index, values=label_counts.values)
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Recent Screening Data")
    st.dataframe(df.sort_values(by="created_at", ascending=False).head(10), use_container_width=True)

# --- MAIN ROUTER ---
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    main_auth_page()
else:
    st.sidebar.title("Navigation")
    st.sidebar.write(f"Welcome, **{st.session_state['user_name']}**!")
    st.sidebar.write(f"Role: *{st.session_state['user_role']}*")
    
    page_options = ["Screening", "Patient History"]
    if st.session_state["user_role"] == "Doctor":
        page_options.append("Analytics")
        
    page = st.sidebar.radio("Go to", page_options)
    st.sidebar.divider()
    
    if st.sidebar.button("Logout"): logout()
    
    if page == "Screening": screening_page()
    elif page == "History": patient_history_page()
    elif page == "Analytics": analytics_page()