import streamlit as st
import pandas as pd
from PIL import Image
import io
import plotly.express as px
from supabase import create_client, Client
from datetime import datetime
import requests 
import traceback
import base64
from fpdf import FPDF # For PDF Report

# --- API URL ---
BACKEND_URL = "http://127.0.0.1:8000/predict/"

# --- CUSTOM CSS ---
st.markdown("""
<style>
    html, body, [class*="st-"], [class*="css-"] { font-family: 'Source Sans Pro', sans-serif; }
    h1 { font-size: 2.5em; margin-bottom: 0.5em; }
    h2, h3 { margin-top: 1.5em; margin-bottom: 0.8em; }
    .stButton>button { margin-top: 10px; margin-bottom: 10px; }
    .stMultiSelect, .stTextInput, .stFileUploader, .stNumberInput, .stSelectbox { margin-bottom: 15px; }
    .stAlert[data-baseweb="alert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- PAGE CONFIG ---
st.set_page_config(page_title="InSight: Cataract Screening", layout="wide")

# --- SUPABASE CONNECTION ---
@st.cache_resource
def init_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase_client()


# --- PROFESSIONAL PDF HELPER FUNCTION ---
def create_screening_report(patient_id, prediction, confidence, recommendation, screened_by, original_image_bytes, overlay_bytes, original_image_type_mime):
    """
    Generates a professional PDF report from the screening results,
    including the original image and the Grad-CAM visualization side-by-side.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Title ---
    pdf.set_font("Times", "B", 20)
    pdf.cell(0, 10, "InSight Ocular Screening Report", 0, 1, "C")
    pdf.ln(5)
    # Add a horizontal line
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # --- Patient Information ---
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Patient Information", 0, 1, "L")
    pdf.set_font("Times", "", 12)
    
    # Using two cells for clean alignment (Label + Value)
    pdf.cell(50, 8, "Patient Identifier:", 0, 0, "L")
    pdf.cell(0, 8, f"{patient_id}", 0, 1, "L")
    
    pdf.cell(50, 8, "Screening Date:", 0, 0, "L")
    pdf.cell(0, 8, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "L")
    
    pdf.cell(50, 8, "Screened By (Clinician):", 0, 0, "L")
    pdf.cell(0, 8, f"{screened_by}", 0, 1, "L")
    pdf.ln(10)

    # --- Ocular Screening Analysis ---
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Ocular Screening Analysis", 0, 1, "L")

    # --- Result ---
    pdf.set_font("Times", "B", 12)
    pdf.cell(50, 8, "Assessment:", 0, 0, "L")
    if prediction == "Cataract Detected":
        pdf.set_text_color(220, 53, 69) # Red
    else:
        pdf.set_text_color(25, 135, 84) # Green
    pdf.cell(0, 8, f"{prediction}", 0, 1, "L")
    pdf.set_text_color(0, 0, 0) # Reset color

    # --- Confidence ---
    pdf.set_font("Times", "", 12)
    if prediction == "Cataract Detected":
        conf_text = f"{confidence:.1%}"
    else:
        conf_text = f"{1-confidence:.1%}"
    pdf.cell(50, 8, "Confidence Level:", 0, 0, "L")
    pdf.cell(0, 8, conf_text, 0, 1, "L")

    # --- Recommendation ---
    pdf.set_font("Times", "B", 12)
    pdf.cell(50, 8, "Clinical Recommendation:", 0, 0, "L")
    pdf.set_font("Times", "", 12)
    pdf.cell(0, 8, recommendation, 0, 1, "L")
    pdf.ln(10)
    
    # --- IMAGES (Side-by-Side) ---
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Analysis Visualization", 0, 1, "L")

    # Define image layout
    page_width = 210
    margin = 10
    gap = 10
    img_width = (page_width - 2 * margin - gap) / 2 # 85mm
    x1 = margin
    x2 = margin + img_width + gap
    y_start = pdf.get_y()

    # Sub-headers for images
    pdf.set_font("Times", "", 12)
    pdf.set_xy(x1, y_start)
    pdf.cell(img_width, 8, "Original Fundus Image", 0, 0, "C")
    pdf.set_xy(x2, y_start)
    pdf.cell(img_width, 8, "AI Visualization (Grad-CAM)", 0, 1, "C")
    pdf.ln(2)
    
    y_images = pdf.get_y()

    # --- Original Image ---
    try:
        original_image_stream = io.BytesIO(original_image_bytes)
        # Dynamically get image type (JPG, PNG, etc.)
        img_type = original_image_type_mime.split('/')[-1].upper()
        if img_type == "JPEG": img_type = "JPG"
        
        pdf.image(original_image_stream, x=x1, y=y_images, w=img_width, type=img_type)
    except Exception as e:
        pdf.set_xy(x1, y_images)
        pdf.set_font("Times", "I", 10)
        pdf.set_text_color(220, 53, 69)
        pdf.multi_cell(img_width, 8, f"Error embedding original image: {e}", 1, "C")
        pdf.set_text_color(0, 0, 0)

    # --- Grad-CAM Image ---
    try:
        overlay_stream = io.BytesIO(overlay_bytes)
        pdf.image(overlay_stream, x=x2, y=y_images, w=img_width, type='PNG')
    except Exception as e:
        pdf.set_xy(x2, y_images)
        pdf.set_font("Times", "I", 10)
        pdf.set_text_color(220, 53, 69)
        pdf.multi_cell(img_width, 8, f"Error embedding Grad-CAM: {e}", 1, "C")
        pdf.set_text_color(0, 0, 0)
        
    # Move cursor down past the images
    # We set Y explicitly in case one image fails
    pdf.set_y(y_images + img_width + 10) # Move down past 85mm image + 10mm buffer

    # --- Footer & Disclaimer ---
    pdf.set_y(-35) # Position 35mm from bottom
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Footer line
    pdf.ln(5)
    
    pdf.set_font("Times", "I", 9)
    pdf.set_text_color(108, 117, 125)
    pdf.multi_cell(0, 5, 
        "DISCLAIMER: This report is generated by an AI-assisted screening tool (InSight) "
        "and is intended for informational purposes only. It is not a substitute for a comprehensive "
        "examination and diagnosis by a qualified ophthalmologist. All clinical decisions "
        "must be made by a licensed healthcare professional.", 0, "C")
    
    pdf.set_y(-15) # Position 15mm from bottom
    pdf.set_font("Times", "", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, "C")

    return bytes(pdf.output())


# --- AUTHENTICATION FUNCTIONS ---
@st.cache_data(ttl=600)
def get_user_role(_session):
    user_id = _session.user.id
    try:
        data = supabase.table('user_roles').select('role').eq('user_id', user_id).single().execute()
        if data.data and 'role' in data.data: return data.data['role']
        else: return "Nurse"
    except Exception as e:
        print(f"Error fetching role: {e}"); return "Nurse"

def main_auth_page():
    st.title("InSight Cataract Screening")
    st.write("Please log in or sign up to continue.")
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    
    # --- LOGIN TAB (This part is unchanged and correct) ---
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
                    user_id = session.user.id
                    
                    # This code correctly fetches the role from your table
                    role_data = supabase.table('user_roles').select('role').eq('user_id', user_id).single().execute()
                    
                    user_role = "Nurse" # Default if not found
                    if role_data.data and 'role' in role_data.data: 
                        user_role = role_data.data['role']
                        
                    st.session_state["user_role"] = user_role
                    st.session_state["user_name"] = session.user.email
                    st.rerun()
                except Exception as e: st.error(f"Login failed: {e}")

    # --- SIGNUP TAB (This part is new) ---
    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email")
            password = st.text_input("Password (min. 6 characters)", type="password")
            
            # --- NEW: ROLE DROPDOWN ---
            role = st.selectbox(
                "Select your role",
                ("Nurse", "Doctor"),
                help="Select your clinical designation."
            )
            # --- END NEW ---

            submitted = st.form_submit_button("Sign Up")
            if submitted:
                try:
                    # 1. Create the user in the main 'auth.users' table
                    session = supabase.auth.sign_up({"email": email, "password": password})
                    user = session.user if session and hasattr(session, 'user') else None

                    if user:
                        # 2. Get the new user's ID and email
                        user_id = user.id
                        user_email = user.email
                        
                        # --- NEW: Insert role into your 'user_roles' table ---
                        # This matches your table structure: (user_id, email, role)
                        try:
                            supabase.table('user_roles').insert({
                                'user_id': user_id, 
                                'email': user_email, # Matches your table's email column
                                'role': role
                            }).execute()
                            
                            st.success("Signup successful! Please check your email for verification and then log in.")
                        
                        except Exception as e:
                            st.error(f"Signup succeeded but failed to set role: {e}")
                            st.info("Please contact an administrator to have your role assigned.")
                        # --- END NEW ---
                            
                    else: 
                        st.error("Signup failed or user info not returned.")
                        
                except Exception as e:
                    error_message = str(e)
                    if "user already registered" in error_message.lower(): st.warning("Email already registered.")
                    elif "Password should be at least 6 characters" in error_message: st.error("Password too weak.")
                    else: st.error(f"Sign up failed: {error_message}")

def logout():
    try: supabase.auth.sign_out()
    except Exception as e: print(f"Supabase signout failed: {e}")
    st.session_state["logged_in"] = False
    st.session_state.pop("session", None); st.session_state.pop("user_role", None); st.session_state.pop("user_name", None)


# --- SCREENING PAGE ---
def screening_page():
    st.title("New Patient Screening")
    st.write("Upload a fundus photograph to screen for cataracts.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    patient_id = st.text_input("Patient Identifier")

    if uploaded_file is not None and patient_id:
        st.write("")
        col1, col2 = st.columns(2)
        
        # Get original image bytes and type
        image_bytes = uploaded_file.getvalue()
        image_type = uploaded_file.type # e.g., "image/png"
        pil_image = Image.open(io.BytesIO(image_bytes))

        with col1:
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        analysis_key = f"analysis_results_{patient_id}_{uploaded_file.name}"
        if analysis_key not in st.session_state:
             st.session_state[analysis_key] = None

        if st.button("Analyze Image"):
            with col2:
                with st.spinner("Analyzing image... (Connecting to backend)"):
                    try:
                        files = {'file': (uploaded_file.name, image_bytes, image_type)}
                        response = requests.post(BACKEND_URL, files=files)
                        response.raise_for_status() 
                        
                        result_json = response.json()
                        pred_prob = result_json["prediction_probability"]
                        gradcam_base64 = result_json["gradcam_image_base64"]
                        gradcam_bytes = base64.b64decode(gradcam_base64)

                        st.session_state[analysis_key] = {
                            "pred_prob": pred_prob,
                            "overlay_bytes": gradcam_bytes,
                        }
                        st.rerun()

                    except requests.exceptions.ConnectionError:
                         st.error(f"Connection Error: Could not connect to backend at {BACKEND_URL}.")
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        st.error(traceback.format_exc())
                        st.session_state[analysis_key] = {"error": str(e)}

        if st.session_state[analysis_key] and "error" not in st.session_state[analysis_key]:
             results = st.session_state[analysis_key]
             pred_prob = results["pred_prob"]
             overlay_bytes = results["overlay_bytes"] 

             with col2:
                prediction_threshold = 0.5 
                if pred_prob > prediction_threshold:
                    label = "Cataract Detected"
                    recommendation = "Urgent: Refer to Ophthalmologist"
                    st.error(f"**Result: {label}** (Confidence: {pred_prob:.1%})")
                else:
                    label = "No Cataract Detected"
                    recommendation = "Routine: Next annual check-up"
                    st.success(f"**Result: {label}** (Confidence: {1-pred_prob:.1%})")

                st.warning(f"**Recommendation: {recommendation}**")

                st.image(overlay_bytes, caption="Grad-CAM Heatmap", use_container_width=True)
                
                # --- THREE BUTTONS IN COLUMNS ---
                btn_col1, btn_col2, btn_col3 = st.columns(3)

                with btn_col1:
                    save_key = f"save_btn_{patient_id}_{uploaded_file.name}"
                    if st.button("Save Results", key=save_key, use_container_width=True):
                        with st.spinner("Saving results..."):
                            try:
                                data_to_insert = {
                                    "patient_identifier": patient_id, "image_filename": uploaded_file.name,
                                    "predicted_label": label, "prediction_probability": pred_prob,
                                    "recommendation": recommendation, "screened_by_user": st.session_state["user_name"]
                                }
                                api_response = supabase.table('screenings').insert(data_to_insert).execute()
                                if api_response and hasattr(api_response, 'data') and api_response.data:
                                    st.success(f"Saved results for {patient_id}.")
                                    st.session_state[analysis_key] = None
                                else:
                                    st.warning(f"Save submitted, but confirmation failed.")
                            except Exception as e:
                                st.error(f"Error saving to database: {e}")

                with btn_col2:
                    # Pass the original image bytes and type to the PDF function
                    pdf_bytes = create_screening_report(
                        patient_id,
                        label,
                        pred_prob,
                        recommendation,
                        st.session_state["user_name"],
                        image_bytes,   # <-- Pass original image bytes
                        overlay_bytes, # <-- Pass Grad-CAM bytes
                        image_type     # <-- Pass original image MIME type
                    )
                    
                    st.download_button(
                        label="Export PDF Report",
                        data=pdf_bytes,
                        file_name=f"InSight_Report_{patient_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with btn_col3:
                    st.download_button(
                        label="Export Grad-CAM",
                        data=overlay_bytes, # Use the same bytes
                        file_name=f"{patient_id}_gradcam_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )

    # --- NEW: DISCLAIMER ---
    st.caption("DISCLAIMER: This report is generated by an AI-assisted screening tool "
                "and is intended for informational purposes only. It is not a substitute for a "
                "comprehensive examination and diagnosis by a qualified ophthalmologist. All clinical "
                "decisions must be made by a licensed healthcare professional.")
                # --- END NEW ---


# --- PATIENT HISTORY PAGE ---
def patient_history_page():
    st.title("Patient Screening History")
    st.write("Enter a Patient Identifier to view their past screening results.")
    search_id = st.text_input("Patient Identifier")
    if st.button("Search History"):
        if not search_id: st.warning("Please enter a Patient Identifier."); return
        with st.spinner(f"Searching for patient {search_id}..."):
            try:
                query = supabase.table('screenings').select("*").eq('patient_identifier', search_id).order('created_at', desc=True).execute()
                history_df = pd.DataFrame(query.data)
                if history_df.empty:
                    st.info(f"No screening history found for patient {search_id} (or you may not have permission to view their records).")
                else:
                    st.success(f"Found {len(history_df)} screening records for patient {search_id}:")
                    history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    history_df['prediction_probability'] = history_df['prediction_probability'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    display_df = history_df[['created_at', 'predicted_label', 'prediction_probability', 'recommendation', 'screened_by_user', 'image_filename']].rename(columns={'created_at': 'Screening Date', 'predicted_label': 'Prediction', 'prediction_probability': 'Confidence', 'recommendation': 'Recommendation', 'screened_by_user': 'Screened By', 'image_filename': 'Original Image File'})
                    st.dataframe(display_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error fetching patient history: {e}")


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
    total_screened = len(df); total_cataract = len(df[df['predicted_label'] == 'Cataract Detected'])
    percent_cataract = (total_cataract / total_screened) * 100 if total_screened > 0 else 0
    col1, col2 = st.columns(2)
    col1.metric("Total Patients Screened (in your view)", total_screened)
    col2.metric("Cataracts Detected (in your view)", total_cataract, f"{percent_cataract:.1f}% of total")
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
    st.subheader("Recent Screening Data (in your view)")
    st.dataframe(df.sort_values(by="created_at", ascending=False).head(10), use_container_width=True)


# --- MAIN APP LOGIC ---
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
    st.sidebar.button("Logout", on_click=logout)
    
    if page == "Screening": 
        screening_page()
    elif page == "Patient History": 
        patient_history_page()
    elif page == "Analytics": 
        analytics_page()