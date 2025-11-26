# frontend/streamlit_app.py
import streamlit as st
import requests
from auth import signup, login, supabase  # custom Supabase auth helpers

# ==============================
# Config
# ==============================
API_URL = "http://127.0.0.1:8000/predict/"  # backend FastAPI

st.set_page_config(page_title="InSight Cataract Detection")

# ==============================
# Sidebar: Authentication
# ==============================
st.sidebar.title("User Authentication")

if "user" not in st.session_state:
    # ------------------------------
    # Not logged in
    # ------------------------------
    auth_action = st.sidebar.radio("Action", ["Login", "Signup"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if auth_action == "Signup":
        role = st.sidebar.selectbox("Role", ["doctor", "nurse"])
        if st.sidebar.button("Create Account"):
            success, msg = signup(email, password, role)
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

    if auth_action == "Login":
        if st.sidebar.button("Login"):
            success, user = login(email, password)
            if success:
                # Save user info in session
                st.session_state["user"] = {
                    "id": user.id,        # Supabase user UUID
                    "email": user.email,
                    "role": None,         # will fetch below
                }
                st.sidebar.success(f"Welcome {email}!")

                # Fetch role from profiles table
                try:
                    profile = (
                        supabase.table("profiles")
                        .select("role")
                        .eq("id", user.id)
                        .single()
                        .execute()
                    )
                    if profile.data and "role" in profile.data:
                        st.session_state["user"]["role"] = profile.data["role"]
                        st.sidebar.info(f"Role: {profile.data['role']}")
                except Exception as e:
                    st.sidebar.warning(f"Could not fetch role: {e}")
            else:
                st.sidebar.error(user)  # user contains error message
else:
    # ------------------------------
    # Already logged in
    # ------------------------------
    user = st.session_state["user"]
    st.sidebar.success(f"Logged in as {user['email']}")
    if user.get("role"):
        st.sidebar.info(f"Role: {user['role']}")

    if st.sidebar.button("Logout"):
        del st.session_state["user"]
        st.rerun()

# ==============================
# Main: Cataract Detection
# ==============================
st.title("InSight Cataract Detection System")

if "user" not in st.session_state:
    st.warning("Please log in to continue.")
    st.stop()

user = st.session_state["user"]

st.write("Upload an eye image and fill in patient details for screening.")

# Patient details
patient_name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
medical_history = st.text_area("Relevant Medical History")
symptoms = st.text_area("Self-reported Symptoms")

# File uploader
uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        if not patient_name:
            st.error("Please provide the patient's name.")
        else:
            with st.spinner("Analyzing..."):
                files = {"file": uploaded_file.getvalue()}
                data = {
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender,
                    "medical_history": medical_history,
                    "symptoms": symptoms,
                    "user_id": user["id"],  # doctor/nurse ID from Supabase
                }
                response = requests.post(API_URL, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()

                    # Prediction and confidence as %
                    st.success(f"Prediction: **{result['class']}**")
                    st.write(f"Confidence: {result['confidence'] * 100:.2f}%")

                    # All class probabilities in %
                    probs = {
                        label: f"{val * 100:.2f}%"
                        for label, val in result["all_probabilities"].items()
                    }
                    st.write("**Class Probabilities:**")
                    st.table(probs)

                    # Grad-CAM overlay
                    st.image(result["gradcam_url"], caption="Grad-CAM Overlay")

                else:
                    st.error(f"Error: {response.text}")
