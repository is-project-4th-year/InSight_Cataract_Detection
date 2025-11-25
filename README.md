# InSight: AI-Powered Cataract Screening System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Backend](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Database](https://img.shields.io/badge/Supabase-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


---

## 1. Project Overview

**InSight** is a secure, multi-user web application designed to revolutionize early cataract detection in low-resource settings. By leveraging a **ResNet-18 Convolutional Neural Network (CNN)**, the system analyzes retinal fundus images to provide real-time diagnostic classification ("Cataract" vs. "Normal").

Beyond simple prediction, InSight prioritizes **clinical interpretability** through Grad-CAM visualizations and ensures **data security** via Role-Based Access Control (RBAC).

---

## 2. The Problem & The Gap

Cataracts remain the leading cause of preventable blindness worldwide, disproportionately affecting regions with a shortage of ophthalmologists.

| The Challenge | The InSight Solution |
| :--- | :--- |
| **Late Diagnosis** | Automated screening allows for rapid triage by nurses, not just specialists. |
| **"Black Box" AI** | Most AI tools provide a diagnosis without explanation. InSight uses **Grad-CAM** to show *where* the model is looking. |
| **Workflow Friction** | Existing models are often isolated scripts. InSight is a full-stack **Clinical Decision Support System (CDSS)** with PDF reporting and patient history. |
| **Invalid Inputs** | Standard models crash or hallucinate on wrong images. InSight includes a **MobileNetV2 "Gatekeeper"** to reject non-medical uploads (e.g., cars, faces). |

---

## 3. System Architecture & Methodology

InSight employs a microservices architecture to separate the heavy ML inference from the user interface.

### The AI Engine
* **Architecture:** ResNet-18 (Residual Network) with Transfer Learning.
* **Training Strategy:** Trained on the **ODIR-5K dataset**.
* **Loss Function:** Utilized **Focal Loss** (gamma=2.0) to handle severe class imbalance, forcing the model to learn from hard-to-classify cataract cases rather than overfitting on healthy eyes.
* **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) generates a heatmap overlay, highlighting the lens opacity that triggered the diagnosis.

### Tech Stack
* **Frontend:** Streamlit (Python) for a responsive clinician dashboard.
* **Backend:** FastAPI (Asynchronous inference engine).
* **Database:** Supabase (PostgreSQL) for secure auth and persistent storage.
* **Validation:** MobileNetV2 pre-validation layer for input integrity.

---

## 4. Performance Metrics

The model was evaluated on a held-out test set of 1,098 images, achieving state-of-the-art performance suitable for screening purposes.

| Metric | Score | Clinical Significance |
| :--- | :--- | :--- |
| **Accuracy** | **99.52%** | High reliability for general screening. |
| **Recall (Sensitivity)** | **99.4%** | Critical for medicine; ensures we rarely miss a positive cataract case. |
| **Precision** | **99.0%** | Minimizes false alarms (healthy patients sent to doctors). |
| **F1-Score** | **0.992** | Harmonic balance between precision and recall. |
| **AUC-ROC** | **0.998** | Excellent discrimination capability. |

---

## 5. Key Features

### Secure & Role-Based
* **Authentication:** Secure email/password login via Supabase.
* **RBAC:**
    * **Nurse Role:** Upload scans, Batch processing, View history, Generate PDFs.
    * **Doctor Role:** All Nurse features + Access to the **Analytics Dashboard**.

### Smart Screening
* **Batch Processing:** Upload 50+ images at once. The system auto-extracts Patient IDs from filenames and processes the queue.
* **Input Validation:** The system automatically rejects invalid images (e.g., selfies, objects) before processing.

### Professional Reporting
* **PDF Generation:** Auto-generates a formal medical referral report containing:
    * Patient Details & Timestamp.
    * Diagnosis & Confidence Score.
    * Original Scan + Grad-CAM Saliency Map.
    * Clinical Disclaimer.

### Doctor's Analytics
* Interactive dashboard visualizing screening volumes, positivity rates, and model confidence distributions to track hospital throughput.

---

## 6. Installation & Setup

To run InSight locally, follow these steps:

### Prerequisites
* Python 3.9+
* Git

### Steps
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/InSight.git](https://github.com/YourUsername/InSight.git)
    cd InSight
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Secrets**
    Create a `.streamlit/secrets.toml` file and add your Supabase credentials:
    ```toml
    [supabase]
    url = "your-supabase-url"
    key = "your-supabase-anon-key"
    ```

4.  **Run the Backend (Inference Engine)**
    ```bash
    uvicorn backend_app:app --reload
    ```

5.  **Run the Frontend (Dashboard)**
    ```bash
    streamlit run app.py
    ```

---

## 7. Project Status

* **Completed:** Core AI Model, Web Interface, Database Integration, PDF Reporting, Batch Processing.
* **In Progress:** Integration with hospital EMR APIs, Mobile App development.

---

## 8. Author

**Njuguna Faith Nyambura**
* **Student ID:** 150325
* **Role:** Lead Developer & AI Researcher

---
*Disclaimer: InSight is a Clinical Decision Support System (CDSS) and is not intended to replace professional medical diagnosis. All results should be verified by a qualified ophthalmologist.*
