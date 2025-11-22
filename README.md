# InSight: Automated Detection of Cataracts in Ocular Fundus Images using a Convolutional Neural Network

## 1. Project Overview

**InSight** is a secure, multi-user web application designed to assist in the early detection of cataracts. It provides a platform for clinicians to upload retinal fundus images, receive an immediate AI-powered classification and manage patient screening history.

The system is built using python as a modern web service, featuring a **Streamlit** frontend, a **FastAPI** backend for machine learning inference and a **Supabase (PostgreSQL)** database for secure user authentication and data storage.

## 2. Objectives

* Develop a secure, multi-user web portal for cataract screening.
* Implement role-based access control (RBAC) for "Nurse" and "Doctor" roles.
* Provide real-time AI classification of fundus images (Cataract / No Cataract).
* Generate professional, exportable PDF reports for each screening.
* Store and retrieve patient screening history from a secure cloud database.
* Offer a high-level analytics dashboard for "Doctor" users to track screening metrics.

## 3. Key Features

* **Secure Authentication:** Users must log in or sign up to access the system.
* **Role-Based Access:**
    * **Nurses** can screen patients, view history and generate reports.
    * **Doctors** can do all of the above and access an analytics dashboard.
* **AI-Powered Screening:** Upload an image and get an instant prediction and Grad-CAM visualization.
* **Patient History:** Search for any patient by their ID to view all past screening results.
* **PDF Report Generation:** Download a professional, multi-page PDF for each screening, including patient details, results and both the original and Grad-CAM images.
* **Analytics Dashboard:** A doctor-exclusive page showing screening totals, detection rates and other key metrics with interactive charts.

## 4. Tools & Technologies

* **Frontend:** Streamlit
* **Backend (ML Model):** Python, FastAPI (assumed), TensorFlow / Keras
* **Database & Auth:** Supabase (PostgreSQL)
* **Core Python Libraries:** Pandas, Plotly (for charts), FPDF (for PDFs)
* **Version Control:** Git & GitHub

## 5. Status

This project is in active development, with core features for user management, AI screening and reporting now implemented.

## 6. Author

* **Njuguna Faith Nyambura**
* Student ID: 150325

  

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/fY9FAi32)  
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=19909051&assignment_repo_type=AssignmentRepo)  

### Resources  
- [Git/Github Cheatsheet](https://philomatics.com/git-cheatsheet-release)  
