from supabase import create_client
import streamlit as st

# Supabase credentials
SUPABASE_URL = "https://lbzmscmfrbrejszeuxey.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxiem1zY21mcmJyZWpzemV1eGV5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg0NTU4NDIsImV4cCI6MjA3NDAzMTg0Mn0.xggsC9HVrLKu5qeEvyzPqWud2Jq97xaRtlZHG0Hend8"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def signup(email, password, role="doctor"):
    """Register a new user and save role in profiles table"""
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
        })

        if response.user:
            # Insert role into profiles table
            supabase.table("profiles").insert({
                "id": response.user.id,   # user UUID
                "role": role,
                "full_name": email.split("@")[0]  # optional
            }).execute()

            return True, "Account created successfully!"
        else:
            return False, "Signup failed: No user returned."

    except Exception as e:
        return False, f"Signup failed: {str(e)}"


def login(email, password):
    """Login an existing user"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })

        if response.user:
            return True, response.user
        else:
            return False, "Invalid login credentials."

    except Exception as e:
        return False, f"Login failed: {str(e)}"
