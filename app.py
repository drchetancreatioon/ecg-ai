import streamlit as st
import numpy as np
import google.generativeai as genai
import cv2
from PIL import Image
from model.inference import predict_ecg

# --- CONFIGURATION ---
st.set_page_config(page_title="ECG AI Interpreter", page_icon="🩺", layout="wide")

# API Key Handling (Priority: Streamlit Secrets > Manual Input)
api_key = st.secrets.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=api_key)

# --- UI HEADER ---
st.title("🩺 Advanced ECG AI Interpreter")
st.markdown("""
This tool uses a **Deep Learning CNN** to classify ECG signals and **Google Gemini 1.5** to provide clinical explanations. 
Upload a digital ECG file (.npy) or a clear photo of an ECG strip.
""")
st.sidebar.header("Settings & Info")
st.sidebar.info("Architecture: 1D-CNN (PyTorch) + Gemini 1.5 Flash")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload ECG Photo or Data File", type=["npy", "jpg", "jpeg", "png"])

if uploaded_file:
    data = None
    is_valid = False

    # 1. HANDLE IMAGE UPLOAD
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG Image", use_container_width=True)
        
        with st.status("Processing image into signal..."):
            # Convert to grayscale
            img_array = np.array(image.convert('L'))
            # Resize to 12 leads (rows) and 1000 time-steps (cols)
            resized = cv2.resize(img_array, (1000, 12))
            # Normalize pixel values (0-255) to signal range (-1.0 to 1.0)
            data = (resized.astype(np.float32) / 127.5) - 1.0
            is_valid = True
            st.warning("⚠️ Image digitized. Accuracy depends on photo clarity and lead alignment.")

    # 2. HANDLE NUMPY DATA UPLOAD
    elif uploaded_file.name.endswith(".npy"):
        data = np.load(uploaded_file)
        if data.shape == (12, 1000):
            st.success("Digital signal loaded successfully (12 leads, 1000 samples).")
            st.line_chart(data[0]) # Visualize Lead I
            is_valid = True
        else:
            st.error(f"Invalid Data Shape: {data.shape}. Expected (12, 1000).")

    # 3. ANALYSIS TRIGGER
    if is_valid:
        if st.button("🚀 Analyze ECG Now"):
            col1, col2 = st.columns(2)

            # Deep Learning Prediction
            with st.spinner("Deep Learning Model Analyzing..."):
                label, conf = predict_ecg(data)
            
            with col1:
                st.subheader("🤖 AI Diagnosis")
                st.metric(label="Predicted Condition", value=label)
                st.progress(conf)
                st.write(f"Confidence Score: {conf*100:.2f}%")

            # Gemini Clinical Interpretation
            with col2:
                st.subheader("🧠 Clinical Insight")
                with st.spinner("Gemini generating report..."):
                    try:
                        model = genai.GenerativeModel('gemini-1.5-pro')
                        prompt = (
                            f"The patient's ECG was classified as '{label}' by a Deep Learning model "
                            f"with {conf*100:.1f}% confidence. Provide a concise medical explanation "
                            f"of this finding, including pathophysiology and common clinical next steps."
                        )
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Gemini API Error: {e}")

else:
    st.info("Waiting for file upload...")

# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This is an AI research tool. Not for final medical diagnostic use. Consult a cardiologist.")
