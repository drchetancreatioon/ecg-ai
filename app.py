import streamlit as st
import numpy as np
import google.generativeai as genai
from model.inference import predict_ecg

# --- CONFIG ---
st.set_page_config(page_title="ECG AI Interpreter", page_icon="🩺")
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "YOUR_KEY_HERE"))

st.title("🩺 Advanced ECG AI Interpreter")
st.markdown("---")

import cv2
from PIL import Image

uploaded_file = st.file_uploader("Upload ECG Photo or Data", type=["npy", "jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        # --- NEW IMAGE PROCESSING LOGIC ---
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_container_width=True)
        
        # Convert image to grayscale and resize to match model expected length
        img_array = np.array(image.convert('L')) # Grayscale
        resized = cv2.resize(img_array, (1000, 12)) # Force into 12 leads x 1000 samples
        
        # Normalize pixel values (0-255) to signal values (-1 to 1)
        data = (resized.astype(np.float32) / 127.5) - 1.0
        st.warning("⚠️ Note: Image-to-signal conversion is an approximation.")
    else:
        # Load standard numpy data
        data = np.load(uploaded_file)

    if st.button("Analyze ECG"):
        label, conf = predict_ecg(data)
        # ... (rest of your analysis code)
            
            # Gemini Clinical Explanation
            with st.spinner("Gemini generating clinical report..."):
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"The patient's ECG was classified as '{label}'. Explain this condition, its significance in a clinical setting, and common next steps for a physician."
                response = model.generate_content(prompt)
                
            st.markdown("### 📋 Clinical Insight (Gemini AI)")
            st.write(response.text)
    else:
        st.error("Invalid shape! Please upload a 12-lead signal with 1000 samples (12, 1000).")

st.sidebar.info("This tool uses a Hybrid CNN + LLM architecture for medical signal analysis.")
