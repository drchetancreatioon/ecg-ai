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

# --- ROBUST MODEL SELECTOR ---
def get_best_model():
    """Automatically finds a working model to prevent 404 errors."""
    try:
        # Get list of models your key has access to
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority order for 2026
        targets = [
            'models/gemini-2.5-flash', 
            'models/gemini-2.0-flash', 
            'models/gemini-1.5-flash-latest' # The -latest alias is more stable than the base name
        ]
        
        for t in targets:
            if t in available:
                return t
        return available[0] if available else 'models/gemini-pro'
    except Exception:
        return 'models/gemini-1.5-flash-latest' # Absolute fallback

# --- UI HEADER ---
st.title("🩺 Advanced ECG AI Interpreter")
st.markdown("---")

# Display active model in sidebar for transparency
active_model = get_best_model()
st.sidebar.success(f"Connected to: {active_model.split('/')[-1]}")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload ECG Photo or Data File", type=["npy", "jpg", "jpeg", "png"])

if uploaded_file:
    data = None
    is_valid = False

    # 1. IMAGE HANDLING
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_container_width=True)
        
        with st.status("Digitizing image..."):
            img_array = np.array(image.convert('L'))
            resized = cv2.resize(img_array, (1000, 12))
            # Preprocessing for CNN
            data = (resized.astype(np.float32) / 127.5) - 1.0
            is_valid = True

    # 2. NUMPY HANDLING
    elif uploaded_file.name.endswith(".npy"):
        data = np.load(uploaded_file)
        if data.shape == (12, 1000):
            st.line_chart(data[0]) 
            is_valid = True
        else:
            st.error("Invalid shape! Need (12, 1000).")

    # 3. ANALYSIS
    if is_valid:
        if st.button("🚀 Run AI Interpretation"):
            col1, col2 = st.columns(2)

            # Deep Learning Prediction
            with st.spinner("Analyzing waveforms..."):
                label, conf = predict_ecg(data)
            
            with col1:
                st.subheader("🤖 CNN Diagnosis")
                st.metric(label="Condition", value=label)
                st.progress(conf)
                st.write(f"Confidence: {conf*100:.1f}%")

            # Gemini Interpretation
            with col2:
                st.subheader("🧠 Clinical Insight")
                with st.spinner("Generating report..."):
                    try:
                        model = genai.GenerativeModel(active_model)
                        prompt = (
                            f"An ECG was classified as '{label}' with {conf*100:.1f}% confidence. "
                            f"Provide a clinical explanation, its significance, and next steps."
                        )
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Gemini Error: {e}")
                        # --- ADMIN SECTION (Hidden in Sidebar) ---
           with st.sidebar.expander("🛠️ Admin: System Initialization"):
    st.write("If you see 'model.pth not found', click below.")
    if st.button("Initialize & Train Model"):
        with st.spinner("Downloading data & training... this takes a moment."):
            try:
                from model.train_model import download_and_train
                download_and_train()
                st.success("Model initialized! You can now analyze ECGs.")
            except Exception as e:
                st.error(f"Initialization Failed: {e}")
