import streamlit as st
import numpy as np
import google.generativeai as genai
from model.inference import predict_ecg

# --- CONFIG ---
st.set_page_config(page_title="ECG AI Interpreter", page_icon="🩺")
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "YOUR_KEY_HERE"))

st.title("🩺 Advanced ECG AI Interpreter")
st.markdown("---")

uploaded_file = st.file_uploader("Upload ECG Data (.npy format)", type=["npy"])

if uploaded_file:
    # Load data
    data = np.load(uploaded_file)
    
    if data.shape == (12, 1000):
        st.line_chart(data[0]) # Show Lead I for visualization
        
        if st.button("Analyze ECG"):
            with st.spinner("Deep Learning Model Analyzing..."):
                label, conf = predict_ecg(data)
                
            st.subheader(f"Analysis Result: {label}")
            st.info(f"Confidence Score: {conf*100:.2f}%")
            
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
