import streamlit as st
import google.generativeai as genai
from utils.dataset_loader import preprocess_signal
from model.inference import get_prediction

# Configure Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")

st.title("🩺 Advanced AI ECG Interpreter")
st.write("Hybrid System: Deep Learning Classification + Gemini Clinical Insight")

uploaded_file = st.file_uploader("Upload ECG Signal (Numpy or CSV format)", type=['npy', 'csv'])

if uploaded_file:
    # 1. Load and Preprocess
    # (Simulated for example: loading a 12x1000 signal)
    ecg_data = np.load(uploaded_file) 
    tensor = preprocess_signal(ecg_data)
    
    # 2. DL Model Prediction
    label, conf = get_prediction(tensor)
    
    st.success(f"**DL Diagnosis:** {label} ({conf*100:.1f}% Confidence)")

    # 3. Gemini Explanation
    if st.button("Generate Clinical Interpretation"):
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"The patient's 12-lead ECG was analyzed by a CNN model and classified as {label}. Provide a detailed medical explanation for this finding, potential symptoms, and common clinical next steps."
        
        response = model.generate_content(prompt)
        st.markdown("### 🧠 Gemini Interpretation")
        st.write(response.text)
