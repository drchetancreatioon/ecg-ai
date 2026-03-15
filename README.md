# 🩺 Hybrid ECG AI Interpreter

This project combines a **1D-CNN Deep Learning Model** (trained on PhysioNet's PTB-XL) with **Google Gemini AI** to provide both a diagnosis and a clinical explanation.

## 🚀 Setup
1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place your trained `ecg_model.pth` in the `model/` folder.
4. Run the app: `streamlit run app.py`.

## 🛠️ Tech Stack
- **PyTorch**: Deep Learning Inference.
- **Streamlit**: Web Interface.
- **Gemini 1.5 Flash**: Medical Explanation.
