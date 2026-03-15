import torch
import numpy as np
from .ecg_model import ECGClassifier

# These classes match the PTB-XL superclass categories
CLASSES = [
    "Normal (NORM)", 
    "Myocardial Infarction (MI)", 
    "ST/T Change (STTC)", 
    "Conduction Disturbance (CD)", 
    "Hypertrophy (HYP)"
]

def predict_ecg(signal_data, model_path="model/ecg_model.pth"):
    """
    Predicts the ECG category using the trained CNN model.
    Args:
        signal_data (np.array): A 12x1000 numpy array
        model_path (str): Path to the saved .pth model weights
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize Model Architecture
    model = ECGClassifier(num_classes=5).to(device)
    
    # 3. Load Weights safely
    try:
        # map_location ensures it loads even if trained on GPU but running on CPU
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        return "Error: model/ecg_model.pth not found. Please upload weights.", 0.0
    except Exception as e:
        return f"Error loading model: {str(e)}", 0.0
        
    model.eval()

    # 4. Data Transformation
    # Ensure data is float32 and has batch dimension: (1, 12, 1000)
    if not isinstance(signal_data, torch.Tensor):
        signal_tensor = torch.tensor(signal_data, dtype=torch.float32)
    
    if signal_tensor.ndimension() == 2:
        signal_tensor = signal_tensor.unsqueeze(0)
    
    signal_tensor = signal_tensor.to(device)
    
    # 5. Inference
    with torch.no_grad():
        output = model(signal_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][pred_idx].item()
        
    return CLASSES[pred_idx], confidence
