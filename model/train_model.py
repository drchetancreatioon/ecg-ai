
import os
import wfdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ecg_model import ECGClassifier

def download_and_train():
    # 1. Download PTB-XL Dataset (Small version first for testing)
    print("📥 Downloading PTB-XL data...")
    if not os.path.exists('data/ptbxl'):
        # This downloads the metadata and signal files directly from PhysioNet
        wfdb.dl_database('ptb-xl', dl_dir='data/ptbxl', 
                         keep_subdirs=True, 
                         overwrite=False)

    # 2. Load Metadata
    path = 'data/ptbxl/'
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    
    # 3. Simple Label Mapping (Normal vs MI for this example)
    # Mapping NORM to 0, MI to 1, etc.
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key == 'NORM': return 0
            if key == 'MI': return 1
        return 2 # Others
    
    import ast
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y['label'] = Y.scp_codes.apply(aggregate_diagnostic)

    # 4. Training Loop (Minimal for Demo)
    print("🧠 Training Model...")
    model = ECGClassifier(num_classes=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create dummy signal for the .pth file generation if data download is slow
    # In a real run, you'd loop through wfdb.rdsamp(path + f)
    dummy_input = torch.randn(1, 12, 1000)
    dummy_label = torch.tensor([0])

    for epoch in range(1):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_label)
        loss.backward()
        optimizer.step()

    # 5. Save the Brain!
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), "model/ecg_model.pth")
    print("✅ Success: model/ecg_model.pth created!")

if __name__ == "__main__":
    download_and_train()
