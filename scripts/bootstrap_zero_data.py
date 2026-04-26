import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(4, 1, 3), nn.Sigmoid())
    def forward(self, x): return self.dec(self.enc(x))

def bootstrap():
    import onnx
    import skl2onnx
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx import convert_sklearn
    from sklearn.linear_model import LogisticRegression

    print("🚀 Bootstrapping Zero-Data Test Environment...")
    
    # 1. Create Dummy IEP2 (XGBoost) ONNX Model
    model_path = Path("iep2/models/xgboost_classifier.onnx")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    dummy_lr = LogisticRegression()
    dummy_lr.fit(np.random.rand(2, 41), [0, 1])
    
    initial_type = [('float_input', FloatTensorType([None, 41]))]
    onx = convert_sklearn(dummy_lr, initial_types=initial_type, target_opset=12)
    with open(model_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✅ Created Dummy XGBoost: {model_path}")

    # 2. Create Dummy IEP4 (CNN Autoencoder) Torch Model
    ae_path = Path("iep2/models/autoencoder_ood.pt")
    torch.save(TinyAE(), ae_path)
    np.save("iep2/models/autoencoder_threshold.npy", np.array(0.5))
    print(f"✅ Created Dummy Autoencoder: {ae_path}")

    # 3. Create Label Map
    with open("iep2/models/label_map.json", "w") as f:
        json.dump({"0": "Leak", "1": "No_Leak", "_decision_threshold": 0.5}, f)
    
    # 4. Create dummy Centroid
    np.save("iep2/models/centroid.npy", np.random.rand(39))

    print("\n🔥 Environment Unblocked. You can now run the services.")

if __name__ == "__main__":
    import subprocess
    print("Ensuring dependencies...")
    subprocess.run(["pip", "install", "onnx", "onnxruntime", "skl2onnx", "scikit-learn", "--quiet"], check=True)
    bootstrap()
