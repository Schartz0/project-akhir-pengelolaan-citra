"""
Jalankan sekali di lokal untuk konversi model PyTorch ke ONNX:
    python convert_to_onnx.py
"""
import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH  = "best_coral_model.pth"
OUTPUT_PATH = "best_coral_model.onnx"

device = torch.device("cpu")

# Bangun arsitektur yang sama persis seperti saat training
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Dummy input sesuai ukuran yang dipakai saat training (224x224)
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    OUTPUT_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)

print(f"Model berhasil dikonversi ke: {OUTPUT_PATH}")

# Verifikasi
import onnx
onnx_model = onnx.load(OUTPUT_PATH)
onnx.checker.check_model(onnx_model)
print("Verifikasi ONNX: OK")
