from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

def quantize_model(model_path):
    quant_path = model_path.with_name(model_path.stem + ".int8.onnx")
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(quant_path),
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized: {quant_path.name} ({model_path.parent})")

# Model directories
model_dirs = ["onnx/ar-en", "onnx/en-ar"]

# Quantize each encoder and decoder
for model_dir in model_dirs:
    encoder = Path(model_dir) / "encoder_model.onnx"
    decoder = Path(model_dir) / "decoder_model.onnx"
    quantize_model(encoder)
    quantize_model(decoder)
