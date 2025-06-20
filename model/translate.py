import onnxruntime as ort
import onnx
from onnx import numpy_helper
import numpy as np
from transformers import MarianTokenizer
import os
import time
import sys

# === Global translator session cache ===
loaded_sessions = {}

def load_sessions(model_path):
    if model_path in loaded_sessions:
        return loaded_sessions[model_path]

    # Changed to use quantized models
    encoder = ort.InferenceSession(
        f"{model_path}/encoder_model.int8.onnx",
        providers=['CPUExecutionProvider']  # Explicitly use CPU for quantized models
    )
    decoder = ort.InferenceSession(
        f"{model_path}/decoder_model.int8.onnx",
        providers=['CPUExecutionProvider']
    )
    tokenizer = MarianTokenizer.from_pretrained(model_path)

    loaded_sessions[model_path] = (encoder, decoder, tokenizer)
    return encoder, decoder, tokenizer

def print_fc_weights(model_path):
    import onnx, os
    from onnx import numpy_helper

    model_file = os.path.join(model_path, "encoder_model.int8.onnx")
    model      = onnx.load(model_file)
    init       = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    # FC1 = (512×2048), FC2 = (2048×512)
    fc1 = next(((n,a) for n,a in init.items() if a.ndim==2 and a.shape==(512,2048)), None)
    fc2 = next(((n,a) for n,a in init.items() if a.ndim==2 and a.shape==(2048,512)), None)

    if fc1:
        name, arr = fc1
        print(f"🔹 FC1 '{name}' first 10 values:\n  {arr.flatten()[:10]}")
    else:
        print("⚠️  FC1 (512×2048) not found")

    if fc2:
        name, arr = fc2
        print(f"🔹 FC2 '{name}' first 10 values:\n  {arr.flatten()[:10]}")
    else:
        print("⚠️  FC2 (2048×512) not found")

def print_fp32_values(model_path):
    path = os.path.join(model_path, "encoder_model.int8.onnx")
    model = onnx.load(path)
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    # Define the exact names we found in the model
    quantized_weights = {
        'onnx::MatMul_1120_quantized': {
            'scale': 'onnx::MatMul_1120_scale',
            'zero_point': 'onnx::MatMul_1120_zero_point'
        },
        'onnx::MatMul_1121_quantized': {
            'scale': 'onnx::MatMul_1121_scale',
            'zero_point': 'onnx::MatMul_1121_zero_point'
        }
    }

    for qname, params in quantized_weights.items():
        if qname not in init:
            print(f"⚠️  Quantized weight '{qname}' missing")
            continue

        # Get quantized values
        q = init[qname].astype(np.int32)
        
        # Get scale and zero point using exact names
        scale = init.get(params['scale'])
        zp = init.get(params['zero_point'], 0)  # Default to 0 if not found
        
        if scale is None:
            print(f"⚠️  Scale not found for '{qname}', showing quantized values")
            print(f"\n🔹 {qname} → INT8 first 10 values:\n   {q.flatten()[:10]}")
            continue
            
        # Dequantize
        fp32 = (q - zp) * scale
        print(f"\n🔹 {qname} → FP32 first 10 values:")
        print(f"   Scale: {scale}, Zero-point: {zp}")
        print(f"   Values: {np.round(fp32.flatten()[:10], 6)}")
        print(f"   Shape: {init[qname].shape}, Dtype: {init[qname].dtype}")

def translate(text, model_path):
    encoder, decoder, tokenizer = load_sessions(model_path)

    # Tokenize
    enc = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    input_ids = enc["input_ids"].astype(np.int64)  # Ensure int64 input for quantized model
    attention_mask = enc["attention_mask"].astype(np.int64)

    # Encode
    start_time = time.time()
    enc_out = encoder.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0]

    # Decode
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    for _ in range(50):
        out = decoder.run(None, {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": enc_out,
            "encoder_attention_mask": attention_mask,
        })[0]
        next_token = out[0, -1, :].argmax()
        decoder_input_ids = np.hstack([decoder_input_ids, [[next_token]]])
        if next_token == tokenizer.eos_token_id:
            break

    result = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    latency = time.time() - start_time
    return result, latency

# === Session ===
def main():
    print("🌍 Arabic ↔ English Translator (type 'exit' to quit, 'ar-en' or 'en-ar' to switch direction)")

    direction = "en-ar"
    model_path = os.path.join("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/onnx_models/", direction)
    # print_fc_weights(model_path)
    # print_fp32_values(model_path)

    while True:
        user_input = input(f"[{direction}] > ").strip()

        if user_input.lower() == "exit":
            print("👋 Exiting.")
            break
        elif user_input in ["ar-en", "en-ar"]:
            direction = user_input
            model_path = os.path.join("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/onnx_models/", direction)
            print(f"🔄 Switched direction to {direction}")
            continue
        elif user_input == "":
            continue

        try:
            result, latency = translate(user_input, model_path)
            print(f"✅ Translation: {result}")
            print(f"🕒 Inference Time: {latency:.2f} sec\n")
        except Exception as e:
            print(f"❌ Translation failed: {e}")

if __name__ == "__main__":
    main()