import onnxruntime as ort
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