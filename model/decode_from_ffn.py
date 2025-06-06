#!/usr/bin/env python3

import argparse, re, numpy as np, torch
from pathlib import Path
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModel,
    MarianTokenizer,
    modeling_outputs,
)
import onnxruntime as ort

from test_extracted_weights import (
    multi_head_attention_forward
)

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar", torch_dtype=torch.float32)


# ─────────────────────────── CLI ────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--ffn_txt", required=True,
               help="Path to FFN output text (256 Q1.7 ints)")
p.add_argument("--weights_dir", required=True,
               help="Directory containing extracted weights")
p.add_argument("--model_name", default="Helsinki-NLP/opus-mt-en-ar")
p.add_argument("--max_len", type=int, default=6)
p.add_argument("--onnx_dir", required=True,
               help="Directory containing encoder_model.int8.onnx and decoder_model.int8.onnx")
args = p.parse_args()

# Hardcoded original sentence
ORIGINAL_SENTENCE = "My name is Fahad"

# ─────────────────── Load Quantized ONNX Models ──────────────────
def load_onnx_session(model_path):
    return ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )

print("\n🔍 Loading Quantized ONNX Models")
print("───────────────────────────────")
encoder_onnx = load_onnx_session(f"{args.onnx_dir}/en-ar/encoder_model.int8.onnx")
decoder_onnx = load_onnx_session(f"{args.onnx_dir}/en-ar/decoder_model.int8.onnx")
tokenizer = MarianTokenizer.from_pretrained(args.model_name)

# ─────────────────── Load Quantized Weights ──────────────────
def load_quantized_weights(weights_dir, name, shape):
    """Load weights in same format as extract_weights.py outputs"""
    weight_path = Path(weights_dir) / f"{name}.bin"
    scale_path = Path(weights_dir) / f"{name}.scale"
    zp_path = Path(weights_dir) / f"{name}.zp"
    
    weights = np.fromfile(weight_path, dtype=np.int8).reshape(shape)
    scale = np.fromfile(scale_path, dtype=np.float32)
    zp = np.fromfile(zp_path, dtype=np.uint8)
    return weights.astype(np.float32) * scale + zp

# Load weights (only needed for attention calculations)
Wq = [load_quantized_weights(args.weights_dir, f"query_head{i}", (64, 256)) for i in range(4)]
Wk = [load_quantized_weights(args.weights_dir, f"key_head{i}", (64, 256)) for i in range(4)]
Wv = [load_quantized_weights(args.weights_dir, f"value_head{i}", (64, 256)) for i in range(4)]
Wo = load_quantized_weights(args.weights_dir, "attn_out_proj", (256, 256))
W1 = load_quantized_weights(args.weights_dir, "ffn_fc1", (512, 256))
W2 = load_quantized_weights(args.weights_dir, "ffn_fc2", (256, 512))

# ─────────────────────── Tokenization ───────────────────────
print("\n🔍 Original Input Analysis")
print("───────────────────────────")
print(f"Original sentence: '{ORIGINAL_SENTENCE}'")

input_ids = tokenizer(ORIGINAL_SENTENCE, return_tensors="np")["input_ids"]
original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

residual_full = np.concatenate([original_tokens, original_tokens], axis=-1)  # (seq_len, 512)


# ─────────────────── Get Original Embeddings ─────────────────────
print("\n🔍 Running Quantized Encoder")
print("───────────────────────────")
# Run through quantized encoder to get reference embeddings
encoder_inputs = {
    "input_ids": input_ids.astype(np.int64),
    "attention_mask": np.ones_like(input_ids, dtype=np.int64)
}
enc_out = encoder_onnx.run(None, encoder_inputs)[0]

print("\nQuantized Encoder Output (first 16 dims of first token):")
print(" ".join([f"{x:.4f}" for x in enc_out[0, 0, :16]]))

# def get_quantized_positional_encodings(onnx_session, seq_len=4):
#     """Extract positional encodings from quantized ONNX model"""
#     # Try common names for positional embeddings in quantized models
#     pos_emb_names = [
#         "encoder.embed_positions.weight_quantized",
#         "encoder.embed_positions.weight",
#         "embed_positions.weight_quantized",
#         "embed_positions.weight"
#     ]
    
#     # Search through model initializers
#     for initializer in onnx_session.get_overridable_initializers():
#         if initializer.name in pos_emb_names:
#             # Get quantized values and dequantize
#             weights_int8 = np.frombuffer(initializer.raw_data, dtype=np.int8)
#             weights_int8 = weights_int8.reshape(initializer.shape)
            
#             # Find corresponding scale and zero point
#             scale_name = initializer.name.replace("_quantized", "_scale")
#             zp_name = initializer.name.replace("_quantized", "_zero_point")
            
#             scale = next(i for i in onnx_session.get_overridable_initializers() 
#                        if i.name == scale_name).float_data[0]
#             zp = next(i for i in onnx_session.get_overridable_initializers()
#                      if i.name == zp_name).int32_data[0]
            
#             # Dequantize
#             pe = (weights_int8.astype(np.float32) - zp) * scale
#             return pe[:seq_len]  # Return first 'seq_len' positions

# # Usage:
# print("\n[Debug] Available initializers in ONNX encoder:")
# for init in encoder_onnx.get_overridable_initializers():
#     print(f"  – {init.name}  (shape={tuple(init.shape)})")
# print()
# pos_enc = get_quantized_positional_encodings(encoder_onnx, seq_len=4)

print("🔍 Extracting True Positional Encodings")
print("────────────────────────────────────────")
true_pe = {}

def hook_fn(module, input, output):
    true_pe["with_pos"] = input[0].detach().cpu().numpy()

hook = model.model.encoder.layers[0].register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model.model.encoder(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.ones_like(torch.tensor(input_ids), dtype=torch.long)
    )

input_ids=torch.tensor(input_ids, dtype=torch.long)

hook.remove()

token_only = model.model.shared(input_ids).detach().cpu().numpy()
true_pos_enc = true_pe["with_pos"] - token_only

print("True PE shape:", true_pos_enc.shape)
print("First token pos enc:", true_pos_enc[0, 0, :16])

tok  = (model.model.encoder
        .embed_tokens
        .weight[input_ids[0]]
        .detach()
        .cpu()
        .numpy())    

# ─────────────────── Parse VHDL FFN Output ───────────────────
print("\n🔍 Parsing VHDL FFN Output")
print("──────────────────────────")
ints = []
with open(args.ffn_txt) as fh:
    for ln in fh:
        if ln.lstrip().startswith("--"):
            continue
        ints.extend(re.findall(r"-?\d+", ln))
ints = [int(v) for v in ints]               # Q1.7 signed ints
print(f"Read {len(ints)} values from FFN file")
assert len(ints) % 256 == 0, "Expected N×256 ints"

tok_count = len(ints) // 256
print(f"Token count: {tok_count}")
ffn_mat = (np.array(ints, np.float32).reshape(tok_count, 256) / 128.0)
print(f"FFN matrix shape: {ffn_mat.shape}")

# ─────────────────── Compare Embeddings ───────────────────
print("\n🔍 Comparing Encoder vs VHDL Embeddings")
print("───────────────────────────────────────")
print(f"Encoder output shape: {enc_out.shape}")
print(f"VHDL output shape: {ffn_mat.shape}")
print("First 16 dimensions comparison:")
for token_idx in range(min(enc_out.shape[1], ffn_mat.shape[0])):
    print(f"\nToken {token_idx} ('{original_tokens[token_idx]}'):")
    enc_vec  = enc_out[0, token_idx, :16]
    residual_256 = tok[:4, :256] + true_pos_enc[0,:4, :256]
    vhdl_vec = ffn_mat[token_idx, :16]
    enc_q11  = [int(np.clip(round(x * 128), -128, 127)) for x in enc_vec]
    vhdl_q11 = [int(np.clip(round(x * 128), -128, 127)) for x in vhdl_vec]

    residual_full = np.concatenate([residual_256, residual_256], axis=-1) / 2
    vhdl_full = np.concatenate([ffn_mat[token_idx], ffn_mat[token_idx]], axis=-1) / 2
    input_full = (vhdl_full + residual_full)

    residual_vec = residual_full[token_idx, :16]       # shape: (16,)
    residual_q11 = [int(np.clip(x * 128, -128, 127)) for x in residual_vec]

    input_vec = input_full[token_idx, :16]
    input_q11 = [int(np.clip(x * 128, -128, 127)) for x in input_vec]

    print("\n  Encoder (fp32): " + " ".join(f"{x:8.4f}" for x in enc_vec))
    print("\n  VHDL    (fp32): " + " ".join(f"{x:8.4f}" for x in vhdl_vec))
    print("\n  Residual (FP32): " + " ".join(f"{x:8.4f}"   for x in residual_vec))   
    print("\n  Input Full    (fp32):  " + " ".join(f"{x:8.4f}" for x in input_vec))
    print("\n  Encoder (Q11):  " + " ".join(f"{x:6d}"   for x in enc_q11))
    print("\n  VHDL    (Q11):  " + " ".join(f"{x:6d}"   for x in vhdl_q11))
    print("\n  Residual (Q11): " + " ".join(f"{x:6d}"   for x in residual_q11))   
    print("\n  Input Full    (Q11):  " + " ".join(f"{x:6d}" for x in input_q11))

# ─────────────────── Decoder Processing ───────────────────
print("\n🔍 Running Quantized Decoder with VHDL Output")
print("─────────────────────────────────────────────")

# Prepare inputs for quantized decoder
decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
encoder_hidden_states = input_full[np.newaxis, ...]  # Add batch dimension
encoder_attention_mask = np.ones((1, input_full.shape[0]), dtype=np.int64)

generated_ids = [tokenizer.pad_token_id]
for _ in range(args.max_len):
    decoder_outputs = decoder_onnx.run(None, {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
        "encoder_attention_mask": encoder_attention_mask
    })
    next_token = decoder_outputs[0][0, -1, :].argmax()
    generated_ids.append(next_token)
    decoder_input_ids = np.hstack([decoder_input_ids, [[next_token]]])
    if next_token == tokenizer.eos_token_id:
        break

# Get reference translation from quantized decoder
ref_decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
ref_generated_ids = [tokenizer.pad_token_id]
for _ in range(args.max_len):
    ref_decoder_outputs = decoder_onnx.run(None, {
        "input_ids": ref_decoder_input_ids,
        "encoder_hidden_states": enc_out.astype(np.float32),
        "encoder_attention_mask": encoder_inputs["attention_mask"]
    })
    next_token = ref_decoder_outputs[0][0, -1, :].argmax()
    ref_generated_ids.append(next_token)
    ref_decoder_input_ids = np.hstack([ref_decoder_input_ids, [[next_token]]])
    if next_token == tokenizer.eos_token_id:
        break

# ─────────────────── Output Comparison ───────────────────
print("\n🔍 Translation Results")
print("──────────────────────")
ref_translation_text = tokenizer.decode(ref_generated_ids, skip_special_tokens=True)
vhdl_translation_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"Reference Translation: {ref_translation_text}")
print(f"VHDL FFN Translation: {vhdl_translation_text}")

print("\nToken-level comparison:")
print(f"Reference tokens: {ref_generated_ids}")
print(f"VHDL FFN tokens: {generated_ids}")

# ─────────────────── Debug Output ───────────────────
print("\n🔍 Detailed Debug Information")
print("─────────────────────────────")
print("FFN output statistics:")
print(f"  Mean: {ffn_mat.mean():.6f}")
print(f"  Std: {ffn_mat.std():.6f}")
print(f"  Min: {ffn_mat.min():.6f}")
print(f"  Max: {ffn_mat.max():.6f}")

print("\nFirst 16 values of FFN output for first token:")
for i in range(0, 16, 4):
    print("  " + " ".join([f"{x:10.6f}" for x in ffn_mat[0, i:i+4]]))