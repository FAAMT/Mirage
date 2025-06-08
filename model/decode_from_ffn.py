#!/usr/bin/env python3

import sys, argparse, re, numpy as np, torch
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

# ─────────────────── Get Original Embeddings ─────────────────────
print("\n🔍 Running Quantized Encoder")
print("───────────────────────────")
encoder_inputs = {
    "input_ids": input_ids.astype(np.int64),
    "attention_mask": np.ones_like(input_ids, dtype=np.int64)
}

embedding_weights = model.model.shared.weight.detach().cpu().numpy()  # (vocab_size, 256)

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

# Sanity log
print("Token Embeddings (before pos-enc):")
for t in range(token_only.shape[1]):
    vec = token_only[0, t, :16]
    print(f"Token {t}: " + " ".join(f"{v:7.4f}" for v in vec))

print("\n🔍 Extracting True Positional Encodings")
print("────────────────────────────────────────")

# Step 1: Add token + pos
residual = token_only

# Step 2: Get LayerNorm weights from the encoder block
ln = model.model.encoder.layers[0].self_attn_layer_norm
gamma = ln.weight.detach().cpu().numpy()
beta = ln.bias.detach().cpu().numpy()

# Step 3: Apply LayerNorm manually (axis=-1)
mu  = residual.mean(axis=-1, keepdims=True)
var = residual.var(axis=-1, keepdims=True)
eps = 1e-6

normalized = (residual - mu) / np.sqrt(var + eps)
resid_norm = normalized * gamma + beta

print("\nResidual (token + pos-enc):")
for t in range(resid_norm.shape[1]):
    vec = resid_norm[0, t, :16]
    print(f"Token {t}: " + " ".join(f"{v:7.4f}" for v in vec))

with torch.no_grad():
    model_output = model.model.encoder.forward(
        inputs_embeds=torch.tensor(resid_norm).to(model.device),
        attention_mask=torch.ones((1, resid_norm.shape[1]), dtype=torch.long).to(model.device)
    )
test = model_output.last_hidden_state.cpu().numpy()

print("\nTest Output (first 16 dims of first token):")
print("Float values:")
print(" ".join([f"{x:.4f}" for x in test[0, 0, :16]]))

print("\nQ1.7 int8 values:")
# Convert to Q1.7 fixed-point (range -128 to 127)
q_values = np.clip(np.round(test[0, 0, :16] * 128), -128, 127).astype(np.int8)
print(" ".join([f"{x:4d}" for x in q_values]))  # :4d for alignment
enc_out = encoder_onnx.run(None, encoder_inputs)[0]

print("\nQuantized Encoder Output (first 16 dims of first token):")
print(" ".join([f"{x:.4f}" for x in enc_out[0, 0, :16]]))

# sys.exit()

# ─────────────────── Enhanced Debug Output ───────────────────
print("\n🔍 Detailed Encoder Value Breakdown")
print("──────────────────────────────────")

# Get token embeddings and positional encodings
tok = model.model.encoder.embed_tokens.weight[input_ids[0]].detach().cpu().numpy()

# Calculate intermediate values
x_enc = tok  # Token + positional encoding
eps = 1e-6
mu  = x_enc.mean(axis=-1, keepdims=True)
var = x_enc.var(axis=-1, keepdims=True)
# x_ln = (x_enc - mu) / np.sqrt(var + eps)
x_ln = x_enc

# Fix: Add batch dimension if missing (assuming your input is 2D [seq_len, hidden_dim])
if len(x_ln.shape) == 2:
    x_ln = np.expand_dims(x_ln, axis=0)   # Adds batch dimension of size 1

# Then proceed with the original attention code
batch_s, seq_l, _ = x_ln.shape

# Print calculation steps for first token
print("\nToken 0 ('{}') Calculation Steps:".format(original_tokens[0]))
print("─" * 50)

# 1. Show raw token embeddings
print("\n1. Token Embeddings (first 16 dims):")
print("Float32:  " + " ".join(f"{x:8.4f}" for x in tok[0, :16]))
print("Q1.7:     " + " ".join(f"{int(np.round(x*128)):4d}" for x in tok[0, :16]))

# # 2. Show positional encodings
# print("\n2. Positional Encodings (first 16 dims):")
# np.set_printoptions(formatter={'float': '{:8.4f}'.format})
# print("Float32:  ", pe[0, :16])
# print("Q1.7:     " + " ".join(f"{int(np.round(float(x)*128)):4d}" for x in pe[0,0,:16].flatten()))

# 3. Show combined embeddings (token + position)
print("\n3. Combined Embeddings (Token + Position):")
np.set_printoptions(formatter={'float': '{:8.4f}'.format})
print("Float32:  ", x_enc[0, :16])
print("Q1.7:     " + " ".join(f"{int(np.round(float(x)*128)):4d}" for x in x_enc[0,:16].flatten()))

# 4. Show layer normalization
print("\n4. Layer Normalization Output:")
# Fix: Ensure we're formatting scalar values by explicitly converting array elements to float
# Fix: Use numpy array formatting
np.set_printoptions(formatter={'float': '{:8.4f}'.format})
print("Float32:  ", x_ln[0, 0, :16])
print("Q1.7:     " + " ".join(f"{int(np.round(x*128)):4d}" for x in x_ln[0, 0, :16]))

# 5. Show attention output (first head)
print("\n5. Attention Output (Head 0, first 16 dims):")
# Ensure input is 3D: (1, seq_len, d_model)
attn_out = multi_head_attention_forward(x_ln[:,:,:256], Wq, Wk, Wv, Wo)
attn_out = attn_out[0] 
print("Float32:  " + " ".join(f"{x:8.4f}" for x in attn_out[0, :16]))
print("Q1.7:     " + " ".join(f"{int(np.round(x*128)):4d}" for x in attn_out[0, :16]))

# 6. Show residual addition
print("\n6. Residual Addition (Attention + Original):")
attn_full = np.concatenate([attn_out[0], attn_out[0]], axis=-1) / 1
residual_out = x_enc[0] + attn_full
np.set_printoptions(formatter={'float': '{:8.4f}'.format})
print("Float32:  ", residual_out[:16])
print("Q1.7:     " + " ".join(f"{int(np.round(x*128)):4d}" for x in residual_out[:16].flatten()))

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
print(model.model.decoder.layers[0].self_attn.k_proj.in_features)

# ─────────────────── Compare Embeddings ───────────────────
print("\n🔍 Comparing Encoder vs VHDL Embeddings")
print("───────────────────────────────────────")
print(f"Encoder output shape: {enc_out.shape}")
print(f"VHDL output shape: {ffn_mat.shape}")
print("First 16 dimensions comparison:")
token_emb = model.model.encoder.embed_tokens.weight[input_ids[0]].detach().cpu().numpy()


for token_idx in range(min(enc_out.shape[1], ffn_mat.shape[0])):
    print(f"\nToken {token_idx} ('{original_tokens[token_idx]}'):")
    enc_vec  = enc_out[0, token_idx, :16]
    residual_256 = token_emb[0, :256]
    vhdl_vec = ffn_mat[token_idx, :16]
    enc_q11  = [int(np.clip(round(x * 128), -128, 127)) for x in enc_vec]
    vhdl_q11 = [int(np.clip(round(x * 128), -128, 127)) for x in vhdl_vec]

    residual_full = np.concatenate([residual_256, residual_256], axis=-1)
    vhdl_full = np.concatenate([ffn_mat[token_idx], ffn_mat[token_idx]], axis=-1)
    input_full = (vhdl_full + residual_full)

    # Final layernorm using decoder weights
    gamma = model.model.encoder.layers[-1].final_layer_norm.weight.detach().numpy()  # (512,)
    beta  = model.model.encoder.layers[-1].final_layer_norm.bias.detach().numpy()   

    # Apply LayerNorm manually per token
    norm_input_full = []
    for t in range(input_full.shape[0]):
        vec = input_full[t]
        mu  = vec.mean()
        var = vec.var()
        normed = (vec - mu) / np.sqrt(var + 1e-6)
        final = normed * gamma + beta
        norm_input_full.append(final)

    norm_input_full = np.stack(norm_input_full)  # shape (

    residual_vec = residual_full[:16]       # shape: (16,)
    residual_q11 = [int(np.clip(x * 128, -128, 127)) for x in residual_vec]

    input_vec = norm_input_full[token_idx, :16]
    input_q11 = [int(np.clip(x * 128, -128, 127)) for x in input_vec]

    # print("\n  Encoder (fp32): " + " ".join(f"{x:8.4f}" for x in enc_vec))
    # print("\n  VHDL    (fp32): " + " ".join(f"{x:8.4f}" for x in vhdl_vec))
    # print("\n  Residual (FP32): " + " ".join(f"{x:8.4f}"   for x in residual_vec))   
    # print("\n  Input Full    (fp32):  " + " ".join(f"{x:8.4f}" for x in input_vec))
    # print("\n  Encoder (Q11):  " + " ".join(f"{x:6d}"   for x in enc_q11))
    # print("\n  VHDL    (Q11):  " + " ".join(f"{x:6d}"   for x in vhdl_q11))
    # print("\n  Residual (Q11): " + " ".join(f"{x:6d}"   for x in residual_q11))   
    # print("\n  Input Full    (Q11):  " + " ".join(f"{x:6d}" for x in input_q11))

# ─────────────────── Decoder Processing ───────────────────
print("\n🔍 Running Quantized Decoder with VHDL Output")
print("─────────────────────────────────────────────")

# Prepare inputs for quantized decoder
decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
encoder_hidden_states = norm_input_full[np.newaxis, ...]  # Add batch dimension
encoder_attention_mask = np.ones((1, norm_input_full.shape[0]), dtype=np.int64)

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

# Get reference translation from quantized decoder
test_decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
test_generated_ids = [tokenizer.pad_token_id]
for _ in range(args.max_len):
    test_decoder_outputs = decoder_onnx.run(None, {
        "input_ids": test_decoder_input_ids,
        "encoder_hidden_states": test.astype(np.float32),
        "encoder_attention_mask": encoder_inputs["attention_mask"]
    })
    next_token = test_decoder_outputs[0][0, -1, :].argmax()
    test_generated_ids.append(next_token)
    test_decoder_input_ids = np.hstack([test_decoder_input_ids, [[next_token]]])
    if next_token == tokenizer.eos_token_id:
        break

# ─────────────────── Output Comparison ───────────────────
print("\n🔍 Translation Results")
print("──────────────────────")
ref_translation_text = tokenizer.decode(ref_generated_ids, skip_special_tokens=True)
test_translation_text = tokenizer.decode(test_generated_ids, skip_special_tokens=True)
vhdl_translation_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"Reference Translation: {ref_translation_text}")
print(f"Test Translation: {test_translation_text}")
print(f"VHDL FFN Translation: {vhdl_translation_text}")

print("\nToken-level comparison:")
print(f"Reference tokens: {ref_generated_ids}")
print(f"Test FFN tokens: {test_generated_ids}")
print(f"VHDL FFN tokens: {generated_ids}")