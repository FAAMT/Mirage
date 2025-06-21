#!/usr/bin/env python3
import os, sys, time, math
from pathlib import Path
from decimal import Decimal, getcontext

import numpy as np
import torch
import onnxruntime as ort
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from verify_func import (
    load_quantized_weights,
    to_q17,
    to_q117,
    from_q17,
    compute_taylor_q345,
    calculate_projection,
    multi_head_attention_forward,
    ffn_custom_verbose,
    write_q17_weights
)

# ─────────────────────────────────────────────── CONFIG ───────────────────────────────────────────────
CONFIG = {
    "model_name": "Helsinki-NLP/opus-mt-en-ar",
    "sentence":   "My name is Fahad"
}

# ─────────────────────────────── QUANTISED ENCODER / TOKENIZER LOAD ──────────────────────────────────
print("\n Loading FP32 Encoder & Tokenizer")
print("────────────────────────────────────────────")
DIRECTION   = "en-ar"
WEIGHTS_DIR = Path("./weights")

# ─────────────────────────────────────── FP-32 REFERENCE MODEL ────────────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"], torch_dtype=torch.float32)
tokenizer   = MarianTokenizer.from_pretrained(CONFIG["model_name"])

# ─────────────────────────────────────────── TOKENISATION ────────────────────────────────────────────
print("\n Tokenization")
print("──────────────")
input_ids        = tokenizer(CONFIG["sentence"], return_tensors="pt")["input_ids"]
original_tokens  = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"Sentence: '{CONFIG['sentence']}'")
print(f"Token IDs: {input_ids[0].tolist()}")
print(f"Tokens: {original_tokens}")

# ────────────────────────────── EMBEDDINGS & POSITIONAL ENCODING ─────────────────────────────────────
print("\n Embeddings and Positional Encoding")
print("────────────────────────────────────")
token_embeddings = tokenizer(
    CONFIG["sentence"], return_tensors="np", padding=False, truncation=True
)

# Tokenization
print("\n🔍 Tokenization")
print("──────────────")
input_ids = tokenizer(CONFIG["sentence"], return_tensors="pt")["input_ids"]
original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"Sentence: '{CONFIG['sentence']}'")
print(f"Token IDs: {input_ids[0].tolist()}")
print(f"Tokens: {original_tokens}")

# Get embeddings and positional encoding
print("\n🔍 Embeddings and Positional Encoding")
print("────────────────────────────────────")
token_embeddings = model.model.shared(input_ids)

# Hook to get positional encodings and attention outputs
captured_values = {}
def capture_pe(module, input, output):
    captured_values['encoder_input'] = input[0]
    captured_values['attention_output'] = output[0]

def capture_att(module, input, output):
    captured_values['attention_hook'] = output[0]

def capture_res(module, input):
    captured_values['residual_pre_ln'] = input[0]

# We'll use the first layer for comparison
encoder_layer = model.model.encoder.layers[0]
hook_pe = encoder_layer.register_forward_hook(capture_pe)
hook_att = encoder_layer.self_attn_layer_norm.register_forward_hook(capture_att)
hook_res = encoder_layer.self_attn_layer_norm.register_forward_pre_hook(capture_res)

with torch.no_grad():
    encoder_output = model.model.encoder(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.long)
    )
hook_pe.remove()

model_residual = captured_values["residual_pre_ln"].cpu().numpy().astype(np.float32)
attn_hook = captured_values["attention_hook"].cpu().numpy().astype(np.float32)
pos_encoding = captured_values['encoder_input'] - token_embeddings
combined_embeddings = token_embeddings + pos_encoding
combined_embeddings_np = combined_embeddings.detach().cpu().numpy().astype(np.float32)
hidden_np = combined_embeddings_np

combined_embeddings = hidden_np                       # alias for clarity
# Convert to Q1.7 format (int8 with 7 fractional bits)
combined_embeddings_q17 = np.clip(combined_embeddings * 128, -128, 127).astype(np.int8)

print("\n🔍 Combined Embeddings (Token + Positional)")
print("───────────────────────────────────────────")
print("Format: [token_index] token_string")
print("fp32: first 4 ... last 4")
print("Q1.7: first 4 ... last 4 (int8)")
print("─" * 50)

for i, token in enumerate(original_tokens):
    fp32 = combined_embeddings[0, i]
    q17 = combined_embeddings_q17[0, i]
    
    print(f"[{i}] {token}")
    print(f"  fp32: {fp32[:4].tolist()} ... {fp32[-4:].tolist()}")
    print(f"  Q1.7: {q17[:4].tolist()} ... {q17[-4:].tolist()}")
    print("─" * 50)

# ────────────────────────────────────── ENCODING-PHASE VERIFICATION ──────────────────────────────────
print("\n Encoding Phase Calculations")
print("────────────────────────────────────")

for idx, layer in enumerate(model.model.encoder.layers):
    print(f"\n Layer {idx}")
    attn      = layer.self_attn
    num_heads = attn.num_heads
    head_dim  = attn.head_dim
    embed_dim = attn.embed_dim

    # ─────────── Helper to build layer-specific stems ───────────
    layer_prefix = f"layer{idx}_"

    # Directory of extracted weights
    NUM_HEADS = 8
    HEAD_DIM = 64
    D_MODEL = 512
    
    # Load QKV and projections - corrected shapes

    # Bias Extraction
    Bo_model = attn.out_proj.bias.detach().numpy()
    Bo_model = np.zeros_like(Bo_model) # Zero-ed Bias 

    Wq_heads = [load_quantized_weights(WEIGHTS_DIR, f"{layer_prefix}query_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]    
    Wk_heads = [load_quantized_weights(WEIGHTS_DIR, f"{layer_prefix}key_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
    Wv_heads = [load_quantized_weights(WEIGHTS_DIR, f"{layer_prefix}value_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]

    # Concatenate heads horizontally WITHOUT transposing first
    Wq_model = np.concatenate(Wq_heads, axis=0)  # [512, 512] (512x64 × 8 heads)
    Wk_model = np.concatenate(Wk_heads, axis=0)  # [512, 512]
    Wv_model = np.concatenate(Wv_heads, axis=0)  # [512, 512]

    # Transpose the final matrices to get [512,512] shape expected by MHA function
    Wq_model = Wq_model.T  # [512, 512]
    Wk_model = Wk_model.T  # [512, 512]
    Wv_model = Wv_model.T  # [512, 512]

    # Load output projection
    Wo_model = load_quantized_weights(WEIGHTS_DIR, f"{layer_prefix}attn_out_proj_full", (512, 512))
    Wo_model = Wo_model.T
    
    # Load FFN weights with correct shapes from log
    W1 = load_quantized_weights(WEIGHTS_DIR, f"{layer_prefix}ffn_fc2_full", (512, 2048)).T  # FC1: (512, 2048)
    W2 = load_quantized_weights(WEIGHTS_DIR, f"{layer_prefix}ffn_fc1_full", (2048, 512)).T  # FC2: (2048, 512)

    b1 = layer.fc1.bias.detach().numpy()
    b2 = layer.fc2.bias.detach().numpy()
    b1 = np.zeros_like(b1)
    b2 = np.zeros_like(b2)

    # ──────────────────────────────── HEAD-WISE PRINTS ────────────────────────────────
    print("\n Muti-Head Attention Calculations ")
    print("────────────────────────────────────")
    print(f"\n # Heads: {num_heads}     # Dim: {head_dim}")

    # —— Q Weights ——
    print("\nQ Weights (first 4 elements per head):")
    for h in range(num_heads):
        rs, re = h * head_dim, (h + 1) * head_dim
        print(f"\n== Q-Head {h} ===")
        print(f"Row 0 Float: {Wq_model[0, rs:rs+4]}...")
        print(f"Row 0 Q1.7 : {to_q17(Wq_model[0, rs:rs+4])}...")

    # —— K Weights ——
    print("\nK Weights (first 4 elements per head):")
    for h in range(num_heads):
        rs, re = h * head_dim, (h + 1) * head_dim
        print(f"\n=== K-Head {h} ===")
        print(f"Row 0 Float: {Wk_model[0, rs:rs+4]}...")
        print(f"Row 0 Q1.7 : {to_q17(Wk_model[0, rs:rs+4])}...")

    # —— V Weights ——
    print("\nV Weights (first 4 elements per head):")
    for h in range(num_heads):
        rs, re = h * head_dim, (h + 1) * head_dim
        print(f"\n=== V-Head {h} ===")
        print(f"Row 0 Float: {Wv_model[0, rs:rs+4]}...")
        print(f"Row 0 Q1.7 : {to_q17(Wv_model[0, rs:rs+4])}...")

    # ─────────────────────────── MHA FORWARD PASS ────────────────────────────
    mha_out, inter = multi_head_attention_forward(
        hidden_np,
        attn,
        Wq_full=Wq_model,
        Wk_full=Wk_model,
        Wv_full=Wv_model,
        Bo_model=Bo_model,
        Wo_full=Wo_model,
        num_heads=num_heads,
        head_dim=head_dim,
        return_intermediates=True,
    )

    print("\n MHA Projections")
    print(f"\n🔍 Layer {idx} • first-token — first 4 dims per head")
    proj_map = {"Q": "q_proj", "K": "k_proj", "V": "v_proj"}

    for tag, key in proj_map.items():
        print(f"\n{tag} Projection:")
        arr = inter[key]  # (B,T,E) or (B,T,H,D)
        if arr.ndim == 4:
            for h in range(num_heads):
                print(f"  == {tag}-Head {h} ===")
                print(f"    Float: {arr[0,0,h,:4]}")
                print(f"    Q1.7:  {to_q17(arr[0,0,h,:4])}")
        else:
            for h in range(num_heads):
                s, e = h * head_dim, h * head_dim + 4
                print(f"  == {tag}-Head {h} ===")
                print(f"    Float: {arr[0,0,s:e]}")
                print(f"    Q1.7:  {to_q17(arr[0,0,s:e])}")

    # —— Scores & Probabilities ——
    print(f"\n🔍 Layer {idx} • attention scores / probabilities")
    for h in range(num_heads):
        sc = inter["attention_scores"][h][0]
        pr = inter["attention_probs"][h][0]
        print(f"\n  == Head {h} ===")
        print(f"     Scores: {sc[0,:4]}")
        print(f"     Q1.17 : {to_q117(sc[0,:4])}")
        print(f"     Probs : {pr[0,:4]}")
        print(f"     Q1.17 : {to_q117(pr[0,:4])}")
    print("────────────────────────────────────────────────────────────")

    # —— MHA Output ——
    print("\nMHA Final Output:")
    for h in range(num_heads):
        s, e = h * head_dim, h * head_dim + 4
        print(f" == Head {h} ===")
        print(f"    Float: {mha_out[0,0,s:e]}")
        print(f"    Q1.7: {to_q17(mha_out[0,0,s:e])}")

    # ─────────────────────────────── RESIDUAL + LN (POST-MHA) ───────────────────────────────
    res1      = mha_out + hidden_np
    ln1       = layer.self_attn_layer_norm
    mu1, var1 = res1.mean(-1, keepdims=True), res1.var(-1, keepdims=True)
    res1_norm = ((res1 - mu1) / np.sqrt(var1 + ln1.eps)) * ln1.weight.detach().numpy() + ln1.bias.detach().numpy()

    print("\nLN -> [MHA Final Output + Res]:")
    print(f"\n  Tensor Shape: {torch.tensor(res1_norm).shape}")
    for h in range(num_heads):
        s, e = h * head_dim, h * head_dim + 4
        print(f"  Head {h}:")
        print(f"    Float: {res1_norm[0,0,s:e]}")
        print(f"    Q1.7:  {to_q17(res1_norm[0,0,s:e])}")

    # ───────────────────────────────────── FFN ─────────────────────────────────────
    print(f"\n🔍 Layer {idx} FFN Computations")
    print("────────────────────────────────────")
    fc1_dim   = W1.shape[0]
    fc1_pre   = np.empty((res1_norm.shape[1], fc1_dim), dtype=np.float32)
    fc1_post  = np.empty_like(fc1_pre)
    ffn_out   = np.empty_like(res1_norm)

    for t in range(res1_norm.shape[1]):
        pre, post, out_vec = ffn_custom_verbose(
            res1_norm[0, t], W1, b1, W2, b2,
            act_fn=layer.activation_fn,
            tok_idx=t, layer_idx=idx
        )
        fc1_pre[t], fc1_post[t], ffn_out[0, t] = pre, post, out_vec

        print(f"\n== Token {t} ==")
        print("  Input:")
        print(f"    Float: {res1_norm[0,t,:4].tolist()}")
        print(f"    Q1.7:  {to_q17(res1_norm[0,t,:4])}")

        print("\n  FC1 pre-ReLU:")
        print(f"    Float: {pre[:4].tolist()}")
        print(f"    Q1.7:  {to_q17(pre[:4])}")

        print("\n  FC1 post-ReLU:")
        print(f"    Float: {post[:4].tolist()}")
        print(f"    Q1.7:  {to_q17(post[:4])}")

        print("\n  FC2 Output:")
        print(f"    Float: {out_vec[:4].tolist()}")
        print(f"    Q1.7:  {to_q17(out_vec[:4])}")

    print("\nFinal FFN Output (first token, first 4 dims):")
    print(f"    Float: {ffn_out[0,0,:4].tolist()}")
    print(f"    Q1.7:  {to_q17(ffn_out[0,0,:4])}")

    # ─────────────────────────── RESIDUAL + LN (POST-FFN) ───────────────────────────
    res2      = ffn_out + res1_norm
    ln2       = layer.final_layer_norm
    mu2, var2 = res2.mean(-1, keepdims=True), res2.var(-1, keepdims=True)
    res2_norm = ((res2 - mu2) / np.sqrt(var2 + ln2.eps)) * ln2.weight.detach().numpy() + ln2.bias.detach().numpy()

    hidden_np = res2_norm.astype(np.float32)  # feed-forward to next layer

    print("\nLN -> [FFN Final Output + Res]:")
    print(f"\n  Tensor Shape: {torch.tensor(res2_norm).shape}")
    for h in range(num_heads):
        s, e = h * head_dim, h * head_dim + 4
        print(f"  Head {h}:")
        print(f"    Float: {res2_norm[0,0,s:e]}")
        print(f"    Q1.7:  {to_q17(res2_norm[0,0,s:e])}")

# ─────────────────────────────── TRANSLATION USING CUSTOM ENC ───────────────────────────────
enc_out_custom = BaseModelOutput(last_hidden_state=torch.from_numpy(hidden_np))

with torch.no_grad():
    ids_custom = model.generate(
        encoder_outputs=enc_out_custom,
        attention_mask=torch.ones_like(input_ids),
        num_beams=5, max_length=40, early_stopping=True,
    )

translation = tokenizer.decode(ids_custom[0], skip_special_tokens=True)
print("🌐 Translation (custom enc., int8) :", translation)
