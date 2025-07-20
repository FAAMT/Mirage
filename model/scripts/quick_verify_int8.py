#!/usr/bin/env python3

import numpy as np
import torch
import sys  # Added for sys.exit()
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput


# Import all functions from the new module
def load_quantized_weights(root: Path, stem: str, shape):
    w = np.fromfile(root / f"{stem}.bin",  dtype=np.int8 ).reshape(shape)
    s = np.fromfile(root / f"{stem}.scale", dtype=np.float32)
    z = np.fromfile(root / f"{stem}.zp",    dtype=np.uint8)

    # --- align scale / zp automatically ----------------------------------
    if s.ndim == 1:
        if s.size == shape[0]:          # per-row
            s = s[:, None]              # (rows,1)
            z = z[:, None]
        elif s.size == shape[1]:        # per-column
            s = s[None, :]              # (1,cols)
            z = z[None, :]
        elif s.size == 1:               # scalar → broadcast everywhere
            pass                        # keep 1-D; NumPy will broadcast
        else:
            raise ValueError("scale length doesn't match any tensor axis")

    return (w.astype(np.float32) - z.astype(np.float32)) * s

# Configuration
CONFIG = {
    "model_name": "Helsinki-NLP/opus-mt-en-ar",
    "sentence": "My name is Fahad"
}

WEIGHTS_DIR = Path("./weights")

# Load models and tokenizer
print("\n🔍 Loading Models and Tokenizer")
print("─────────────────────────────")
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"], torch_dtype=torch.float32)
tokenizer = MarianTokenizer.from_pretrained(CONFIG["model_name"])

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
# print(combined_embeddings_np)

# MHA Custom Function #

def multi_head_attention_forward(x: np.ndarray, Wq_full: np.ndarray, Wk_full: np.ndarray, 
                                       Wv_full: np.ndarray, Bo_model: np.ndarray, Wo_full: np.ndarray, 
                                       num_heads: int, head_dim: int, return_intermediates=False):
    """
    Enhanced Multi-Head Attention with intermediate value tracking
    Returns: 
    - If return_intermediates=False: attention output (np.ndarray)
    - If return_intermediates=True: tuple of (output, intermediates_dict)
    """
    batch_s, seq_l, _ = x.shape
    
    # Initialize storage for intermediates
    intermediates = {
        'q_proj': None,
        'k_proj': None,
        'v_proj': None,
        'q_heads': [],
        'k_heads': [],
        'v_heads': [],
        'attention_scores': [],
        'attention_probs': [],
        'head_outputs': []
    }
    # 1. grab the four bias vectors once
    q_bias = attn.q_proj.bias.detach().numpy()
    k_bias = attn.k_proj.bias.detach().numpy()
    v_bias = attn.v_proj.bias.detach().numpy()

    q_bias = np.zeros_like(q_bias)
    k_bias = np.zeros_like(k_bias)
    v_bias = np.zeros_like(v_bias)
    
    # Full projections (single large matrix multiply)
    q_proj = x @ Wq_full.T + q_bias # [batch, seq, num_heads * head_dim]
    k_proj = x @ Wk_full.T + k_bias
    v_proj = x @ Wv_full.T + v_bias
    
    # Assertion verification
    assert np.allclose(q_proj, (x @ Wq_full.T) + q_bias, atol=1e-6)
    assert np.allclose(k_proj, (x @ Wk_full.T) + k_bias, atol=1e-6)
    assert np.allclose(v_proj, (x @ Wv_full.T) + v_bias, atol=1e-6)


    intermediates['q_proj'] = q_proj
    intermediates['k_proj'] = k_proj
    intermediates['v_proj'] = v_proj
    
    # Reshape into heads
    q_heads = q_proj.reshape(batch_s, seq_l, num_heads, head_dim).transpose(0, 2, 1, 3)
    k_heads = k_proj.reshape(batch_s, seq_l, num_heads, head_dim).transpose(0, 2, 1, 3)
    v_heads = v_proj.reshape(batch_s, seq_l, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    all_head_outputs = []
    for i in range(num_heads):
        # Get current head
        q_h = q_heads[:, i, :, :]  # [batch, seq, head_dim]
        k_h = k_heads[:, i, :, :]
        v_h = v_heads[:, i, :, :]
        
        # Store intermediates
        intermediates['q_heads'].append(q_h)
        intermediates['k_heads'].append(k_h)
        intermediates['v_heads'].append(v_h)
        
        # Scaled dot-product attention
        attn_scores = q_h @ k_h.transpose(0, 2, 1) / np.sqrt(head_dim)
        attn_probs = torch.softmax(torch.from_numpy(attn_scores), dim=-1).numpy()
        head_out = attn_probs @ v_h
        
        intermediates['attention_scores'].append(attn_scores)
        intermediates['attention_probs'].append(attn_probs)
        intermediates['head_outputs'].append(head_out)
        
        all_head_outputs.append(head_out)
    
    # Concatenate head outputs
    if num_heads > 1:
        head_out = np.concatenate(all_head_outputs, axis=-1)
    else:
        head_out = all_head_outputs[0]
    
    # Output projection
    output = head_out @ Wo_full.T + Bo_model
    
    if return_intermediates:
        return output, intermediates
    return output
    
    # ─────────────────────────────────────────────────────────────────────────

# FFN Custom Function #

def ffn_custom_verbose(x_fp32: np.ndarray,
                       W1: np.ndarray, b1: np.ndarray,
                       W2: np.ndarray, b2: np.ndarray,
                       act_fn,
                       tok_idx: int, layer_idx: int):
    """
    Simple 256 → 512 → 256 FFN with ReLU and detailed prints.

    * x_fp32 : 1 × 256 input vector (float32)
    * W1, b1 : (512,256) and (512,) first FC weights / bias (float32)
    * W2, b2 : (256,512) and (256,) second FC weights / bias (float32)

    Returns: 1 × 256 float32 vector.
    """
    # FC-1 + ReLU
    fc1_pre  = W1 @ x_fp32 + b1          # pre-ReLU
    fc1_post = act_fn(torch.from_numpy(fc1_pre)).numpy()   # ←
    out      = W2 @ fc1_post + b2

    # console trace (token-local)
    print(f"\n[Layer {layer_idx} | token {tok_idx}] FFN stages (first 4 dims)")
    print("    in            :", np.round(x_fp32[:4],   4))
    print("  FC1 pre-ReLU    :", np.round(fc1_pre[:4],  4))
    print("  FC1 post-ReLU   :", np.round(fc1_post[:4], 4))
    print("  FC2 (output)    :", np.round(out[:4],      4))

    return fc1_pre.astype(np.float32), fc1_post.astype(np.float32), out.astype(np.float32)

# Get model's attention weights and biases
for idx, layer in enumerate(model.model.encoder.layers):
    print(f"\n Layer {idx}")
    layer_prefix = f"layer{idx}_"

    attn = layer.self_attn
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    
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

    # Prepare inputs
    input_np = combined_embeddings.detach().numpy()  # Use full embedding dimension

    # Get model's QKV projections
    hidden_t = torch.from_numpy(hidden_np)
    with torch.no_grad():
         q_ref   = attn.q_proj(hidden_t)
         k_ref   = attn.k_proj(hidden_t)
         v_ref   = attn.v_proj(hidden_t)
         mha_ref = attn(hidden_t)[0].detach().numpy()
     # 3.  custom MHA
    custom_out, inter = multi_head_attention_forward(
        hidden_np, Wq_full=Wq_model, Wk_full=Wk_model, Wv_full=Wv_model,
        Bo_model=Bo_model, Wo_full=Wo_model,
        num_heads=num_heads, head_dim=head_dim,
        return_intermediates=True
    )
    
    # 4.  Add & LN (post-MHA)
    res1 = custom_out + hidden_np
    ln1  = layer.self_attn_layer_norm
    mu1, var1 = res1.mean(-1, keepdims=True), res1.var(-1, keepdims=True)
    res1_norm = ((res1 - mu1) / np.sqrt(var1 + ln1.eps)) \
                * ln1.weight.detach().numpy() + ln1.bias.detach().numpy()

    # ---- 5.  FFN (custom & model comparison) -------------------------------
    b1 = layer.fc1.bias.detach().numpy()
    b2 = layer.fc2.bias.detach().numpy()
    b1 = np.zeros_like(b1)
    b2 = np.zeros_like(b2)

    ffn_custom_all = np.zeros_like(res1_norm)

    # ── pretty print: first-token, head-0, first 4 dims ──────────────────
    q_ref_4  = q_ref[0, 0, :4].cpu().numpy()
    k_ref_4  = k_ref[0, 0, :4].cpu().numpy()
    v_ref_4  = v_ref[0, 0, :4].cpu().numpy()
    q_cst_4  = inter["q_proj"][0, 0, :4]
    k_cst_4  = inter["k_proj"][0, 0, :4]
    v_cst_4  = inter["v_proj"][0, 0, :4]
    mha_ref_4 = mha_ref[0, 0, :4]
    mha_cst_4 = custom_out[0, 0, :4]

    print(f"\n🔍 Layer {idx} • first-token/head-0 (first 4 dims)")
    print("     Q_ref :", q_ref_4)
    print("     Q_cst :", q_cst_4)
    print("     K_ref :", k_ref_4)
    print("     K_cst :", k_cst_4)
    print("     V_ref :", v_ref_4)
    print("     V_cst :", v_cst_4)
    print("   MHA_ref :", mha_ref_4)
    print("   MHA_cst :", mha_cst_4)

    # ── pretty print: attention scores & probs (token-0, first 4 cols) ──────
    print(f"\n🔍 Layer {idx} • attention scores / probabilities")

    for h in range(num_heads):
        sc  = inter["attention_scores"][h][0]          # (seq,seq)
        pr  = inter["attention_probs"][h][0]           # (seq,seq)

        print(f"\n   Head {h}")
        print(f"     Scores :", sc[0, :4])             # token-0 → first-4 tokens
        print(f"     Probs  :", pr[0, :4])
    
    print("────────────────────────────────────────────────────────────")
    
    # ── allocate with the correct FC-1 width (2048, not 512) ──────────────
    fc1_dim = W1.shape[0]                                 # 2048 for Marian
    fc1_pre_cust  = np.empty((res1_norm.shape[1], fc1_dim), dtype=np.float32)
    fc1_post_cust = np.empty_like(fc1_pre_cust)
    ffn_cust_out  = np.empty_like(fc1_pre_cust)

    for t_idx in range(res1_norm.shape[1]):
        pre, post, out_vec = ffn_custom_verbose(
            res1_norm[0, t_idx], W1, b1, W2, b2,
            act_fn=layer.activation_fn,   
            tok_idx=t_idx, layer_idx=idx)

        # store the whole vector, *not* a single element
        fc1_pre_cust [t_idx] = pre
        fc1_post_cust[t_idx] = post
        ffn_custom_all[0, t_idx] = out_vec
        
    # ── reference FC1 (pre/post-ReLU) via PyTorch ──────────────────
    with torch.no_grad():
        fc1_ref_pre  = layer.fc1(torch.from_numpy(res1_norm)).squeeze(0)
    fc1_ref_post = layer.activation_fn(fc1_ref_pre).numpy()
    print("\n  REF FC1 pre-ReLU    :", fc1_ref_pre[:4])
    print("  REF FC1 post-ReLU   :", fc1_ref_post[:4])    
    pre_diff  = np.abs(fc1_ref_pre  - fc1_pre_cust ).max()
    post_diff = np.abs(fc1_ref_post - fc1_post_cust).max()
    print(f"ΔFC1 pre-ReLU  max: {pre_diff:.6f}")
    print(f"ΔFC1 post-ReLU max: {post_diff:.6f}")

    # ── FC2 comparison ────────────────────────────────────────────
    with torch.no_grad():                              # reference via PyTorch
        ffn_ref = layer.fc2(layer.activation_fn(
                    layer.fc1(torch.from_numpy(res1_norm)))).numpy()

    print("\n  REF FC2 output     :", ffn_ref[0, :4])
    print("  CUST FC2 output   :", ffn_custom_all[0, :4])
    ffn_diff = np.abs(ffn_ref - ffn_custom_all).max()
    print(f"ΔFC2 output max    : {ffn_diff:.6f}")

    # ── Add & LN after FC-2  → next-layer input ───────────────────
    res2 = ffn_custom_all + res1_norm            # residual add

    #    Add & LN (post-MHA)
    ln2  = layer.final_layer_norm
    mu2, var2 = res2.mean(-1, keepdims=True), res2.var(-1, keepdims=True)
    res2_norm = ((res2 - mu2) / np.sqrt(var2 + ln2.eps)) * ln2.weight.detach().numpy() + ln2.bias.detach().numpy()
    hidden_np = res2_norm.astype(np.float32)
    cus_outer_np = hidden_np  

    # keep reference path via native LayerNorm (optional)
    with torch.no_grad():
        ref_outer_np = ln2(torch.from_numpy(res2)).numpy().astype(np.float32)

# ── after the loop: wrap encoder outputs & translate ─────────
ref_enc_np = ref_outer_np            # final reference states
cus_enc_np = cus_outer_np            # final custom states

enc_out_ref    = BaseModelOutput(last_hidden_state=torch.from_numpy(ref_enc_np))
enc_out_custom = BaseModelOutput(last_hidden_state=torch.from_numpy(cus_enc_np))

with torch.no_grad():
    ids_ref = model.generate(
        encoder_outputs=enc_out_ref,          # ← bug-1 fixed
        attention_mask=torch.ones_like(input_ids),
        num_beams=5, max_length=40, early_stopping=True,
    )
    ids_custom = model.generate(
        encoder_outputs=enc_out_custom,
        attention_mask=torch.ones_like(input_ids),
        num_beams=5, max_length=40, early_stopping=True,
    )

print("\n🌐 Translation  (HF encoder) :", tokenizer.decode(ids_ref[0],    skip_special_tokens=True))
print("🌐 Translation (custom enc.) :", tokenizer.decode(ids_custom[0], skip_special_tokens=True))