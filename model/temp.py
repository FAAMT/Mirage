#!/usr/bin/env python3

import numpy as np
import torch
import sys  # Added for sys.exit()
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput


# Configuration
CONFIG = {
    "model_name": "Helsinki-NLP/opus-mt-en-ar",
    "sentence": "My name is Fahad"
}

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
                       tok_idx: int, layer_idx: int,
                       show: int = 4):
    """
    Simple 256 → 512 → 256 FFN with ReLU and detailed prints.

    * x_fp32 : 1 × 256 input vector (float32)
    * W1, b1 : (512,256) and (512,) first FC weights / bias (float32)
    * W2, b2 : (256,512) and (256,) second FC weights / bias (float32)

    Returns: 1 × 256 float32 vector.
    """
    # FC-1 + ReLU
    hidden = W1 @ x_fp32 + b1
    hidden = np.maximum(hidden, 0.0)

    # FC-2 + ReLU
    out = W2 @ hidden + b2
    out = np.maximum(out, 0.0)

    print(f"\n[Layer {layer_idx} | token {tok_idx}] "
          f"FFN in → out first {show} dims:")
    print("    in :", np.round(x_fp32[:show], 4))
    print("  FC1+ :", np.round(hidden[:show], 4))
    print("   out :", np.round(out[:show], 4))

    return out.astype(np.float32)
# ─────────────────────────────────────────────────────────────────────────

# Get model's attention weights and biases
for idx, layer in enumerate(model.model.encoder.layers):
    print(f"\n Layer {idx}")
    attn = layer.self_attn
    num_heads = attn.num_heads
    head_dim = attn.head_dim

    # Extract weights from the model
    Wq_model = attn.q_proj.weight.detach().numpy()
    Wk_model = attn.k_proj.weight.detach().numpy()
    Wv_model = attn.v_proj.weight.detach().numpy()
    Bo_model = attn.out_proj.bias.detach().numpy()
    Wo_model = attn.out_proj.weight.detach().numpy()

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
    W1 = layer.fc1.weight.detach().numpy()
    b1 = layer.fc1.bias.detach().numpy()
    W2 = layer.fc2.weight.detach().numpy()
    b2 = layer.fc2.bias.detach().numpy()

    ffn_custom = ffn_custom_verbose(res1_norm[0, 0], W1, b1, W2, b2,
                                    tok_idx=0, layer_idx=idx)

    with torch.no_grad():                              # reference via PyTorch
        ffn_ref = layer.fc2(layer.activation_fn(
                    layer.fc1(torch.from_numpy(res1_norm)))).numpy()

    print("   ΔFFN max :", np.abs(ffn_ref - ffn_custom).max())

    # 6.  Add & LN (post-FFN)  → next-layer input
    res2 = ffn_custom + res1_norm
    ln2  = layer.final_layer_norm
    mu2, var2 = res2.mean(-1, keepdims=True), res2.var(-1, keepdims=True)
    hidden_np = ((res2 - mu2) / np.sqrt(var2 + ln2.eps)) \
                * ln2.weight.detach().numpy() + ln2.bias.detach().numpy()
    hidden_np = hidden_np.astype(np.float32)


    # ── pretty print: first-token, head-0, first 4 dims ──────────────────
    q_ref_4  = q_ref[0, 0, :4].cpu().numpy()
    k_ref_4  = k_ref[0, 0, :4].cpu().numpy()
    v_ref_4  = v_ref[0, 0, :4].cpu().numpy()
    q_cst_4  = inter["q_proj"][0, 0, :4]
    k_cst_4  = inter["k_proj"][0, 0, :4]
    v_cst_4  = inter["v_proj"][0, 0, :4]
    mha_ref_4 = mha_ref[0, 0, :4]
    mha_cst_4 = custom_out[0, 0, :4]

    # ── pretty print: FFN vectors (same token, 4 dims) ──────────────────
    ffn_ref_4  = torch.from_numpy(ffn_ref).numpy()[0, :4]        # model FFN output
    ffn_cst_4  = ffn_custom[:4]                # custom FFN output
    post_ln_4  = hidden_np[0, 0, :4]           # after FFN Add&LN

    print(f"\n🔍 Layer {idx} • first-token/head-0 (first 4 dims)")
    print("     Q_ref :", q_ref_4)
    print("     Q_cst :", q_cst_4)
    print("     K_ref :", k_ref_4)
    print("     K_cst :", k_cst_4)
    print("     V_ref :", v_ref_4)
    print("     V_cst :", v_cst_4)
    print("   MHA_ref :", mha_ref_4)
    print("   MHA_cst :", mha_cst_4)
    print("   FFN_ref :", ffn_ref_4)
    print("   FFN_cst :", ffn_cst_4)
    print("   post-LN :", post_ln_4)
    print("────────────────────────────────────────────────────────────")

# ── after the loop: final hidden state & translations ───────────────────
custom_enc_np = hidden_np
print("\n🔍 Final encoder first-token dims 0-3 :", custom_enc_np[0, 0, :4])

enc_out_custom = BaseModelOutput(
    last_hidden_state=torch.from_numpy(custom_enc_np)
)

with torch.no_grad():
    ids_custom = model.generate(
        encoder_outputs=enc_out_custom,
        attention_mask=torch.ones_like(input_ids),
        num_beams=5, max_length=40, early_stopping=True,
    )
    ids_ref = model.generate(                                  # HF reference
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        num_beams=5, max_length=40, early_stopping=True,
    )

print("\n🌐 Translation  (HF encoder) :", tokenizer.decode(ids_ref[0],    skip_special_tokens=True))
print("🌐 Translation (custom enc.) :", tokenizer.decode(ids_custom[0], skip_special_tokens=True))
