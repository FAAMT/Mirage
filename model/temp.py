#!/usr/bin/env python3

import numpy as np
import torch
import sys  # Added for sys.exit()
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput

def multi_head_attention_forward(x: np.ndarray, Wq_full: np.ndarray, Wk_full: np.ndarray, 
                                       Wv_full: np.ndarray, Wo_bias: np.ndarray, Wo_full: np.ndarray, 
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
    output = head_out @ Wo_full.T + Wo_bias
    
    if return_intermediates:
        return output, intermediates
    return output

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

# Get model's attention weights and biases
attn = encoder_layer.self_attn
num_heads = attn.num_heads
head_dim = attn.head_dim

# Extract weights from the model
Wq_model = attn.q_proj.weight.detach().numpy()
Wk_model = attn.k_proj.weight.detach().numpy()
Wv_model = attn.v_proj.weight.detach().numpy()
Wo_bias = attn.out_proj.bias.detach().numpy()
Wo_model = attn.out_proj.weight.detach().numpy()

# Reshape into separate heads for our custom implementation
Wq_custom = [Wq_model[i*head_dim:(i+1)*head_dim, :] for i in range(num_heads)]
Wk_custom = [Wk_model[i*head_dim:(i+1)*head_dim, :] for i in range(num_heads)]
Wv_custom = [Wv_model[i*head_dim:(i+1)*head_dim, :] for i in range(num_heads)]

# Prepare inputs
input_np = combined_embeddings.detach().numpy()  # Use full embedding dimension

print("\n🔍 Multi-Head Attention Comparison")
print("─────────────────────────────────")

# Run model's attention
print("\nModel's Attention Implementation:")
print("--------------------------------")
model_output = captured_values['attention_output'].detach().numpy()

# Get model's QKV projections
with torch.no_grad():
    q_model = attn.q_proj(captured_values['encoder_input'])
    k_model = attn.k_proj(captured_values['encoder_input'])
    v_model = attn.v_proj(captured_values['encoder_input'])
    
    # Reshape into heads
    q_model_heads = q_model.view(q_model.size(0), q_model.size(1), num_heads, head_dim).transpose(1, 2)
    k_model_heads = k_model.view(k_model.size(0), k_model.size(1), num_heads, head_dim).transpose(1, 2)
    v_model_heads = v_model.view(v_model.size(0), v_model.size(1), num_heads, head_dim).transpose(1, 2)

# Run our custom attention with intermediates
custom_output, intermediates = multi_head_attention_forward(
    input_np, 
    Wq_full=Wq_model,
    Wk_full=Wk_model,
    Wv_full=Wv_model,
    Wo_bias=Wo_bias,
    Wo_full=Wo_model,
    num_heads=num_heads,
    head_dim=head_dim,
    return_intermediates=True
)

# Step 1: Add MHA + token + pos_enc
residual = custom_output + combined_embeddings_np

# Step 2: Get LayerNorm weights from the encoder block
ln = model.model.encoder.layers[0].self_attn_layer_norm
gamma = ln.weight.detach().cpu().numpy()
beta = ln.bias.detach().cpu().numpy()

residual  = residual.astype(np.float32)
gamma = gamma.astype(np.float32)
beta  = beta.astype(np.float32)

# Step 3: Apply LayerNorm manually (axis=-1)
mu  = residual.mean(axis=-1, keepdims=True)
var = residual.var(axis=-1, keepdims=True)
eps = ln.eps

normalized = (residual - mu) / np.sqrt(var + eps)
resid_norm = normalized * gamma + beta

torch_out = ln(torch.from_numpy(residual)).detach().numpy()

# Compare QKV projections for first head
print("\n🔍 QKV Projections Comparison (First Head)")
print("──────────────────────────────────────────")

# Model's Q projections (first head)
print("\nModel Q (first head, first token):")
print(q_model_heads[0, 0, 0, :16].detach().numpy())

# Custom Q projections (first head)
print("\nCustom Q (first head, first token):")
print(intermediates['q_proj'][0, 0, :16])

print("\nQ-projection bias terms (first 16):")
print(Wo_bias[:16])

# Model's K projections (first head)
print("\nModel K (first head, first token):")
print(k_model_heads[0, 0, 0, :16].detach().numpy())

# Custom K projections (first head)
print("\nCustom K (first head, first token):")
print(intermediates['k_proj'][0, 0, :16])

# Model's V projections (first head)
print("\nModel V (first head, first token):")
print(v_model_heads[0, 0, 0, :16].detach().numpy())

# Custom V projections (first head)
print("\nCustom V (first head, first token):")
print(intermediates['v_proj'][0, 0, :16])

# Compare attention scores for first head
print("\n🔍 Attention Scores Comparison (First Head)")
print("──────────────────────────────────────────")

# Compare outputs
print("\n🔍 Output Comparison")
print("───────────────────")

print("\n Encoder MHA FFN -> Add & Norm Output (first 16 dimensions):")
print("FP32:", " ".join(f"{x:.4f}" for x in attn_hook[0, :16]))
print("Q1.7:", " ".join(f"{int(round(x*128)):4d}" for x in attn_hook[0, :16]))

# print("\nCustom Output (first 16 dimensions):")
# print("FP32:", " ".join(f"{x:.4f}" for x in custom_output[0, 0, :16]))
# print("Q1.7:", " ".join(f"{int(round(x*128)):4d}" for x in custom_output[0, 0, :16]))

print("\nCustom MHA Output -> Add & Norm (first 16 dimensions):")
print("FP32:", " ".join(f"{x:.4f}" for x in resid_norm[0, 0, :16]))
print("Q1.7:", " ".join(f"{int(round(x*128)):4d}" for x in resid_norm[0, 0, :16]))

print("\nDifference (absolute):")
diff = np.abs(attn_hook[0, :16] - resid_norm[0, 0, :16])
print("FP32:", " ".join(f"{x:.4f}" for x in diff))
print("Max difference:", np.max(diff))


print("\n Decoding the following tensors...")

print("\nEncoder FFN -> Add & Norm Output (first 16 dimensions):")
print("FP32:", " ".join(f"{x:.4f}" for x in model_output[0, 0, :16]))
print("Q1.7:", " ".join(f"{int(round(x*128)):4d}" for x in model_output[0, 0, :16]))

# Step 1: Add MHA + token + pos_enc
residual = model_output + combined_embeddings_np

# Step 2: Get LayerNorm weights from the encoder block
ln = model.model.encoder.layers[0].final_layer_norm
gamma = ln.weight.detach().cpu().numpy()
beta = ln.bias.detach().cpu().numpy()

residual  = residual.astype(np.float32)
gamma = gamma.astype(np.float32)
beta  = beta.astype(np.float32)

# Step 3: Apply LayerNorm manually (axis=-1)
mu  = residual.mean(axis=-1, keepdims=True)
var = residual.var(axis=-1, keepdims=True)
eps = ln.eps

normalized = (residual - mu) / np.sqrt(var + eps)
resid_norm = normalized * gamma + beta

# 1) Choose which tensor you want the decoder to see:
# --------------------------------------------------------------------------
# a) Reference: model's own encoder output (all layers finished)
enc_states = encoder_output.last_hidden_state                   # torch Tensor

print(resid_norm)
print(enc_states)

# b) Later: your custom pipeline once you have FFN + LN done
# enc_states = torch.from_numpy(custom_encoder_np)              # shape (B,S,D)

# 2) Wrap it in BaseModelOutput so .generate() skips the encoder
enc_out = BaseModelOutput(last_hidden_state=enc_states)

# 3) Run generation
with torch.no_grad():
    translated_ids = model.generate(
        encoder_outputs=enc_out,                                # <-- here
        attention_mask=torch.ones_like(input_ids, dtype=torch.long),
        max_length=40,
        num_beams=5,
        early_stopping=True
    )

# 4) Decode tokens to text
translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
print("\n🌐 Translation using supplied encoder states:")
print(translation)