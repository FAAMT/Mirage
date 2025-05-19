#!/usr/bin/env python3
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from test_extracted_weights import (
    load_sliced_weights,
    D_MODEL, HEAD_DIM, NUM_HEADS,
    multi_head_attention_forward as multi_head_attention
)

# Directory of extracted weights
WEIGHTS_DIR = Path("./weights")

# Load QKV and projections
Wq = [load_sliced_weights(WEIGHTS_DIR, f"query_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
Wk = [load_sliced_weights(WEIGHTS_DIR, f"key_head{i}",   (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
Wv = [load_sliced_weights(WEIGHTS_DIR, f"value_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
Wo = load_sliced_weights(WEIGHTS_DIR, "attn_out_proj", (D_MODEL, D_MODEL))

# Load FFN weights: ffn_fc1 (512,256) and ffn_fc2 (256,512)
W1 = load_sliced_weights(WEIGHTS_DIR, "ffn_fc1", (2*D_MODEL, D_MODEL))
W2 = load_sliced_weights(WEIGHTS_DIR, "ffn_fc2", (D_MODEL, 2*D_MODEL))

# Load tokenizer and embeddings
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
model = AutoModel.from_pretrained(
    "Helsinki-NLP/opus-mt-ar-en", torch_dtype=torch.float32
)
embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()

# Prepare input
sentence = "My name is Fahad"
print(f"Input sentence: '{sentence}'")
ids = tokenizer(sentence, return_tensors="np")["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(ids[0])
print(f"Tokenized to: {tokens}")

x = embeddings[ids][..., :D_MODEL]  # shape (1, seq, 256)
print(f"Truncated x shape: {x.shape}")

def to_q17(x):
    """Convert float32 to Q1.7 fixed-point representation."""
    return np.clip(np.round(x * 128), -128, 127).astype(np.int8)

def from_q17(x):
    """Convert Q1.7 fixed-point representation to float32."""
    return x.astype(np.float32) / 128.0

# Create the weights directory if it doesn't exist
Path("weights").mkdir(parents=True, exist_ok=True)

# Store token embeddings in a text file (VHDL friendly format)
embedding_file_path = Path("tb_stim/token_embeddings.txt")
with open(embedding_file_path, "w") as f:
    f.write("-- Token Embeddings (Q1.7)\n")
    f.write("-- Format: token_index => (dim_0, dim_1, ..., dim_N);  Q1.7 values\n")
    for i, token in enumerate(tokens):
        x_q17 = to_q17(x[0, i, :])
        embedding_str = ", ".join(f"{val}" for val in x_q17)
        f.write(f"{i} => ({embedding_str});  -- Token: {i}\n")
print(f"\nToken embeddings (VHDL friendly format) stored in: {embedding_file_path}")

# Print first few elements before positional encoding
print("\nFirst 5 embedding values for token 0 before pos-enc:")
for i in range(5):
    val = x[0,0,i]
    print(f"  x[0,0,{i}] = {val:.6f}")
# Q1.7 equivalents
print("Q1.7 fixed-point equivalents before pos-enc:")
for i in range(5):
    val = x[0,0,i]
    q_fixed = int(np.clip(np.round(val * 128), -128, 127))
    print(f"  {val:.6f} *128 → {q_fixed} (Q1.7: {q_fixed}/128 = {q_fixed/128:.6f})")

# Positional encoding
def get_pos_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i//2)) / d_model)
    pe = np.zeros_like(angle)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

pe = get_pos_encoding(x.shape[1], D_MODEL)
x_enc = x + pe[None, :, :]
print("\nFirst 5 embedding values for token 0 after pos-enc:")
for i in range(5):
    val = x_enc[0,0,i]
    print(f"  x_enc[0,0,{i}] = {val:.6f}")
# Q1.7 equivalents after pos-enc
print("Q1.7 fixed-point equivalents after pos-enc:")
for i in range(5):
    val = x_enc[0,0,i]
    q_fixed = int(np.clip(np.round(val * 128), -128, 127))
    print(f"  {val:.6f} *128 → {q_fixed} (Q1.7: {q_fixed}/128 = {q_fixed/128:.6f})")

# Multi-Head Attention with detailed math
print("\n--- Running Single-Head Attention with debug ---")
t0 = time.perf_counter()
mha_out = multi_head_attention(x_enc, Wq, Wk, Wv, Wo)
t1 = time.perf_counter()
print(f"MHA output shape: {mha_out.shape}")

# Show attention calculations for head 0
head_index = 0
print(f"\n--- Attention Details for Head {head_index} ---")
q_mat_head = x_enc[0] @ Wq[head_index].T  # shape (seq_len, HEAD_DIM)
k_mat_head = x_enc[0] @ Wk[head_index].T

# Show Q * K^T for a few elements of each token
print("\nQ * K^T for a few elements (first token's Q with each token's K):")
num_elements_to_show = 3
for i, token_q in enumerate(tokens):
    print(f"  --- Q for token '{token_q}' ---")
    for j, token_k in enumerate(tokens):
        print(f"    -- K for token '{token_k}' --")
        for k in range(num_elements_to_show):
            q_elem = q_mat_head[i, k]
            k_elem = k_mat_head[j, k]
            product = q_elem * k_elem
            print(f"      Q[{i}][{k}]({q_elem:.4f}) * K[{j}][{k}]({k_elem:.4f}) = {product:.4f}")

# Show softmax calculations for a few scores (first token's attention scores)
print("\nSoftmax calculations for a few attention scores (first token):")
scaled_attention_logits = (q_mat_head[0] @ k_mat_head.T) / np.sqrt(HEAD_DIM)
num_scores_to_show = min(len(tokens), 3)
print(f"  Scaled Logits for first token with all Keys: {scaled_attention_logits}")
for i in range(num_scores_to_show):
    logit = scaled_attention_logits[i]
    exp_logit = np.exp(logit)
    sum_exp = np.sum(np.exp(scaled_attention_logits))
    softmax_score = exp_logit / sum_exp
    print(f"  Softmax({logit:.4f}) = exp({logit:.4f}) / sum(exp(all logits)) = {exp_logit:.4f} / {sum_exp:.4f} = {softmax_score:.4f}")

# Custom FFN with detailed math
print("\n--- Running FFN with debug ---")
def feed_forward_custom(x, W_fc1, W_fc2):
    print(f"FFN input shape: {x.shape}")
    # FC1 debug
    print("First 5 MACs for FC1 (hidden dim 0):")
    for i in range(5):
        a = x[0,0,i]
        b = W_fc1[i,0]
        print(f"  {a:.6f} * {b:.6f} = {a*b:.6f}")
    h = np.matmul(x, W_fc1.T)
    h_relu = np.maximum(0, h)
    # Show first 5 ReLU outputs
    print("First 5 ReLU outputs for hidden0:")
    for i in range(5):
        print(f"  max(0,{h[0,0,i]:.6f}) = {h_relu[0,0,i]:.6f}")
    # FC2 debug
    print("First 5 MACs for FC2 (output dim 0):")
    for i in range(5):
        a = h_relu[0,0,i]
        b = W_fc2[i,0]
        print(f"  {a:.6f} * {b:.6f} = {a*b:.6f}")
    out = np.matmul(h_relu, W_fc2.T)
    return out

ffn_out = feed_forward_custom(mha_out, W1, W2)
print(f"FFN output shape: {ffn_out.shape}")

# Print final encoder output for token 0
print("\nFinal encoder output for first token (float32):")
print(ffn_out[0,0])
# Q1.7 equivalents of final output
print("Q1.7 equivalents of final token output:")
for i in range(5):
    val = ffn_out[0,0,i]
    q_fixed = int(np.clip(np.round(val * 128), -128, 127))
    print(f"  {val:.6f} *128 → {q_fixed} (Q1.7: {q_fixed}/128 = {q_fixed/128:.6f})")

print(f"Total elapsed time: {(time.perf_counter()-t0)*1000:.2f} ms")

# Write Q1.7 weights to files
def write_q17_weights(weights, filename):
    """Convert weights to Q1.7 and write to a file."""
    weights_q17 = to_q17(weights)
    with open(filename, "w") as f:
        f.write("-- Q1.7 Fixed-Point Weights\n")
        if weights.ndim == 1:
            f.write(f"-- Format: (dim_0, dim_1, ..., dim_N)\n")
            weight_str = ", ".join(f"{val}" for val in weights_q17)
            f.write(f"({weight_str});\n")
        elif weights.ndim == 2:
            f.write(f"-- Format: ( (dim0_0, dim0_1, ..., dim0_N), (dim1_0, dim1_1, ..., dim1_N), ... );\n")
            for row in weights_q17:
                row_str = ", ".join(f"{val}" for val in row)
                f.write(f"({row_str}),\n")
        elif weights.ndim == 3:
            f.write(f"-- Format: ( ( (dim00_0, dim00_1, ..., dim00_N), ...), ( (dim10_0, dim10_1, ...), ...), ... );\n")
            for matrix in weights_q17:
                matrix_str = "("
                for row in matrix:
                    row_str = ", ".join(f"{val}" for val in row)
                    matrix_str += f"({row_str}),"
                matrix_str = matrix_str.rstrip(",") + "),"
                f.write(f"({matrix_str.rstrip(',')}),\n")

# Convert and save weights to Q1.7 fixed-point representation
write_q17_weights(Wq[0], "tb_stim/q_weights.txt")
write_q17_weights(Wk[0], "tb_stim/k_weights.txt")
write_q17_weights(Wv[0], "tb_stim/v_weights.txt")
write_q17_weights(Wo, "tb_stim/proj_layer_weights.txt")
write_q17_weights(W1, "tb_stim/fc_1_weights.txt")
write_q17_weights(W2, "tb_stim/fc_2_weights.txt")
