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

# Store token embeddings in a text file (VHDL friendly format)
user_tokens = [0, 1, 2, 3]  # Change or expand as needed

embedding_file_path = Path("tb_stim/token_embeddings.txt")
with open(embedding_file_path, "w") as f:
    f.write("-- Token Embeddings (Q1.7)\n")
    f.write("-- Format: ( (dim0_0, dim0_1, ..., dim0_N), (dim1_0, dim1_1, ..., dim1_N), ... );\n")
    # Convert all embeddings to Q1.7 first
    embeddings_q17 = to_q17(x_enc[0])  # Assuming x shape is (1, num_tokens, dim)
    # Write in the same format as 2D weights
    for token_emb in embeddings_q17:
        embedding_str = ", ".join(f"{val}" for val in token_emb)
        f.write(f"({embedding_str}),\n")

# Multi-Head Attention with detailed math
print("\n--- Running Single-Head Attention with debug ---")
t0 = time.perf_counter()
mha_out = multi_head_attention(x_enc, Wq, Wk, Wv, Wo)
t1 = time.perf_counter()
print(f"MHA output shape: {mha_out.shape}")

# Show attention calculations for head 0
head_index = 0
print(f"\n--- Attention Details for Head {head_index} ---")

# Print token embeddings (Q1.7 format)
print("\nToken Embeddings:")
for i, token in enumerate(tokens):
    x_q17 = to_q17(x_enc[0, i, :])
    print(f"Token {i}:")
    print(f"  Float: {x_enc[0, i, :]}")
    print(f"  Q1.7:  {x_q17 / 2}")
# Print Q weights for this head (Q1.7 format)
print(f"\nQ Weights for Head {head_index}:")
Wq_q17 = to_q17(Wq[head_index])
for i, row in enumerate(Wq_q17):
    print(f"Row {i}:")
    print(f"  Float: {Wq[head_index][i]}")
    print(f"  Q1.7:  {row}")

# Calculate and print Q matrix with proper 32-bit accumulation
print("\nQ Matrix Calculations (32-bit accumulated values):")
q_mat_head = np.zeros((x_enc[0].shape[0], HEAD_DIM), dtype=np.int32)  # Store 32-bit accumulated values

for token_idx in range(x_enc[0].shape[0]):
    token_vec = x_enc[0, token_idx, :]
    token_vec_q17 = to_q17(token_vec).astype(np.int16)  # Input in Q1.7 (int8 -> int16)
    weights = Wq[head_index]
    weights_q17 = to_q17(weights).astype(np.int16)      # Weights in Q1.7 (int8 -> int16)
    
    print(f"\nToken {token_idx} Q calculation (32-bit accumulation):")
    for dim in range(HEAD_DIM):
        # Calculate dot product with 32-bit accumulation
        dot_sum_q17 = 0
        print(f"  Dim {dim} MAC Operations:")
        for i in range(D_MODEL):
            mult = token_vec_q17[i] * weights_q17[dim,i]  # int16 * int16 -> int32
            dot_sum_q17 += mult
            
            # Print first 3 MAC operations for debugging
            if i < 3:
                print(f"    {token_vec_q17[i]:4d} * {weights_q17[dim,i]:4d} = {mult:6d} | Accum: {dot_sum_q17:8d}")
        
        if D_MODEL > 3:
            print(f"    ... plus {D_MODEL-3} more MAC operations ...")
        
        # Store the 32-bit accumulated value (Q2.14 format)
        q_mat_head[token_idx, dim] = dot_sum_q17
        
        # Convert to final Q1.7 (divide by 128)
        q_val = dot_sum_q17 // 256
        print(f"  Final Q[{token_idx}][{dim}]:")
        print(f"    Raw 32-bit: {dot_sum_q17} (Q2.14)")
        print(f"    Scaled: {q_val} (Q1.7)")
        print(f"    Float: {dot_sum_q17 / (128.0 * 128.0):.6f}")

# Repeat the same pattern for K and V matrices
print("\nK Matrix Calculations (32-bit accumulated values):")
k_mat_head = np.zeros((x_enc[0].shape[0], HEAD_DIM), dtype=np.int32)

for token_idx in range(x_enc[0].shape[0]):
    token_vec = x_enc[0, token_idx, :]
    token_vec_q17 = to_q17(token_vec).astype(np.int16)
    weights = Wk[head_index]
    weights_q17 = to_q17(weights).astype(np.int16)
    
    print(f"\nToken {token_idx} K calculation (32-bit accumulation):")
    for dim in range(HEAD_DIM):
        dot_sum_q17 = 0
        for i in range(D_MODEL):
            mult = token_vec_q17[i] * weights_q17[dim,i]
            dot_sum_q17 += mult
  
            # Print first 3 MAC operations for debugging
            if i < 3:
                print(f"    {token_vec_q17[i]:4d} * {weights_q17[dim,i]:4d} = {mult:6d} | Accum: {dot_sum_q17:8d}")
        
        if D_MODEL > 3:
            print(f"    ... plus {D_MODEL-3} more MAC operations ...")
        
        # Store the 32-bit accumulated value (Q2.14 format)
        k_mat_head[token_idx, dim] = dot_sum_q17
        
        # Convert to final Q1.7 (divide by 128)
        k_val = dot_sum_q17 // 256
        print(f"  Final K[{token_idx}][{dim}]:")
        print(f"    Raw 32-bit: {dot_sum_q17} (Q2.14)")
        print(f"    Scaled: {k_val} (Q1.7)")
        print(f"    Float: {dot_sum_q17 / (128.0 * 128.0):.6f}")


# Similarly for V matrix
print("\nV Matrix Calculations (32-bit accumulated values):")
v_mat_head = np.zeros((x_enc[0].shape[0], HEAD_DIM), dtype=np.int32)

for token_idx in range(x_enc[0].shape[0]):
    token_vec = x_enc[0, token_idx, :]
    token_vec_q17 = to_q17(token_vec).astype(np.int16)
    weights = Wv[head_index]
    weights_q17 = to_q17(weights).astype(np.int16)
    
    print(f"\nToken {token_idx} V calculation (32-bit accumulation):")
    for dim in range(HEAD_DIM):
        dot_sum_q17 = 0
        for i in range(D_MODEL):
            mult = token_vec_q17[i] * weights_q17[dim,i]
            dot_sum_q17 += mult
       
            # Print first 3 MAC operations for debugging
            if i < 3:
                print(f"    {token_vec_q17[i]:4d} * {weights_q17[dim,i]:4d} = {mult:6d} | Accum: {dot_sum_q17:8d}")
        
        if D_MODEL > 3:
            print(f"    ... plus {D_MODEL-3} more MAC operations ...")
        
        # Store the 32-bit accumulated value (Q2.14 format)
        v_mat_head[token_idx, dim] = dot_sum_q17
        
        # Convert to final Q1.7 (divide by 128)
        v_val = dot_sum_q17 // 256
        print(f"  Final V[{token_idx}][{dim}]:")
        print(f"    Raw 32-bit: {dot_sum_q17} (Q2.14)")
        print(f"    Scaled: {v_val} (Q1.7)")
        print(f"    Float: {dot_sum_q17 / (128.0 * 128.0):.6f}")

# Q * K^T calculations using the 32-bit accumulated values
print("\nQ * K^T Calculations (using 32-bit values):")
for i in range(len(tokens)):
    for j in range(len(tokens)):
        score = 0
        for k in range(HEAD_DIM):
            # Multiply Q2.14 values and scale down to Q1.7
            score += (q_mat_head[i,k] * k_mat_head[j,k]) // (128 * 128)
        print(f"Attention score Q[{i}] * K[{j}]^T: {score} (raw), {score / 256.0:.4f} (scaled)")

# ---------------------------------------------------------------------------
# Final summary print-outs  ⬅️  UPDATED
# ---------------------------------------------------------------------------
print("\nFinal Q Matrix (seq_len × head_dim):")
print("Float:");  print(q_mat_head)
print("Int16:");   print(q_mat_head // 256)

print("\nFinal K Matrix (seq_len × head_dim):")
print("Float:");  print(k_mat_head)
print("Int16:");   print(k_mat_head // 256)

print("\nFinal V Matrix (seq_len × head_dim):")
print("Float:");  print(v_mat_head)
print("Int16:");   print(v_mat_head // 256)

# Show Q * K^T for first five elements with exact Q1.7 values
print("\nQ * K^T Matrix (First 5 Elements)")
print("--------------------------------")
print("Format: Q[token_idx][element_idx](Q1.7_value) * K[token_idx][element_idx](Q1.7_value) = product")
print("All values shown in Q1.7 fixed-point format\n")

num_elements_to_show = 5
for i, token_q in enumerate(tokens):
    print(f"\nQ Matrix - Token {i} ('{token_q}'):")
    q_row = q_mat_head[i]
    q_row_q17 = to_q17(q_row)
    
    for j, token_k in enumerate(tokens):
        print(f"\n  K Matrix - Token {j} ('{token_k}'):")
        k_row = k_mat_head[j]
        k_row_q17 = to_q17(k_row)
        
        for k in range(num_elements_to_show):
            q_val = q_row_q17[k]
            k_val = k_row_q17[k]
            product = (q_val/128.0) * (k_val/128.0)
            print(f"    Q[{i}][{k}] = {q_val:4d} * K[{j}][{k}] = {k_val:4d} → Product = {product:.4f}")
            
# Show detailed simulation with 6 passes and full accumulation
print("\n6-Pass MAC Simulation with Full Accumulation:")
print("------------------------------------------")
print("Format: 5 full passes + 6th pass showing first rows")
print(f"Elements: {HEAD_DIM} | Chunk Size: 15 | MACs/Row: 5 | Passes: 6\n")

num_tokens = len(tokens)
elements_per_chunk = 15
macs_per_row = 5
rows_per_chunk = elements_per_chunk // macs_per_row
total_passes = 5

# Initialize raw_scores array before the token loop
raw_scores = np.zeros((4, 4), dtype=np.int32)  # 4 Q heads x 4 chunks

for token_idx in range(num_tokens):
    print(f"\nToken {token_idx} ('{tokens[token_idx]}'):")
    
    # Calculate all products and store components for each Q
    
    for q_idx in range(0,4):  # Assuming num_q_heads is defined somewhere
        print(f"\nProcessing Q{q_idx}:")
        mac_details = []
        for dim in range(HEAD_DIM):
            # Cap values at 16-bit signed limits before multiplication
            q_val = max(-32768, min(32767, q_mat_head[q_idx, dim]))
            k_val = max(-32768, min(32767, k_mat_head[token_idx, dim]))
            
            q_val = q_val // 256
            k_val = k_val // 256
            
            product = q_val * k_val
            # Cap the product as well (since multiplication could overflow)
            product = max(-32768, min(32767, product))
            
            mac_details.append({
                'q': q_val,
                'k': k_val,
                'product': product
            })

        # Simulate passes with different display modes
        pass_accumulator = {pass_num: 0 for pass_num in range(1, total_passes + 1)}
        chunk_accumulators = {chunk_num: 0 for chunk_num in range(4)}  # Assuming 4 chunks (0-3)
        cross_pass_chunk_accumulators = {chunk_num: 0 for chunk_num in range(4)}

        for pass_num in range(1, total_passes + 1):
            print(f"\nPass {pass_num}:")
            pass_sum = 0
            token_idx = pass_num - 1
            
            # Pre-calculate Q slice for this pass
            q_pass_start = (pass_num - 1) * elements_per_chunk
            q_pass_end = min(q_pass_start + elements_per_chunk, 60) 
            
            for chunk_num, chunk_start in enumerate(range(0, 60, elements_per_chunk)):
                chunk_end = min(chunk_start + elements_per_chunk, 60)
                chunk_accumulator = 0
                
                # Pre-calculate K slice for this pass
                k_pass_start = (pass_num - 1) * elements_per_chunk
                k_pass_end = min(k_pass_start + elements_per_chunk, 60)
                
                # Special handling for 5th pass
                if pass_num == 5:
                    print("SPECIAL FINAL PASS PROCESSING (last 4 K values + zero padding)")
                    # Take last 4 elements from full K matrix
                    k_pass_slice = [max(-32768, min(32767, k_mat_head[chunk_num,dim])) // 256 for dim in range(HEAD_DIM-4, HEAD_DIM)]
                    k_pass_slice = k_pass_slice + [0]
                    # Also adjust Q slice to match (4 elements + zero pad)
                    q_pass_slice = [max(-32768, min(32767, q_mat_head[q_idx,dim])) // 256 for dim in range(HEAD_DIM-4, HEAD_DIM)]
                    q_pass_slice = q_pass_slice + [0]
                else:
                    # Normal pass processing
                    k_pass_slice = [max(-32768, min(32767, k_mat_head[chunk_num,dim])) // 256 for dim in range(k_pass_start, k_pass_end)]
                    q_pass_slice = [max(-32768, min(32767, q_mat_head[q_idx,dim])) // 256 for dim in range(q_pass_start, q_pass_end)]

                print(f"Pass Q Vals: {q_pass_slice}")
                print(f"Pass K Vals: {k_pass_slice}")
                print(f"Chunk Num: {chunk_num} Chunk Start: {chunk_start} Chunk End: {chunk_end}")

                # Calculate products (handles both normal and special pass cases)
                products = [max(-32768, min(32767, q * k)) for q, k in zip(q_pass_slice, k_pass_slice)]
            
                # Process rows
                chunk_rows = [products[i:i+macs_per_row] for i in range(0, len(products), macs_per_row)]
                q_rows = [q_pass_slice[i:i+macs_per_row] for i in range(0, len(q_pass_slice), macs_per_row)]
                k_rows = [k_pass_slice[i:i+macs_per_row] for i in range(0, len(k_pass_slice), macs_per_row)]
                
                for row_local, (q_vals, k_vals, prods) in enumerate(zip(q_rows, k_rows, chunk_rows)):
                    current_row = chunk_num * rows_per_chunk + row_local
                    row_sum = sum(prods)
                    chunk_accumulator += row_sum
                    pass_sum += row_sum

                    print(f"    Row {current_row+1}: ({row_local})")
                    for i, (q, k, p) in enumerate(zip(q_vals, k_vals, prods), start=1):
                        print(f"      MAC {i}: Q={q:4d} K={k:4d} → {p:6d}")
                    print(f"      Row sum: {row_sum:8d} | Chunk accum: {chunk_accumulator:8d}")
                    print("      " + "-"*40)

                chunk_accumulators[chunk_num] = chunk_accumulator
                cross_pass_chunk_accumulators[chunk_num] += chunk_accumulators[chunk_num]
                
                print(f"\nCHUNK {chunk_num} ACCUMULATION:")
                print(f"  Current pass chunk sum: {chunk_accumulator:8d}")
                print(f"  New cross-pass total:   {cross_pass_chunk_accumulators[chunk_num]:8d}")
                print("  " + "="*40)

            pass_accumulator[pass_num] = pass_sum

        # Print final accumulations for this Q
        print(f"\nFINAL ACCUMULATIONS FOR Q {q_idx}:")
        print("Per-chunk totals across all passes:")
        for chunk_num, total in cross_pass_chunk_accumulators.items():
            print(f"Chunk {chunk_num} total: {total:8d}")
            
            # Store the total in raw_scores array
            raw_scores[q_idx, chunk_num] = int(total)
            
# After all tokens and Q heads are processed, print the raw_scores matrix
print("\nFinal Raw Scores Matrix (4 Q heads × 4 chunks):")
print(raw_scores)     

# 2. Scale by 1/sqrt(head_dim)  ───────────────────────────────────────────────
# print("\n2. Scaling by 1/sqrt(head_dim):")
# scale_factor = 1 / np.sqrt(HEAD_DIM)
# print(f"  Scale factor: 1/sqrt({HEAD_DIM}) = {scale_factor:.6f}")

scaled_scores = (raw_scores * 1).astype(np.int32)
for i in range(min(4, len(tokens))):
    print(f" \n Scaled score token {i}: {raw_scores[i]} = {scaled_scores[i]} \n")

# -------------------------------------------------------
# 2.5 NUMERICAL-STABILITY MAX-SHIFT  (row-wise!)
# -------------------------------------------------------
row_max         = np.max(scaled_scores, axis=1, keepdims=True)      # (T,1)
shifted_scores  = (scaled_scores - row_max).astype(np.int32)

# verbose
for i in range(min(4, shifted_scores.shape[0])):
    print(f"  Row {i} max = {row_max[i,0]}")
    print(f"  Shifted row {i}: {shifted_scores[i]}")

# -------------------------------------------------------
# 3. 3rd-ORDER TAYLOR EXP APPROX  (per element, scalar math)
# -------------------------------------------------------
exp_scores = np.empty_like(shifted_scores, dtype=np.int32)

for r in range(shifted_scores.shape[0]):          # each token / row
    for c in range(shifted_scores.shape[1]):      # each col / head-chunk
        x      = shifted_scores[r, c]             # Q2.14
        x2     = (x * x)   >> 14                 # x²   (Q2.14)
        x3     = (x2 * x)  >> 14                 # x³   (Q2.14)

        exp_scores[r, c] = (1 << 14) + x + (x2 // 2) + (x3 // 3)

        if r < 2 and c < 2:      # trim spam
            print(f"  Row {r} Col {c}: x={x}, exp≈{exp_scores[r,c]}")

# -------------------------------------------------------
# 4. ROW-WISE Σexp  (still Q2.14)
# -------------------------------------------------------
sum_exp = np.sum(exp_scores, axis=1, keepdims=True)   # (T,1)
for i in range(min(4, sum_exp.shape[0])):
    terms = ' + '.join(str(v) for v in exp_scores[i])
    print(f"  Σexp row {i}: {terms} = {sum_exp[i,0]}")

# -------------------------------------------------------
# 5. SOFTMAX NORMALISATION
# -------------------------------------------------------
# result wanted in Q0.14 (0-to-1 probabilities)
softmax_q14 = (exp_scores << 14) // sum_exp           # integer divide

# -------------------------------------------------------
# 6. OPTIONAL: convert to float for display
# -------------------------------------------------------
softmax_float = softmax_q14.astype(np.float32) / (1 << 14)
print("\nSoft-max probabilities (float):")
for r in range(softmax_float.shape[0]):
    print(f"  Row {r}: {softmax_float[r]}")


# # Integer division for probabilities (using fixed-point math)
# print("\n5. Final Softmax Probabilities (Q0.16):")
# probs = np.zeros(len(tokens), dtype=np.int32)
# for i in range(len(tokens)):
#     # Multiply by 1<<16 before division to maintain precision
#     prob = (exp_scores[i] << 16) // sum_exp
#     probs[i] = prob
    
#     float_prob = prob / (1 << 16)
#     print(f"  P(token {i} '{tokens[i]}') = {exp_scores[i]}/{sum_exp} ≈ {float_prob:.6f}")
#     print(f"    => Attention weight: {float_prob:.4%}")
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
write_q17_weights(Wq[0], "tb_stim/q0_weights.txt")
write_q17_weights(Wk[0], "tb_stim/k0_weights.txt")
write_q17_weights(Wv[0], "tb_stim/v0_weights.txt")
write_q17_weights(Wo, "tb_stim/proj_layer_weights.txt")
write_q17_weights(W1, "tb_stim/fc_1_weights.txt")
write_q17_weights(W2, "tb_stim/fc_2_weights.txt")
