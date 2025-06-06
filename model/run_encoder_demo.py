#!/usr/bin/env python3
import time
import numpy as np
import torch, math
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from decimal import Decimal, getcontext

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
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model = AutoModel.from_pretrained(
    "Helsinki-NLP/opus-mt-en-ar", torch_dtype=torch.float32
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

seq_len = 6                         # = 6 tokens here
pe = (model.encoder                    # grab learned positional-emb matrix
           .embed_positions
           .weight[:seq_len, :D_MODEL]       # slice to (seq_len, 256)
           .detach()
           .cpu()
           .numpy())                         # → NumPy float32

x_enc = x + pe[None, :, :]                   # broadcast-add ⇒ (1, seq_len, 256)

print("\nFirst 5 embedding values for token 0 after pos-enc:")
for i in range(5):
    val = x_enc[0,0,i]
    print(f"  x_enc[0,0,{i}] = {val:.6f}")
# Q1.7 equivalents after pos-enc
print("Q1.7 fixed-point equivalents after pos-enc:")
for i in range(5):
    val = x_enc[0,0,i]
    q_fixed_un = int(np.round(val * 128))
    q_fixed = int(np.clip(np.round(val * 128), -128, 127))
    print(f"  {val:.6f} *128 → {q_fixed_un} (Q1.7: {q_fixed}/128 = {q_fixed/128:.6f})")

# Store token embeddings in a text file (VHDL friendly format)
user_tokens = [0, 1, 2, 3, 4, 5]  # Change or expand as needed

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
eps = 1e-5
mu  = x_enc.mean(axis=-1, keepdims=True)
var = x_enc.var(axis=-1, keepdims=True)
x_ln = (x_enc - mu) / np.sqrt(var + eps)

mha_out = multi_head_attention(x_ln, Wq, Wk, Wv, Wo)  # <= changed arg
x_enc   = x_enc + mha_out          # residual connection

print("\n[DEBUG] after 1st LN + residual (first token, 8 dims):",
      np.round(x_enc[0,0,:8], 4))
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
def compute_taylor_q345(scores):
    table = []
    int_vals = []

    print("\n========== Starting Taylor Softmax Computation ==========\n")
    for block in range(4):
        start_idx = 4 * block
        group = scores[start_idx: start_idx + 4]
        max_val = max(group)
        taylor_group = []

        print(f"\n--- Q{block} ---")
        print(f"Scores: {group}")
        print(f"Max score in block: {max_val}")
        
        for i, score in enumerate(group):
            z = score - max_val
            z_q15 = z
            z_sq = int(z_q15 * z_q15)
            z_cu = int(z_sq * z_q15)

            term1 = z_q15 << 30
            term2 = int(z_sq / 2) << 15
            term3 = ((z_cu * 21846)) 
            taylor = 1 + term1 + term2 + term3

            print(f"\nToken {i}:")
            print(f"  Raw Score     = {score}")
            print(f"  z = score - max = {z}")
            print(f"  z_q15 (>>8)    = {z_q15}")
            print(f"  z^2            = {z_sq}")
            print(f"  z^3            = {z_cu}")
            print(f"  Term1 (z<<30)  = {term1}")
            print(f"  Term2          = {term2}")
            print(f"  Term3 (approx) = {term3}")
            print(f"  Taylor Approx  = {taylor}")

            taylor_group.append(taylor)

            table.append([
                f"Q{block}", i, score, z, z_q15,
                term1, term2, term3, taylor
            ])

        taylor_sum = sum(taylor_group)
        print(f"\nTotal Taylor Sum: {taylor_sum}")
        
        softmax_fp = []
        for i in range(4):
            softmax_q015 = (taylor_group[i] / taylor_sum)  # Convert to Q0.15
            softmax_float = 1 - softmax_q015  # Convert to float

            print(f"Softmax[{i}] Q0.15  = {int(softmax_q015)}")
            print(f"Softmax[{i}] Float  = {softmax_float:.6f}")

            table[4 * block + i].append(taylor_sum)
            table[4 * block + i].append(int(softmax_q015))
            softmax_fp.append(round(softmax_float, 6))

        # Earlier in your script (right after imports), set decimal precision:
        getcontext().prec = 4  # 4 digits to ensure 3 significant digits in operations
        
        # Then modify the print section to:
        print(f"\nFinal Softmax Vector (Float): Q{block} → {softmax_fp}")
        print("Softmax Vector (7-bit integer representation):")
       
        for i, val in enumerate(softmax_fp):
            # Convert with precise decimal arithmetic
            int7_val = int(round(float(Decimal(str(val)) * Decimal(128))))
            int7_val = max(0, min(127, int7_val))
            int_vals.append(int7_val)
            print(f"  Element {i}: {val:.6f} → {int7_val} (0x{int7_val:02x})")
            print(f"    Precise calculation: {Decimal(str(val)):.6f} * 127 = {Decimal(str(val)) * Decimal(128):.3f} → rounded {int7_val}")
        print("-" * 60)
        # Earlier in your script (right after imports), set decimal precision:
        getcontext().prec = 4  # 4 digits to ensure 3 significant digits in operations
 
    return int_vals

# -------------------------------------------------------
# 6. ATTENTION OUTPUT CALCULATION (Softmax * V)
# -------------------------------------------------------

# First, let's get the softmax results in a usable format
softmax_results = compute_taylor_q345(raw_scores.flatten())

# Extract just the softmax float values from the results
if softmax_results is None:
    raise ValueError("Softmax computation failed - returned None")

# Extract just the softmax float values (they're at index -2 in each entry)
print("\nFinal Softmax Matrix (4x4):")
softmax_matrix = np.array(softmax_results).reshape(4, 4)
print(softmax_matrix)

print("\nFinal V Matrices (4x4):")
# Convert V matrices to numpy arrays (assuming they're in similar format as Q/K)
V_matrices = v_mat_head  // 256
print(V_matrices)

print("\n=== Calculating Attention Outputs ===")

# Compute head-specific softmax matrices and attention outputs
attention_outputs = []
softmax_matrices = []  # will hold a (NUM_HEADS × 4 × 4) tensor
attention_4x64 = []

for head_idx in range(NUM_HEADS):
    print(f"\nProcessing Head {head_idx}:")
    
    # Extract Q, K, V for this head (each is shape (4, HEAD_DIM))
    V_head = V_matrices[head_idx]

    print(f"Softmax matrix (4x4) for head {head_idx}:")
    print(softmax_matrix)

    for row in softmax_matrix:
            q15_row = "  ".join(f"{(val / (1 << 7)):8.3f}" for val in row)
            print(q15_row)
    
    print(f"\nV matrix (4x64) for head {head_idx} (first 5 columns shown):")
    print(V_matrices)
    # 3) Attention output (4×64), processed for head 0 (or macs=0)
    for macs in range(0, 4):  # Typically one head at a time
        print(f"\nAttention output (4x64) for head {macs}")
        
        attention_outputs = []  # Collect 64 output values
    
        for i in range(64):  # Loop over each feature dimension
            attention_output = 0.0
    
            # Compute weighted sum of V[j][i] using softmax weights for position `macs`
            for j in range(4):  # Loop over 4 tokens/rows
                attention_output += softmax_matrix[macs][j] * V_matrices[j][i]
    
            attention_outputs.append(attention_output)
    
        # Reshape the 64-length list into 8×8 for clearer display
        attention_array = np.array(attention_outputs).reshape(8, 8)
        attention_4x64.append(attention_outputs)
        
        # Print the 8×8 attention output with detailed breakdown for each value
        print("Formatted Attention Output (8x8) with MAC breakdowns:")
        for row_idx in range(8):
            for col_idx in range(8):
                index = row_idx * 8 + col_idx
                output_val = attention_outputs[index]
                
        print(f"Final Attention Output {macs} (8x8):")
        for row in attention_array:
            formatted_row = "  ".join(f"{val:8.2f}" for val in row)
            q15_row = "  ".join(f"{(val / (1 << 15)):8.3f}" for val in row)
            print(f"Float:  {formatted_row}")
            print(f"Q1.15:  {q15_row}")

def calculate_projection(attention_4x64, Wo):
    """
    Calculate projected output with detailed MAC breakdowns
    Args:
        attention_4x64: 4x64 attention matrix (float32)
        Wo: 64x256 projection matrix (float32)
    Returns:
        Projected outputs with detailed computation logs
    """
    # Convert to numpy arrays if they aren't already
    attention_4x64 = np.array(attention_4x64, dtype=np.float32)
    Wo = np.array(Wo, dtype=np.float32)
    
    # Initialize accumulators
    projected_outputs = []
    global_pass_counter = 0
    
    for att_idx in range(4):  # For each of the 4 tokens
        print(f"\n=== Processing Projected Attention {att_idx} ===")
        attn_vec = attention_4x64[att_idx]
        proj_out = np.zeros(256)
        
        for out_idx in range(256):  # For each output dimension
            wo_column = Wo[:, out_idx]
            macs = []
            row_accumulator = 0
            pass_accumulator = 0
            
            print(f"\nOutput element {out_idx}:")
            print("="*40)
            
            for i in range(64):  # For each of 64 MAC operations
                # Calculate MAC with proper scaling
                scaled_attn = round((math.ceil(attn_vec[i]/(2**8)))-1)
                scaled_wo = round((wo_column[i] * (2 ** 7)))
                print(f"Wo => {wo_column[i]} INT8 => {scaled_wo}")
                mac = scaled_attn * scaled_wo
                macs.append(mac)
                
                # Update accumulators
                row_accumulator += 1
                pass_accumulator += mac
                
                # Print every 5 MACs
                if (i + 1) % 5 == 0:
                    start_idx = i - 4
                    print("Individual MAC elements:")
                    for j in range(start_idx, i+1):
                        print(f"  MAC {j}: {round((math.ceil(attn_vec[j]/(2**8)))-1)} * {round((wo_column[i] * (2 ** 7)))} = {macs[j]}")
                    current_sum = sum(macs[start_idx:i+1])
                    print(f"Sum of MACs {start_idx:2d} to {i:2d}: {current_sum:12.2f}")
                    print(f"Pass Accum: {pass_accumulator:12.2f}")

            # Handle final group if not divisible by 5
            if 64 % 5 != 0:
                remaining = 64 % 5
                start_idx = 64 - remaining
                current_sum = sum(macs[start_idx:64])
                print(f"Sum of MACs {start_idx:2d} to {63:2d}: {current_sum:12.2f}")
                print("-"*30)
            
            # Final output value
            out_val = sum(macs)
            proj_out[out_idx] = out_val
            
            # Print final total for this output element
            print(f"\nTotal sum of 64 products: {out_val:12.2f}")
            
            # Update pass counter every 12 rows (256/4 = 64, 64/12 ≈ 5 full passes)
            if row_accumulator >= 12:
                global_pass_counter += 1
                print(f"\nPass {global_pass_counter} completed")
                row_accumulator = 0
        
        projected_outputs.append(proj_out)
        
        # Print token summary
        print(f"\nCompleted Token {att_idx} projection")
        print(f"Output shape: {proj_out.shape}")
        print("Sample values:", proj_out[:5])
    
    return np.array(projected_outputs)

proj_out = calculate_projection(attention_4x64, Wo)

print("\n--- Running FFN with debug ---")
def feed_forward_verbose(tok_idx, x, W_fc1, W_fc2,
                         mac_group: int = 5,
                         neuron_group: int = 12,
                         verbose: bool = True):
    """
    Fully-connected 256→512→256 feed-forward layer with exhaustive logging.

    ▸ Now converts the token vector and both weight matrices to signed-int8
      (Q1.7: round(val*128) clip [-128,127]) inside the function.
    ▸ All MACs are carried out in int16→int32 arithmetic; results are
      de-quantised back to float32 on return.
    ▸ Prints every individual product (a * b = p) plus the original
      group-sum diagnostics.

    Everything else – pass counts, grouping logic, ReLU placement – is
    unchanged.
    """
    import numpy as np                           # local to keep signature identical

    # ---------- Sanity checks ----------
    x      = np.asarray(x,      np.float32).reshape(-1)
    W_fc1  = np.asarray(W_fc1,  np.float32)
    W_fc2  = np.asarray(W_fc2,  np.float32)

    assert x.shape      == (256,),          "x must be 1×256"
    assert W_fc1.shape  == (512, 256),      "W_fc1 must be 512×256"
    assert W_fc2.shape  == (256, 512),      "W_fc2 must be 256×512"

    print(f"W_fc2: {W_fc2}")
    # ---------- Quantise to int8 ----------
    w_to_i8 = lambda a: np.clip(np.round(a * 128), -128, 127).astype(np.int8)
    x_to_i8 = lambda a: np.round(np.right_shift((a).astype(np.int16), 8))

    x_i8   = x_to_i8(x)           # (256,)
    W1_i8  = w_to_i8(W_fc1)       # (512,256)
    W2_i8  = w_to_i8(W_fc2)       # (256,512)

    hidden_i16 = np.empty(512, dtype=np.int16)

    # ---------- First FC (256 → 512) ----------
    row_acc          = 0
    pass_counter_hid = 0

    for n_idx in range(512):
        w_i8 = W1_i8[n_idx]                     # (256,)
        macs = []
        if verbose:
            print(f"\n=== Neuron {n_idx:03d} ===")

        acc32 = 0
        # ---- 256 MACs → 52 MAC-groups ----
        for m in range(256):
            a = int(x_i8[m])
            b = int(w_i8[m])
            prod = a * b                        # int16 * int16 → int32
            macs.append(prod)
            acc32 += prod

            if verbose:
                print(f"    MAC {m:03d}: {a:4d} * {b:4d} = {prod:6d}")

            # print running Σ every mac_group elements
            last_in_group = (m + 1) % mac_group == 0 or m == 255
            if last_in_group and verbose:
                g_start = m - (m % mac_group)
                g_sum   = sum(macs[g_start:m+1])
                print(f"  MACs {g_start:03d}-{m:03d}: Σ = {g_sum: 8d}")

        # ReLU (still int32), clamp to int16
        relu_val = max(0, acc32)
        hidden_i16[n_idx] = np.int16(np.clip(relu_val, 0, 32767))
        if verbose:
            print(f"  ReLU out (int16): {hidden_i16[n_idx]:6d}")

        # neuron-group pass bookkeeping
        row_acc += 1
        if row_acc == neuron_group:
            pass_counter_hid += 1
            if verbose:
                print(f"-- Hidden Pass {pass_counter_hid:02d} complete")
            row_acc = 0

    # add final partial pass (8 neurons)
    if row_acc:
        pass_counter_hid += 1
        if verbose:
            print(f"-- Hidden Pass {pass_counter_hid:02d} complete (final)")

    assert pass_counter_hid == 43, "Hidden layer must finish in 43 passes"

    if verbose:
        tot_act = hidden_i16.astype(np.int32).sum()
        print(f"\nHidden layer total activation sum (int32) = {tot_act: 8d}")

    # ---------- Second FC (512 → 256) ----------
    out_i32 = np.empty(256, dtype=np.int32)
    for o_idx in range(256):
        w2_i8 = W2_i8[o_idx]     
        macs = []               # (512,)
        if verbose:
            print(f"\n---Token {tok_idx} Output neuron {o_idx:03d} ---")
        acc32 = 0
        for m in range(512):
            a = int(np.right_shift(hidden_i16[m].astype(np.int16),4))
            b = int(w2_i8[m])
            prod = a * b
            macs.append(prod)
            acc32 += prod
            print(f"    MAC {m:03d}: {a:6d} * {b:4d} = {prod:8d}")
            if (m + 1) % mac_group == 0 or m == 511:
                g_start = m - (m % mac_group)
                g_sum = sum(macs[g_start:m+1])
                if verbose:
                    print(f"  MACs {g_start:03d}-{m:03d}: Σ = {g_sum: 10d}")
        # Apply ReLU before storing
        out_i32[o_idx] = max(0, acc32)
        if verbose:
            print(f" Token {tok_idx} Output[{o_idx}] (int32): {out_i32[o_idx]: 10d}")

    # Print final FFN output array (256 values)
    print("\nFinal FFN Output (256 values) after ReLU:")
    for i in range(0, 256, 8):  # Print 8 values per line
        line = ", ".join(f"{val:8d}" for val in out_i32[i:i+8])
        print(f"  [{i:3d}-{i+7:3d}]: {line}")

    # ---------- De-quantise back to float32 ----------
    # two Q1.7 multiplies → scale factor Q²
    return out_i32.astype(np.float32) / (128 * 128)

# Initialize container for all token outputs
all_ffn_outputs = []

for tok_idx, tok_vec in enumerate(proj_out):      # proj_out is (4,256)
    print(f"\n### Token {tok_idx} ###")
    ffn_out = feed_forward_verbose(tok_idx, tok_vec, W1, W2)  # shape = (256,)
    all_ffn_outputs.append(ffn_out)

# Convert to numpy array (4 tokens × 256 dims)
ffn_out = np.array(all_ffn_outputs)  # shape = (4,256)

# NEW CODE #
# 1. Add residual connection
# residual = x_enc[0]  # shape = (6,256) - but we only have 4 tokens processed
# residual = residual[:4]  # Take first 4 tokens to match our FFN outputs

# # Add residual (broadcasting works now that shapes match)
# ffn_out += residual[:, :256]

# # 2. Apply LayerNorm
# gamma = model.encoder.layers[-1].final_layer_norm.weight.detach().numpy()  # (512,)
# beta = model.encoder.layers[-1].final_layer_norm.bias.detach().numpy()     # (512,)

# # Only use first 256 dimensions of gamma/beta if needed
# if len(gamma) > 256:
#     gamma = gamma[:256]
#     beta = beta[:256]

# # Vectorized LayerNorm
# mean = ffn_out.mean(axis=1, keepdims=True)
# var = ffn_out.var(axis=1, keepdims=True)
# ffn_out = (ffn_out - mean) / np.sqrt(var + 1e-5)
# ffn_out = ffn_out * gamma + beta

print(f"FFN output shape: {ffn_out.shape}")  # Should be (4,256)
# Write **all token** FFN outputs to Q1.7 txt  ──────────────
ffn_out_file = Path("tb_stim/ffn_out.txt")
with open(ffn_out_file, "w") as f:
    f.write("-- FFN Output (Q1.7 Fixed-Point)\n")
    f.write("-- Format: one 256-int row per token\n")

    ffn_out_q17 = np.round(ffn_out * 128).astype(np.int32)   # (T,256)
    for row in ffn_out_q17:                                  # loop over tokens
        row_str = ", ".join(str(v) for v in row)
        f.write(f"({row_str}),\n")
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
