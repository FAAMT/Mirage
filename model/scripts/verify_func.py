#!/usr/bin/env python3
import numpy as np
import math, torch
from pathlib import Path
from decimal import Decimal, getcontext

def load_quantized_weights(root: Path, stem: str, shape):
    w_path  = root / f"{stem}.bin"
    s_path  = root / f"{stem}.scale"
    zp_path = root / f"{stem}.zp"

    weights = np.fromfile(w_path,  dtype=np.int8 ).reshape(shape)
    scale   = np.fromfile(s_path,  dtype=np.float32)
    zp      = np.fromfile(zp_path, dtype=np.uint8)

    # ── NEW: reshape per-row scale / zp to broadcast over columns ──────────
    if scale.ndim == 1:
        scale = scale[:, None]       # (rows,1)
    if zp.ndim == 1:
        zp = zp[:, None]             # (rows,1)

    return (weights.astype(np.float32) - zp.astype(np.float32)) * scale

def to_q17(x):
    """Convert float32 to Q1.7 fixed-point representation."""
    return np.clip(np.round(x * 128), -128, 127).astype(np.int8)

def to_q117(x):
    """Convert float32 to Q1.17 fixed-point representation."""
    return np.clip(np.round(x * 65536), -65536, 65535).astype(np.int8)

def from_q17(x):
    """Convert Q1.7 fixed-point representation to float32."""
    return x.astype(np.float32) / 128.0

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

        getcontext().prec = 4  # 4 digits to ensure 3 significant digits in operations
        
        print(f"\nFinal Softmax Vector (Float): Q{block} → {softmax_fp}")
        print("Softmax Vector (7-bit integer representation):")
       
        for i, val in enumerate(softmax_fp):
            int7_val = int(round(float(Decimal(str(val)) * Decimal(128))))
            int7_val = max(0, min(127, int7_val))
            int_vals.append(int7_val)
            print(f"  Element {i}: {val:.6f} → {int7_val} (0x{int7_val:02x})")
            print(f"    Precise calculation: {Decimal(str(val)):.6f} * 127 = {Decimal(str(val)) * Decimal(128):.3f} → rounded {int7_val}")
        print("-" * 60)
        getcontext().prec = 4  # 4 digits to ensure 3 significant digits in operations
 
    return int_vals

def calculate_projection(attention_4x64, Wo):
    """
    Calculate projected output with detailed MAC breakdowns
    Args:
        attention_4x64: 4x64 attention matrix (float32)
        Wo: 64x256 projection matrix (float32)
    Returns:
        Projected outputs with detailed computation logs
    """
    attention_4x64 = np.array(attention_4x64, dtype=np.float32)
    Wo = np.array(Wo, dtype=np.float32)
    
    projected_outputs = []
    global_pass_counter = 0
    
    for att_idx in range(4):
        print(f"\n=== Processing Projected Attention {att_idx} ===")
        attn_vec = attention_4x64[att_idx]
        proj_out = np.zeros(256)
        
        for out_idx in range(256):
            wo_column = Wo[:, out_idx]
            macs = []
            row_accumulator = 0
            pass_accumulator = 0
            
            print(f"\nOutput element {out_idx}:")
            print("="*40)
            
            for i in range(64):
                scaled_attn = round((math.ceil(attn_vec[i]/(2**8)))-1)
                scaled_wo = round((wo_column[i] * (2 ** 7)))
                print(f"Wo => {wo_column[i]} INT8 => {scaled_wo}")
                mac = scaled_attn * scaled_wo
                macs.append(mac)
                
                row_accumulator += 1
                pass_accumulator += mac
                
                if (i + 1) % 5 == 0:
                    start_idx = i - 4
                    print("Individual MAC elements:")
                    for j in range(start_idx, i+1):
                        print(f"  MAC {j}: {round((math.ceil(attn_vec[j]/(2**8)))-1)} * {round((wo_column[i] * (2 ** 7)))} = {macs[j]}")
                    current_sum = sum(macs[start_idx:i+1])
                    print(f"Sum of MACs {start_idx:2d} to {i:2d}: {current_sum:12.2f}")
                    print(f"Pass Accum: {pass_accumulator:12.2f}")

            if 64 % 5 != 0:
                remaining = 64 % 5
                start_idx = 64 - remaining
                current_sum = sum(macs[start_idx:64])
                print(f"Sum of MACs {start_idx:2d} to {63:2d}: {current_sum:12.2f}")
                print("-"*30)
            
            out_val = sum(macs)
            proj_out[out_idx] = out_val
            
            print(f"\nTotal sum of 64 products: {out_val:12.2f}")
            
            if row_accumulator >= 12:
                global_pass_counter += 1
                print(f"\nPass {global_pass_counter} completed")
                row_accumulator = 0
        
        projected_outputs.append(proj_out)
        
        print(f"\nCompleted Token {att_idx} projection")
        print(f"Output shape: {proj_out.shape}")
        print("Sample values:", proj_out[:5])
    
    return np.array(projected_outputs)


# MHA Custom Function #

def multi_head_attention_forward(x: np.ndarray, attn, Wq_full: np.ndarray, Wk_full: np.ndarray, 
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

################################################
#   Uncomment this section for verbose output  #
################################################

#    # Q, K, and V MAC Projections with proper scaling
#     print("\n=== Multi-Head Projection Results ===")
#     for token_idx in range(seq_l):
#        print(f"\n=== Token {token_idx} Projections ===")
#        for head_idx in range(8):
#            print(f"\n=== Head {head_idx} ===")
   
#            # Calculate start and end indices for this head's weights
#            start_idx = head_idx * head_dim
#            end_idx = (head_idx + 1) * head_dim
   
#            for proj_type, W_proj, bias in [('Q', Wq_full.T, q_bias), 
#                                            ('K', Wk_full.T, k_bias), 
#                                            ('V', Wv_full.T, v_bias)]:
#                print(f"\n{proj_type} Projection (Elements {start_idx}-{end_idx-1}):")
   
#                # Store results for first 5 elements of this head
#                results = []
   
#                # Scaling factors based on theoretical [-7,7] range
   
#                # Adjust scaling factor based on actual max observed:
#                actual_max = np.max(np.abs(x))
#                scale_factor = min(7.0, actual_max * 1.05)  # 10% headroom
#                inverse_scale = 1.0 / scale_factor if scale_factor != 0 else 1.0
   
#                for elem_idx in range(64):
#                    actual_elem_idx = start_idx + elem_idx
#                    vhdl_total = 0
#                    py_total = 0.0
   
#                    # Keep Weight calculations
#                    for weight_idx in range(512):  # Input dimension
#                        x_val = x[0, token_idx, weight_idx] * inverse_scale  # Scale down FP32 input
#                        w_val = W_proj[actual_elem_idx, weight_idx]
   
#                        # VHDL-style (Q1.7 fixed-point)
#                        x_q17 = to_q17(x_val).astype(np.int32)
#                        w_q17 = to_q17(w_val).astype(np.int32)
#                        vhdl_total += x_q17 * w_q17
   
#                        # Python reference (float32) - scale back before accumulation
#                        py_total += (x_val * scale_factor) * w_val
   
#                    # Apply Q1.7 scaling and bias (with scale compensation)
#                    vhdl_scaled = (vhdl_total / (128**2)) * scale_factor
#                    py_scaled = py_total  # Already scaled back during accumulation
#                    final_vhdl = vhdl_scaled + bias[actual_elem_idx]
#                    final_py = py_scaled + bias[actual_elem_idx]
   
#                    results.append({
#                        'elem_idx': actual_elem_idx,
#                        'vhdl_raw': vhdl_total,
#                        'vhdl_scaled': vhdl_scaled,
#                        'vhdl_final': final_vhdl,
#                        'py_scaled': py_scaled,
#                        'py_final': final_py,
#                        'q17_vhdl': to_q17(final_vhdl),
#                        'q17_py': to_q17(final_py)
#                    })
#                    print("-"*150)
#                    print(f"{actual_elem_idx:8d} | {'VHDL':8s} | {vhdl_total:12d} | {vhdl_scaled:12.4f} | {final_vhdl:12.4f} | {to_q17(final_vhdl):6d}")
#                    print(f"         | {'Python':8s} |              | {py_scaled:12.4f} | {final_py:12.4f} | {to_q17(final_py):6d} | {abs(to_q17(final_vhdl) - to_q17(final_py)):6d}")
   
#                    # Tiling parameters
#                    num_tiles = 103
#                    macs_per_tile = 5
   
#                    # Tiling Calculations
#                    if elem_idx < 5:
#                        # print(f"\n{'Tile':5s} | {'MACs':9s} |   {'VHDL Sum':10s} | {'VHDL Accum':12s} | {'Python Accum FP32':12s} | {'Python Accum INT8':12s} | {'Diff':12s}")
#                        print("-"*150)
   
#                        tile_vhdl_total = 0
#                        tile_py_total = 0.0
   
#                        for tile_idx in range(num_tiles):
#                            start_mac = tile_idx * macs_per_tile
#                            end_mac = min((tile_idx + 1) * macs_per_tile, 512)
   
#                            # VHDL calculation (int32)
#                            tile_vhdl = 0
#                            tile_py = 0.0
   
#                            # print(f" {'IN_A INT8':10s} | {'IN_A FP32':12s} |   {'IN_B INT8':10s} | {'IN_B FP32':12s} | {'MAC SUM INT 8':12s} | {'MAC SUM FP32':12s}")
#                            for weight_idx in range(start_mac, end_mac):
#                                x_val = x[0, token_idx, weight_idx] * inverse_scale  # Scaled down
#                                w_val = W_proj[actual_elem_idx, weight_idx]
   
#                                # VHDL-style (Q1.7 fixed-point)
#                                x_q17 = to_q17(x_val).astype(np.int32)
#                                w_q17 = to_q17(w_val).astype(np.int32)
#                                mac = x_q17 * w_q17
#                                tile_vhdl += mac
   
#                                # Python reference (float32) - scale back before accumulation
#                                mac_py = (x_val * scale_factor) * w_val
#                                tile_py += mac_py
   
#                                # print(f"{x_q17 :12.4f} | {x_val:12.4f} | "
#                                #       f"{w_q17 :12.4f} | {w_val:12.4f} | "
#                                #       f"{mac :12.4f} | {mac_py:12.4f} | "
#                                #       f"{abs(mac - (mac_py * scale_factor) / 128):12.4f}")
   
#                            tile_vhdl_total += tile_vhdl
#                            tile_py_total += tile_py
#                            # print(f"\n{tile_idx:5d} | [{start_mac:3d}-{end_mac-1:3d}] | "
#                            #       f"{tile_vhdl :12.4f} | {tile_vhdl_total:12.4f} | "
#                            #       f" {tile_py_total:12.4f} | {round(tile_py_total * scale_factor):12.4f} |"
#                            #       f" {round(abs(tile_vhdl_total - round(tile_py_total * scale_factor)) / 128):12.4f}")

    # Full projections (single large matrix multiply)
    q_proj = x @ Wq_full.T  # [batch, seq, num_heads * head_dim]
    k_proj = x @ Wk_full.T 
    v_proj = x @ Wv_full.T 

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

    print("\nFFN Sanity Checks:")
    print(f"FC1 input range: [{x_fp32.min():.4f}, {x_fp32.max():.4f}]")
    print(f"FC1 output range: [{fc1_pre.min():.4f}, {fc1_pre.max():.4f}]")
    print(f"FC2 output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"NaN/Inf check: {np.any(np.isnan(out)) or np.any(np.isinf(out))}")
    # # console trace (token-local)
    # print(f"\n[Layer {layer_idx} | token {tok_idx}] FFN stages (first 4 dims)")
    # print("    in            :", np.round(x_fp32[:4],   4))
    # print("  FC1 pre-ReLU    :", np.round(fc1_pre[:4],  4))
    # print("  FC1 post-ReLU   :", np.round(fc1_post[:4], 4))
    # print("  FC2 (output)    :", np.round(out[:4],      4))

    return fc1_pre.astype(np.float32), fc1_post.astype(np.float32), out.astype(np.float32)

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

def inspect_head_alignment(name, fp32_W, int8_W, num_heads=8,
                           head_dim=64, thr=0.20, max_hits=30):
    """
    Print suspicious diff entries (abs diff > thr) per head.
    Also test a 'transpose hypothesis' to catch 64x512 ↔ 512x64 mix-ups.
    """
    H, D = fp32_W.shape          # (512, 512)   for Marian
    print(f"\n=== Inspecting {name}: {H}x{D} ===")

    bad_total, bad_after_T = 0, 0

    for h in range(num_heads):
        c0,  c1  = h*head_dim, (h+1)*head_dim          # head-column slice
        fp_head  = fp32_W[:, c0:c1]                    # (512, 64)
        q_head   = int8_W[:, c0:c1]                    # what you *think* is right
        q_head_T = int8_W_T[:, c0:c1] if (int8_W_T:=int8_W.T).shape==fp32_W.shape else None

        # --- element-wise diff in the assumed orientation ---------------
        diff = np.abs(fp_head - q_head)
        rows, cols = np.where(diff > thr)
        bad_total += rows.size

        if rows.size:
            print(f"\n–– Head {h}  suspicious elements (>{thr}):")
            for r, c in zip(rows, cols):
                print(f"  row {r:3d} col {c:2d} │ fp32 {fp_head[r,c]: .4f}"
                      f"  vs  int8 {q_head[r,c]: .4f}  Δ={diff[r,c]: .4f}")
                if max_hits and (max_hits:=max_hits-1)==0:
                    print("  … further hits suppressed\n"); break

        # --- same rows/cols but with quant matrix transposed ------------
        if q_head_T is not None:
            diff_T = np.abs(fp_head - q_head_T)
            bad_after_T += np.count_nonzero(diff_T > thr)

    # quick summary
    print(f"\n{bad_total} bad cells in normal orientation.")
    if q_head_T is not None:
        print(f"{bad_after_T} bad cells if INT-8 matrix were transposed.\n"
              f"{'⚠️  Likely orientation mismatch!' if bad_after_T*4 < bad_total else '✓ orientation looks fine.'}")
