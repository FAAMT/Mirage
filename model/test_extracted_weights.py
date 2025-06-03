#!/usr/bin/env python3
# ------------------------------------------------------------
# Test extracted & quantized INT-8 weights for a custom
# Transformer accelerator by simulating a forward pass in Python.
# Supports single-head emulation when NUM_HEADS=1.
# ------------------------------------------------------------
import numpy as np
from pathlib import Path
import math, sys

# --- Configuration (should match accelerator specs) ---
D_MODEL = 256
NUM_HEADS = 1
HEAD_DIM = 64
D_FF = 512
SEQ_LEN = 10  # Example sequence length for testing
BATCH_SIZE = 1  # Example batch size

# --- Helper Functions ---
def dequantize(int8_data: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
    """Dequantize INT8 data to float32."""
    if scale.ndim == 0:
        return (int8_data.astype(np.float32) - zero_point.astype(np.float32)) * scale.astype(np.float32)
    elif scale.ndim == 1:
        return ((int8_data.astype(np.float32) - zero_point.reshape(-1,1).astype(np.float32)) \
                * scale.reshape(-1,1).astype(np.float32))
    else:
        raise ValueError(f"Unexpected dimensions for scale: {scale.ndim}")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def load_sliced_weights(weights_dir: Path, name_stem: str, expected_shape: tuple) -> np.ndarray:
    bin_file = weights_dir / f"{name_stem}.bin"
    scale_file = weights_dir / f"{name_stem}.scale"
    zp_file = weights_dir / f"{name_stem}.zp"
    if not all([bin_file.exists(), scale_file.exists(), zp_file.exists()]):
        raise FileNotFoundError(f"Weight files for {name_stem} not found in {weights_dir}")

    int8_data = np.fromfile(bin_file, dtype=np.int8).reshape(expected_shape)
    scale = np.fromfile(scale_file, dtype=np.float32)
    zero_point = np.fromfile(zp_file, dtype=np.uint8)
    if scale.size == 1: scale = scale[0]
    if zero_point.size == 1: zero_point = zero_point[0]
    return dequantize(int8_data, scale, zero_point)


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    matmul_qk = np.matmul(q, k.swapaxes(-2, -1))
    dk = float(k.shape[-1])
    scaled_logits = matmul_qk / math.sqrt(dk)
    if mask is not None:
        scaled_logits += (mask * -1e9)
    weights = softmax(scaled_logits, axis=-1)
    return np.matmul(weights, v)


def multi_head_attention_forward(x: np.ndarray, list_Wq: list, list_Wk: list, list_Wv: list, W_o_full: np.ndarray) -> np.ndarray:
    """
    Simulates Multi-Head Attention. If NUM_HEADS=1, uses single head and slices W_o accordingly.
    """
    batch_s, seq_l, _ = x.shape
    head_out = None
    # only one head loop
    for i in range(NUM_HEADS):
        q_h = x @ list_Wq[i].T
        k_h = x @ list_Wk[i].T
        v_h = x @ list_Wv[i].T
        head_out = scaled_dot_product_attention(q_h, k_h, v_h)
    # head_out shape: (batch, seq, HEAD_DIM)

    # Output projection: map HEAD_DIM->D_MODEL if single head
    if NUM_HEADS == 1:
        # slice full projection 256x256 to 256xHEAD_DIM
        W_o = W_o_full[:, :HEAD_DIM]  # (256,64)
    else:
        W_o = W_o_full  # (256,NUM_HEADS*HEAD_DIM)

    # apply projection: head_out (B, L, HEAD_DIM or D_MODEL) @ W_o.T
    return np.matmul(head_out, W_o.T)


def feed_forward_network_forward(x: np.ndarray, W_fc1: np.ndarray, W_fc2: np.ndarray) -> np.ndarray:
    h = np.matmul(x, W_fc1.T)
    h_relu = np.maximum(0, h)
    return np.matmul(h_relu, W_fc2.T)


def main_test(weights_directory_str: str):
    weights_dir = Path(weights_directory_str)
    if not weights_dir.is_dir():
        print(f"ERROR: Weights directory not found: {weights_directory_str}")
        return

    all_Wq = [load_sliced_weights(weights_dir, f"query_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
    all_Wk = [load_sliced_weights(weights_dir, f"key_head{i}",   (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
    all_Wv = [load_sliced_weights(weights_dir, f"value_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
   # Load attention output projection weights
    W_o_full = load_sliced_weights(weights_dir, "attn_out_proj", (round(D_MODEL / 4), D_MODEL))
    print(f"W_o_full shape: {W_o_full.shape}")
    print("W_o_full sample (first 2 rows, 5 cols):")
    print(W_o_full[:2, :5])
    
    # Load FFN layer 1 weights
    W_fc1 = load_sliced_weights(weights_dir, "ffn_fc1", (D_FF, D_MODEL))
    print(f"W_fc1 shape: {W_fc1.shape}")
    print("W_fc1 sample (first 2 rows, 5 cols):")
    print(W_fc1[:2, :5])
    
    # Load FFN layer 2 weights
    W_fc2 = load_sliced_weights(weights_dir, "ffn_fc2", (D_MODEL, D_FF))
    print(f"W_fc2 shape: {W_fc2.shape}")
    print("W_fc2 sample (first 2 rows, 5 cols):")
    print(W_fc2[:2, :5])


    print("--- Weights Loaded Successfully ---")
    x = np.random.rand(BATCH_SIZE, SEQ_LEN, D_MODEL).astype(np.float32)
    print(f"Sample Input x shape: {x.shape}")

    print("--- Running Multi-Head Attention ---")
    mha_out = multi_head_attention_forward(x, all_Wq, all_Wk, all_Wv, W_o_full)
    print(f"MHA output shape: {mha_out.shape}")

    print("--- Running Feed-Forward Network ---")
    ffn_out = feed_forward_network_forward(mha_out, W_fc1, W_fc2)
    print(f"FFN output shape: {ffn_out.shape}")

    print("--- Test Forward Pass Completed ---")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} /path/to/weights_dir")
    main_test(sys.argv[1])