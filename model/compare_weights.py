#!/usr/bin/env python3
import sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from verify_func import inspect_head_alignment

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
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"
WEIGHTS_DIR = Path("./weights")
NUM_HEADS = 8
HEAD_DIM = 64
D_MODEL = 512

def print_sample_elements(test_weights, quant_weights, name, layer_idx, num_elements=5):
    print(f"\n=== Layer {layer_idx} {name} Sample Elements ===")
    print("Test values (first few elements):", test_weights.flatten()[:num_elements].round(6))
    print("Quant values (first few elements):", quant_weights.flatten()[:num_elements].round(6))

def print_comparison(test_weights, quant_weights, name, layer_idx, num_rows=3, num_cols=5):
    print(f"\n=== Layer {layer_idx} {name} Weight Comparison ===")
    print(f"Test weights shape: {test_weights.shape}")
    print(f"Quant weights shape: {quant_weights.shape}")
    
    max_diff = np.max(np.abs(test_weights - quant_weights))
    avg_diff = np.mean(np.abs(test_weights - quant_weights))
    print(f"\nMax difference: {max_diff:.6f}")
    print(f"Avg difference: {avg_diff:.6f}")
    
    # Print sample comparisons
    # for i in range(min(num_rows, test_weights.shape[0])):
    #     print(f"\nRow {i} comparison:")
    #     print("Test  :", test_weights[i, :num_cols].round(6))
    #     print("Quant :", quant_weights[i, :num_cols].round(6))
    #     print("Diff  :", (test_weights[i, :num_cols] - quant_weights[i, :num_cols]).round(6))

def compare_heads(test_weights, quant_heads, name, layer_idx):
    print(f"\n=== Layer {layer_idx} Per-Head Comparison for {name} ===")
    for head_idx in range(NUM_HEADS):
        start = head_idx * HEAD_DIM
        end = (head_idx + 1) * HEAD_DIM
        
        test_head  = test_weights[start:end, :]
        quant_head = quant_heads[head_idx]
        
        max_diff = np.max(np.abs(test_head - quant_head))
        avg_diff = np.mean(np.abs(test_head - quant_head))
        
        print(f"\nHead {head_idx}:")
        print(f"Max diff: {max_diff:.6f}  Avg diff: {avg_diff:.6f}")
        
        # Print first row comparison
        # print("Test row :", test_head[0, :5].round(6))
        # print("Quant row:", quant_head[0, :5].round(6))

def main():
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    
    # Process each layer
    for layer_idx, layer in enumerate(model.model.encoder.layers):
        attn = layer.self_attn

        # Get test weights directly from model
        Wq_test = attn.q_proj.weight.detach().numpy()
        Wk_test = attn.k_proj.weight.detach().numpy()
        Wv_test = attn.v_proj.weight.detach().numpy()
        Wo_test = attn.out_proj.weight.detach().numpy()
        W1_test = layer.fc1.weight.detach().numpy()
        W2_test = layer.fc2.weight.detach().numpy()

        # Load quantized weights for this layer
        Wq_heads = [load_quantized_weights(WEIGHTS_DIR, f"layer{layer_idx}_query_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
        Wk_heads = [load_quantized_weights(WEIGHTS_DIR, f"layer{layer_idx}_key_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
        Wv_heads = [load_quantized_weights(WEIGHTS_DIR, f"layer{layer_idx}_value_head{i}", (HEAD_DIM, D_MODEL)) for i in range(NUM_HEADS)]
        
        # Reconstruct full matrices
        Wq_quant = np.concatenate(Wq_heads, axis=0)
        Wk_quant = np.concatenate(Wk_heads, axis=0)
        Wv_quant = np.concatenate(Wv_heads, axis=0)
        
        # Load other quantized weights for this layer
        Wo_quant = load_quantized_weights(WEIGHTS_DIR, f"layer{layer_idx}_attn_out_proj_full", (D_MODEL, D_MODEL))
        W1_quant = load_quantized_weights(WEIGHTS_DIR, f"layer{layer_idx}_ffn_fc1_full", (2048, 512))
        W2_quant = load_quantized_weights(WEIGHTS_DIR, f"layer{layer_idx}_ffn_fc2_full", (512, 2048))
        
        mis = np.mean(np.abs(W1_test - W1_quant))   # transpose candidate

        print("avg diff if transposed:", mis)
        for name, fp, q in [("FC1", W1_test, W2_quant.T), ("FC2", W2_test, W1_quant.T)]:
            print(f"Layer {layer_idx} {name} max diff:", np.max(abs(fp - q)), 
                  "avg diff:", np.mean(abs(fp - q)))

        print_comparison(Wq_test, Wq_quant, "Query Weights", layer_idx)
        print_sample_elements(Wq_test, Wq_quant, "Query Weights", layer_idx)
        compare_heads(Wq_test, Wq_heads, "Query", layer_idx)
        
        print_comparison(Wk_test, Wk_quant, "Key Weights", layer_idx)
        print_sample_elements(Wk_test, Wk_quant, "Key Weights", layer_idx)
        compare_heads(Wk_test, Wk_heads, "Key", layer_idx)
        
        print_comparison(Wv_test, Wv_quant, "Value Weights", layer_idx)
        print_sample_elements(Wv_test, Wv_quant, "Value Weights", layer_idx)
        compare_heads(Wv_test, Wv_heads, "Value", layer_idx)
        
        # print_comparison(Wo_test, Wo_quant, "Output Projection", layer_idx)
        # print_sample_elements(Wo_test, Wo_quant, "Output Projection", layer_idx)
        
        print_comparison(W1_test, W2_quant.T, "FFN FC1", layer_idx)
        print_sample_elements(W1_test, W2_quant.T, "FFN FC1", layer_idx)
        
        print_comparison(W2_test, W1_quant.T, "FFN FC2", layer_idx)
        print_sample_elements(W2_test, W1_quant.T, "FFN FC2", layer_idx)

if __name__ == "__main__":
    main()