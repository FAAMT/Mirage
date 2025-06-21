#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def load_quantized_weights(root: Path, stem: str, shape):
    """Load quantized weights from .bin, .scale, and .zp files"""
    w = np.fromfile(root / f"{stem}.bin", dtype=np.int8).reshape(shape)
    s = np.fromfile(root / f"{stem}.scale", dtype=np.float32)
    z = np.fromfile(root / f"{stem}.zp", dtype=np.uint8)

    # Align scale/zp automatically
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

    return w, s, z

def format_weights(weights, quant_format="Q1.7"):
    """Format weights into the specified text format, processing full rows"""
    # Format the header
    shape_str = "x".join(map(str, weights.shape))
    header = f"-- Format: {shape_str} (dim0, dim1, ..., dimN);"
    
    # Format the weights data by full rows
    lines = []
    
    # Process each row separately
    for row_idx in range(weights.shape[0]):
        row_data = weights[row_idx, :].flatten().tolist()
        row_str = "(" + ", ".join(map(str, row_data)) + ")"
        lines.append(row_str)
    
    return header, lines

def write_weight_file(file_path, weight_name, quant_format, weights):
    """Write weights to a text file in the specified format"""
    with open(file_path, 'w') as f:
        f.write(f"-- {weight_name} ({quant_format})\n")
        header, weight_lines = format_weights(weights)
        f.write(header + "\n")
        for line in weight_lines:
            f.write(line + "\n")

def process_layer_weights(weights_dir, tb_dir, layer_idx, d_model=512, num_heads=8, head_dim=64):
    """Process all weight files for a given layer"""
    # Define all the weight files to process
    weight_files = [
        ("attn_out_proj_full", (d_model, d_model)),
        ("ffn_fc1_full", (2048, d_model)),
        ("ffn_fc2_full", (d_model, 2048)),
        ("key_full", (d_model, d_model)),
        ("query_full", (d_model, d_model)),
        ("value_full", (d_model, d_model)),
    ]
    
    # Process per-head weights
    for head_type in ["key", "query", "value"]:
        for head_idx in range(num_heads):
            weight_files.append((f"{head_type}_head{head_idx}", (head_dim, d_model)))
    
    # Process each weight file
    for stem, shape in weight_files:
        full_stem = f"layer{layer_idx}_{stem}"
        try:
            w, s, z = load_quantized_weights(weights_dir, full_stem, shape)
            output_file = tb_dir / f"{full_stem}.txt"
            write_weight_file(output_file, f"Layer {layer_idx} {stem}", "Q1.7", w)
            print(f"Successfully processed {full_stem}")
        except FileNotFoundError:
            print(f"Warning: Could not find {full_stem} files")
        except Exception as e:
            print(f"Error processing {full_stem}: {str(e)}")

def main():
    # Configuration
    WEIGHTS_DIR = Path("./weights")
    TB_DIR = Path("./tb_stim")
    NUM_LAYERS = 6  # Adjust based on your model
    D_MODEL = 512
    NUM_HEADS = 8
    HEAD_DIM = 64
    
    # Create output directory if it doesn't exist
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # Process each layer
    for layer_idx in range(NUM_LAYERS):
        print(f"\nProcessing layer {layer_idx}...")
        process_layer_weights(WEIGHTS_DIR, TB_DIR, layer_idx, D_MODEL, NUM_HEADS, HEAD_DIM)
    
    print("\nWeight file conversion completed!")

if __name__ == "__main__":
    main()