#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

def load_quantized_weights(root: Path, stem: str, shape):
    """Load and dequantize weights (matches verify_func.py version)"""
    w_path = root / f"{stem}.bin"
    s_path = root / f"{stem}.scale"
    zp_path = root / f"{stem}.zp"

    weights = np.fromfile(w_path, dtype=np.int8).reshape(shape)
    scale = np.fromfile(s_path, dtype=np.float32)
    zp = np.fromfile(zp_path, dtype=np.uint8)

    # Reshape scale/zp to (rows,1) for broadcasting
    if scale.ndim == 1:
        scale = scale[:, None]
    if zp.ndim == 1:
        zp = zp[:, None]

    return (weights.astype(np.float32) - zp.astype(np.float32)) * scale

def to_q17(x):
    """Convert float32 to Q1.7 fixed-point format"""
    return np.clip(np.round(x * 128), -128, 127).astype(np.int8)

def format_weights(weights, quant_format="Q1.7"):
    """Format dequantized weights into specified format"""
    # Convert to target quantization format
    if quant_format == "Q1.7":
        quant_weights = to_q17(weights)
    else:
        quant_weights = weights  # fallback to float
    
    # Format header
    shape_str = "x".join(map(str, weights.shape))
    header = f"-- Format: {shape_str} (dim0, dim1, ..., dimN);"
    
    # Format data rows
    lines = []
    for row_idx in range(quant_weights.shape[0]):
        row_data = quant_weights[row_idx, :].flatten().tolist()
        row_str = "(" + ", ".join(map(str, row_data)) + ")"
        lines.append(row_str)
    
    return header, lines

def write_weight_file(file_path, weight_name, quant_format, weights):
    """Write weights in consistent format with verification script"""
    with open(file_path, 'w') as f:
        f.write(f"-- {weight_name} ({quant_format})\n")
        header, weight_lines = format_weights(weights, quant_format)
        f.write(header + "\n")
        for line in weight_lines:
            f.write(line + "\n")

def process_layer_weights(weights_dir, tb_dir, layer_idx, d_model=512, num_heads=8, head_dim=64):
    """Process all weight files for a given layer"""
    weight_files = [
        ("attn_out_proj_full", (d_model, d_model)),
        ("ffn_fc1_full", (2048, d_model)),
        ("ffn_fc2_full", (d_model, 2048)),
        ("key_full", (d_model, d_model)),
        ("query_full", (d_model, d_model)),
        ("value_full", (d_model, d_model)),
    ]
    
    # Add per-head weights
    for head_type in ["key", "query", "value"]:
        for head_idx in range(num_heads):
            weight_files.append((f"{head_type}_head{head_idx}", (head_dim, d_model)))
    
    # Process each weight file
    for stem, shape in weight_files:
        full_stem = f"layer{layer_idx}_{stem}"
        try:
            w = load_quantized_weights(weights_dir, full_stem, shape)
            output_file = tb_dir / f"{full_stem}.txt"
            write_weight_file(output_file, f"Layer {layer_idx} {stem}", "Q1.7", w)
            print(f"Processed {full_stem}")
        except FileNotFoundError:
            print(f"Warning: Missing {full_stem} files")
        except Exception as e:
            print(f"Error processing {full_stem}: {str(e)}")

def generate_token_embeddings(tb_dir):
    """Generate token embeddings with positional encoding for a sample sentence using dynamic scaling"""
    CONFIG = {"model_name": "Helsinki-NLP/opus-mt-en-ar"}
    
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"], torch_dtype=torch.float32)
    tokenizer = MarianTokenizer.from_pretrained(CONFIG["model_name"])
    input_ids = tokenizer("My name is Fahad", return_tensors="pt")["input_ids"]
    original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_embeddings = model.model.shared(input_ids)

    # Hook to get positional encodings
    captured_values = {}
    def capture_pe(module, input, output):
        captured_values['encoder_input'] = input[0]

    encoder_layer = model.model.encoder.layers[0]
    hook_pe = encoder_layer.register_forward_hook(capture_pe)

    with torch.no_grad():
        encoder_output = model.model.encoder(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.long)
        )
    hook_pe.remove()

    pos_encoding = captured_values['encoder_input'] - token_embeddings
    combined_embeddings = token_embeddings + pos_encoding
    combined_embeddings_np = combined_embeddings.detach().cpu().numpy().astype(np.float32)
    
    # Calculate dynamic scaling parameters with headroom and limit
       # Calculate dynamic scaling parameters with headroom and limit
    actual_max = np.max(np.abs(combined_embeddings_np))
    scale_factor = min(7.0, actual_max * 1.05)  # 10% headroom
    inverse_scale = 1.0 / scale_factor if scale_factor != 0 else 1.0
    
    # Write each token embedding to separate files
    for i, token in enumerate(original_tokens):
        output_file = tb_dir / f"token_embeddings_{i}.txt"
        with open(output_file, 'w') as f:
            f.write(f"-- Token embedding with positional encoding (Q1.7) - Token {i}: '{token}'\n")
            shape_str = "x".join(map(str, combined_embeddings_np.shape[1:]))
            f.write(f"-- Format: {shape_str} (sequence_length, embedding_dim);\n")
            f.write(f"-- Scale factor: {inverse_scale}\n")
            f.write(f"-- Token: {token}\n")

            # Convert to Q1.7 using dynamic scaling
            token_embedding = combined_embeddings_np[0] * inverse_scale
            token_embedding = to_q17(token_embedding)
            ctr = 0
            for row in token_embedding:
                if i == ctr:
                    row_str = ", ".join(map(str, row))
                    f.write(f"({row_str})\n")
                
                ctr += 1

        print(f"Saved token embedding {i} ('{token}') to {output_file}")

def main():
    # Configuration
    WEIGHTS_DIR = Path("./weights")
    TB_DIR = Path("./tb_stim")
    NUM_LAYERS = 6
    D_MODEL = 512
    NUM_HEADS = 8
    HEAD_DIM = 64
    
    # Create directories
    WEIGHTS_DIR.mkdir(exist_ok=True)
    TB_DIR.mkdir(exist_ok=True)
    
    # Process files
    print("Generating token embeddings...")
    generate_token_embeddings(TB_DIR)
    
    for layer_idx in range(NUM_LAYERS):
        print(f"\nProcessing layer {layer_idx}...")
        process_layer_weights(WEIGHTS_DIR, TB_DIR, layer_idx, D_MODEL, NUM_HEADS, HEAD_DIM)
    
    print("\nConversion completed!")

if __name__ == "__main__":
    main()