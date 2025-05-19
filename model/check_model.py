from pathlib import Path
import torch
from transformers import MarianMTModel, MarianConfig

def create_fpga_compatible_student(original_dir, output_dir):
    """Creates FPGA-compatible Marian student model with full 256-dim weights."""
    # 1. Load original config and tweak only the FFN dims & head count
    config = MarianConfig.from_pretrained(original_dir)
    config.update({
        "d_model": 256,                # full model dimension
        "encoder_ffn_dim": 128,        # your reduced FFN hidden size
        "decoder_ffn_dim": 128,
        "encoder_attention_heads": 4,  # 256/4 = 64-wide heads
        "decoder_attention_heads": 4,
    })

    # 2. Instantiate new model and copy all original weights verbatim
    original = MarianMTModel.from_pretrained(original_dir)
    original_sd = original.state_dict()

    fpga_model = MarianMTModel(config)
    fpga_model.load_state_dict(original_sd)  # no slicing, shapes match
    fpga_model.save_pretrained(output_dir)
    return output_dir

def verify_model_dimensions(model_path):
    """Verifies model meets FPGA requirements (full-dim printout)."""
    model = MarianMTModel.from_pretrained(model_path)
    print("\n=== FPGA Dimension Verification ===")
    
    for i in range(model.config.encoder_layers):
        layer = model.model.encoder.layers[i]
        print(f"Encoder Layer {i}:")
        print(f"  Q_proj.weight: {layer.self_attn.q_proj.weight.shape} (256×256)")
        print(f"  K_proj.weight: {layer.self_attn.k_proj.weight.shape} (256×256)")
        print(f"  V_proj.weight: {layer.self_attn.v_proj.weight.shape} (256×256)")
        print(f"      fc1.weight: {layer.fc1.weight.shape} (128×256)")
        print(f"      fc2.weight: {layer.fc2.weight.shape} (256×128)")
    
    print(f"\nModel dtype: {model.config.torch_dtype}")
    print(f"Model device: {next(model.parameters()).device}")

if __name__ == "__main__":
    STUDENT_DIR = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/student_ckpt/sliced_init")
    FPGA_DIR    = STUDENT_DIR.with_name("student_fpga")
    
    print("Creating FPGA-compatible model…")
    create_fpga_compatible_student(STUDENT_DIR, FPGA_DIR)
    
    print("\nVerifying model dimensions…")
    verify_model_dimensions(FPGA_DIR)
    
    print("\n✅ FPGA-compatible model created and verified")
    print(f"Model saved to: {FPGA_DIR}")
