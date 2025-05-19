#!/usr/bin/env python3
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianConfig
from onnxruntime.quantization import quantize_dynamic, QuantType

def create_fpga_compatible_model(student_dir: Path, output_dir: Path) -> MarianMTModel:
    """Load your sliced student config, adjust dims for FPGA, save PyTorch checkpoint."""
    cfg = MarianConfig.from_pretrained(student_dir)
    cfg.update({
        "d_model": 256,
        "encoder_ffn_dim": 128,
        "decoder_ffn_dim": 128,
        "encoder_attention_heads": 4,
        "decoder_attention_heads": 4,
        "attention_head_dim": 64,
    })
    model = MarianMTModel(cfg)
    model.save_pretrained(output_dir)
    return model

def export_to_onnx(model: MarianMTModel, onnx_dir: Path, seq_len: int = 4):
    """Export encoder and decoder submodules to ONNX, feeding decoder its float embeddings."""
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # build dummy inputs
    ids   = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    mask  = torch.ones(1, seq_len, dtype=torch.long, device=device)
    # encoder export
    enc_path = onnx_dir / "encoder_model.onnx"
    torch.onnx.export(
        model.model.encoder,
        (ids, mask),
        str(enc_path),
        input_names  = ["input_ids", "attention_mask"],
        output_names = ["encoder_hidden_states"],
        dynamic_axes = {
            "input_ids": {0:"batch",1:"seq"},
            "attention_mask": {0:"batch",1:"seq"},
            "encoder_hidden_states": {0:"batch",1:"seq"},
        },
        opset_version=13,
    )
    print(f"🔄 exported {enc_path.name}")

    # prepare decoder embeddings instead of ids
    emb_scale = getattr(model.model.encoder, "embed_scale", 1.0)
    dec_emb = model.model.decoder.embed_tokens(ids) * emb_scale

    # dummy encoder output
    enc_out = torch.randn(1, seq_len, model.config.d_model, device=device)

    # decoder export
    dec_path = onnx_dir / "decoder_model.onnx"
    torch.onnx.export(
        model.model.decoder,
        (None, None, dec_emb, enc_out, mask),
        str(dec_path),
        # here we name the inputs so ONNXRuntime knows decoder_inputs_embeds
        input_names  = [
            "input_ids",            # dummy None
            "attention_mask",       # dummy None
            "decoder_inputs_embeds",
            "encoder_hidden_states",
            "encoder_attention_mask",
        ],
        output_names = ["decoder_outputs"],
        dynamic_axes = {
            "decoder_inputs_embeds":       {0:"batch",1:"seq"},
            "encoder_hidden_states":       {0:"batch",1:"seq"},
            "encoder_attention_mask":      {0:"batch",1:"seq"},
            "decoder_outputs":             {0:"batch",1:"seq"},
        },
        opset_version=13,
    )
    print(f"🔄 exported {dec_path.name}")

    return enc_path, dec_path

def quantize_model(onnx_path: Path):
    """Apply ONNXRuntime dynamic INT8 quantization."""
    out = onnx_path.with_suffix(".int8.onnx")
    quantize_dynamic(
        model_input        = str(onnx_path),
        model_output       = str(out),
        weight_type        = QuantType.QInt8,
        op_types_to_quantize = ["MatMul","Attention","QAttention"]
    )
    print(f"✅ quantized {out.name}")

if __name__ == "__main__":
    STUDENT_DIR    = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/student_ckpt/sliced_init")
    FPGA_MODEL_DIR = STUDENT_DIR.with_name("student_fpga")

    print("▶️  Creating FPGA-compatible PyTorch student…")
    student = create_fpga_compatible_model(STUDENT_DIR, FPGA_MODEL_DIR)

    print("▶️  Exporting encoder/decoder to ONNX…")
    enc_onnx, dec_onnx = export_to_onnx(student, FPGA_MODEL_DIR, seq_len=4)

    print("▶️  Running dynamic INT8 quantization…")
    for path in (enc_onnx, dec_onnx):
        quantize_model(path)

    print("\n🎉 Done! Your quantized graphs are:")
    print(" •", FPGA_MODEL_DIR / "encoder_model.int8.onnx")
    print(" •", FPGA_MODEL_DIR / "decoder_model.int8.onnx")
