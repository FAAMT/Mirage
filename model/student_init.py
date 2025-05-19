#!/usr/bin/env python3
# student_init.py -----------------------------------------------------------
# Create Marian encoder/decoder with:
#   d_model      = 256
#   heads        = 1
#   FFN hidden   = 128   (fc1 will be 128×64 after slicing)
#   FFN output   = 256   (fc2 will be 256×128 after slicing)
# --------------------------------------------------------------------------

from pathlib import Path
from transformers import MarianConfig, MarianMTModel

# ---------- student geometry ----------------------------------------------
cfg = MarianConfig(
    vocab_size=32000,              # keep original tokenizer size
    d_model=256,                  # *** reduced width ***
    encoder_layers=1,
    decoder_layers=1,
    encoder_attention_heads=1,    # *** single head ***
    decoder_attention_heads=1,
    encoder_ffn_dim=128,          # *** hidden 128 ***
    decoder_ffn_dim=128,
    pad_token_id=0,
    bos_token_id=0,
    eos_token_id=0
)

student = MarianMTModel(cfg)      # randomly initialised (N(0,0.02))

out_dir = Path("student_ckpt")
out_dir.mkdir(exist_ok=True)
student.save_pretrained(out_dir)
cfg.save_pretrained(out_dir)

print(f"✅  blank student saved to {out_dir.resolve()}")
print("   shapes  Q,K,V : 64×256   •  fc1 : 128x64   •  fc2 : 256x128")
