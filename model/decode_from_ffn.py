#!/usr/bin/env python3

import argparse, re, numpy as np, torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    modeling_outputs,
)

# ─────────────────────────── CLI ────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--ffn_txt", required=True,
               help="Path to FFN output text (256 Q1.7 ints)")
p.add_argument("--model_name", default="Helsinki-NLP/opus-mt-ar-en")
p.add_argument("--max_len", type=int, default=64)
args = p.parse_args()

# ───────────────────── load + parse FFN text ───────────────
nums = []
with open(args.ffn_txt, "r") as fh:
    for ln in fh:
        if ln.lstrip().startswith("--"):             # skip comments
            continue
        ln = ln.strip(" (),;\n")
        if not ln:
            continue
        nums.extend(re.split(r"[,\s]+", ln))

vals = [float(v) for v in nums if v]
assert len(vals) == 256, f"Expected 256 values, got {len(vals)}"

ffn_vec = (np.array(vals, dtype=np.float32) / 128.0).reshape(1, 256)

# ─────────────────── model + tokenizer ─────────────────────
tok   = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name, torch_dtype=torch.float32
        ).eval()

# zero-pad to 512 hidden dims
enc_hidden = np.concatenate([ffn_vec, np.zeros_like(ffn_vec)], axis=-1)
enc_hidden_t = torch.from_numpy(enc_hidden).unsqueeze(1)        # (1,1,512)

enc_out = modeling_outputs.BaseModelOutput(last_hidden_state=enc_hidden_t)

decoder_input_ids = torch.tensor([[tok.pad_token_id]])

ids = model.generate(
        decoder_input_ids      = decoder_input_ids,
        encoder_outputs        = enc_out,
        max_new_tokens         = args.max_len,
        eos_token_id           = tok.eos_token_id,
        pad_token_id           = tok.pad_token_id,
        use_cache              = True,
     )

print("\n📝 Decoder output")
print("------------------")
decoded_sentence = tok.decode(ids[0], skip_special_tokens=True)
print(decoded_sentence)

# ── token-by-token trace ─────────────────────────────
print("\nToken-by-token trace (step, id, token):")
for step, tid in enumerate(ids[0].tolist()):
    token_txt = tok.convert_ids_to_tokens(tid)
    print(f"  {step:02d}: {tid:5d}  -> {token_txt}")

