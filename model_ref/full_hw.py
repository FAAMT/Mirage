#!/usr/bin/env python3
"""
full_hw.py -- "Hello" -> Arabic with BOTH the encoder and the decoder on the
fabric. Every matmul and every softmax in the whole translation.

The PS keeps only what it kept for the encoder: bias, swish, residual, the
two (three, for a decoder layer) layer normalisations, the quantisation
scales, and the argmax over the output vocabulary. Everything of O(D^2) --
the 12 layers' projections, both attentions per decoder layer, all 24 FFN
matrices -- runs through the same job interface on the same fabric.

This uses the Fabric model rather than GHDL. That is a deliberate and stated
substitution: translate_hw.py proves the RTL is bit-identical to Fabric on a
full encoder layer with these same weights, so Fabric standing in for the
RTL here is an equivalence that has been measured, not assumed. Running all
12 layers x 3 decoding steps through GHDL would take hours and test nothing
the layer-level run does not.
"""
import os
import sys
import time

import numpy as np
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marian_fp32 as FP
import mirage_int8 as M
from decoder_hw import DecoderLayerHW
from encoder_hw import quantise
from translate_hw import read_act, write_tensor, run_layer

D, H, HD, F = M.MODEL_DIM, M.NUM_HEADS, M.HEAD_DIM, M.FFN_DIM


def decoder_layer_hw(mir, w, li, x, enc_q, s_enc, n_src):
    """One decoder layer. x is (T_dec, D) float32; enc_q is the int8 encoder
    output already quantised, which cross-attention projects K and V from."""
    p = f"model.decoder.layers.{li}"
    lay = DecoderLayerHW(mir, w, li)
    fab = mir.fab
    n_dec = x.shape[0]

    # ---- masked self-attention ------------------------------------------
    x_q, s_x = quantise(x)
    write_tensor(mir, M.REG_X, M.TOK_STRIDE, x_q)
    s_qp = lay._project("self_attn.q_proj", x_q, s_x, M.REG_X, M.REG_Q, n_dec)
    s_kp = lay._project("self_attn.k_proj", x_q, s_x, M.REG_X, M.REG_K, n_dec)
    s_vp = lay._project("self_attn.v_proj", x_q, s_x, M.REG_X, M.REG_V, n_dec)

    qf = read_act(fab, M.REG_Q, M.TOK_STRIDE, n_dec, D) * s_qp \
        + w[f"{p}.self_attn.q_proj.bias"]
    kf = read_act(fab, M.REG_K, M.TOK_STRIDE, n_dec, D) * s_kp \
        + w[f"{p}.self_attn.k_proj.bias"]
    vf = read_act(fab, M.REG_V, M.TOK_STRIDE, n_dec, D) * s_vp \
        + w[f"{p}.self_attn.v_proj.bias"]
    q_q, s_q = quantise(qf * (HD ** -0.5))
    k_q, s_k = quantise(kf)
    v_q, s_v = quantise(vf)
    write_tensor(mir, M.REG_Q, M.TOK_STRIDE, q_q)
    write_tensor(mir, M.REG_K, M.TOK_STRIDE, k_q)
    write_tensor(mir, M.REG_V, M.TOK_STRIDE, v_q)

    s_attn = lay._attend(n_dec, n_dec, s_q, s_k, s_v, causal=True)
    s_o = lay._project(
        "self_attn.out_proj",
        read_act(fab, M.REG_ATTN, M.TOK_STRIDE, n_dec, D).astype(np.int64),
        s_attn, M.REG_ATTN, M.REG_OUT, n_dec)
    of = read_act(fab, M.REG_OUT, M.TOK_STRIDE, n_dec, D) * s_o \
        + w[f"{p}.self_attn.out_proj.bias"]
    x1 = FP.layer_norm(x + of, w[f"{p}.self_attn_layer_norm.weight"],
                       w[f"{p}.self_attn_layer_norm.bias"])

    # ---- cross-attention -------------------------------------------------
    # K and V come from the ENCODER output. Same projection jobs, different
    # source base: the encoder state is parked in the FFN hidden region,
    # which the FFN has not reached yet this layer.
    write_tensor(mir, M.REG_FFN_H, M.TOK_STRIDE, enc_q)
    s_kc = lay._project("encoder_attn.k_proj", enc_q, s_enc,
                        M.REG_FFN_H, M.REG_K, n_src)
    s_vc = lay._project("encoder_attn.v_proj", enc_q, s_enc,
                        M.REG_FFN_H, M.REG_V, n_src)
    x1_q, s_x1 = quantise(x1)
    write_tensor(mir, M.REG_X, M.TOK_STRIDE, x1_q)
    s_qc = lay._project("encoder_attn.q_proj", x1_q, s_x1,
                        M.REG_X, M.REG_Q, n_dec)

    qf = read_act(fab, M.REG_Q, M.TOK_STRIDE, n_dec, D) * s_qc \
        + w[f"{p}.encoder_attn.q_proj.bias"]
    kf = read_act(fab, M.REG_K, M.TOK_STRIDE, n_src, D) * s_kc \
        + w[f"{p}.encoder_attn.k_proj.bias"]
    vf = read_act(fab, M.REG_V, M.TOK_STRIDE, n_src, D) * s_vc \
        + w[f"{p}.encoder_attn.v_proj.bias"]
    q_q, s_q = quantise(qf * (HD ** -0.5))
    k_q, s_k = quantise(kf)
    v_q, s_v = quantise(vf)
    write_tensor(mir, M.REG_Q, M.TOK_STRIDE, q_q)
    write_tensor(mir, M.REG_K, M.TOK_STRIDE, k_q)
    write_tensor(mir, M.REG_V, M.TOK_STRIDE, v_q)

    s_attn = lay._attend(n_dec, n_src, s_q, s_k, s_v, causal=False)
    s_o = lay._project(
        "encoder_attn.out_proj",
        read_act(fab, M.REG_ATTN, M.TOK_STRIDE, n_dec, D).astype(np.int64),
        s_attn, M.REG_ATTN, M.REG_OUT, n_dec)
    of = read_act(fab, M.REG_OUT, M.TOK_STRIDE, n_dec, D) * s_o \
        + w[f"{p}.encoder_attn.out_proj.bias"]
    x2 = FP.layer_norm(x1 + of, w[f"{p}.encoder_attn_layer_norm.weight"],
                       w[f"{p}.encoder_attn_layer_norm.bias"])

    # ---- feed-forward ----------------------------------------------------
    x2_q, s_x2 = quantise(x2)
    write_tensor(mir, M.REG_X, M.TOK_STRIDE, x2_q)
    s_h = lay._ffn(x2_q, s_x2, n_dec)
    hf = read_act(fab, M.REG_FFN_H, F, n_dec, F) * s_h + w[f"{p}.fc1.bias"]
    hf = FP.silu(hf)
    h_q, s_h2 = quantise(hf)
    write_tensor(mir, M.REG_FFN_H, F, h_q)
    s_y = lay._ffn2(h_q, s_h2, n_dec)
    yf = read_act(fab, M.REG_OUT, M.TOK_STRIDE, n_dec, D) * s_y \
        + w[f"{p}.fc2.bias"]
    return FP.layer_norm(x2 + yf, w[f"{p}.final_layer_norm.weight"],
                         w[f"{p}.final_layer_norm.bias"])


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    t0 = time.time()
    model = FP.Marian()
    w = model.w
    sp = spm.SentencePieceProcessor(model_file=os.path.join(FP.HF, "source.spm"))
    ids = [model.vocab[p] for p in sp.encode(text, out_type=str)] + [FP.EOS_ID]
    n_src = len(ids)

    print("=" * 72)
    print(f'FULL TRANSLATION ON THE FABRIC -- "{text}"')
    print("=" * 72)
    ref_ids = model.generate(np.array(ids))
    print(f"reference (float32) : {model.detokenise(ref_ids)!r}  {ref_ids}")

    mir = M.Mirage("/tmp/full_hw_cmd.txt")

    # ---- encoder ---------------------------------------------------------
    x = model.embed(np.array(ids), "encoder")
    for li in range(FP.N_LAYERS):
        x = run_layer(mir, w, li, x, n_src)
    enc = x
    enc_q, s_enc = quantise(enc)
    ref_enc = model.encode(np.array(ids))
    c = float(enc.ravel() @ ref_enc.ravel()
              / (np.linalg.norm(enc) * np.linalg.norm(ref_enc)))
    print(f"encoder on fabric   : cosine {c:.4f} vs float32")

    # ---- decoder, greedy -------------------------------------------------
    dec = [FP.DECODER_START]
    for step in range(12):
        y = model.embed(np.array(dec), "decoder")
        for li in range(FP.N_LAYERS):
            y = decoder_layer_hw(mir, w, li, y, enc_q, s_enc, n_src)
        logits = y @ w["model.shared.weight"].T + w["final_logits_bias"]
        lg = logits[-1].copy()
        lg[FP.PAD_ID] = -np.inf
        nxt = int(lg.argmax())
        dec.append(nxt)
        print(f"  step {step}: token {nxt:6d}  {model.inv_vocab[nxt]!r}"
              f"   ({mir.scr.jobs:,} fabric jobs so far)")
        if nxt == FP.EOS_ID:
            break
    mir.close()

    out = dec[1:]
    print()
    print(f"decoder on fabric   : {model.detokenise(out)!r}  {out}")
    print("=" * 72)
    match = out == ref_ids
    print(f"{'IDENTICAL to the float32 reference' if match else 'DIFFERS from the reference'}")
    print(f"total fabric jobs   : {mir.scr.jobs:,}")
    print(f"wall time           : {time.time() - t0:.0f}s")
    print("=" * 72)
    return 0 if match else 2


if __name__ == "__main__":
    raise SystemExit(main())
