#!/usr/bin/env python3
"""
diagnose.py -- stage-by-stage accuracy of the int8 path against float32.

Runs the SAME job schedule as translate_hw.py but only through the Fabric
model, not GHDL. The RTL is already proven bit-identical to Fabric, so
anything this finds is a quantisation problem, and this runs in seconds
instead of minutes.

Reports cosine similarity against the float32 reference at every point the
tensor crosses the PL/PS boundary, so a collapse can be attributed to one
operation rather than to "int8".
"""
import os
import sys

import numpy as np
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marian_fp32 as FP
import mirage_int8 as M
from encoder_hw import EncoderLayerHW, quantise
from translate_hw import read_act, write_tensor

D, H, HD, F = M.MODEL_DIM, M.NUM_HEADS, M.HEAD_DIM, M.FFN_DIM


def cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    model = FP.Marian()
    w = model.w
    sp = spm.SentencePieceProcessor(model_file=os.path.join(FP.HF, "source.spm"))
    ids = [model.vocab[p] for p in sp.encode("Hello", out_type=str)] + [FP.EOS_ID]
    ntok = len(ids)

    tr = {}
    ref = model.encode(np.array(ids), trace=tr)
    x = model.embed(np.array(ids), "encoder")

    mir = M.Mirage(os.path.join("/tmp", "diag_cmd.txt"))
    fab = mir.fab
    nlay = int(os.environ.get("MIRAGE_LAYERS", 6))

    for li in range(nlay):
        p = f"model.encoder.layers.{li}"
        lay = EncoderLayerHW(mir, w, li)
        print(f"\n--- layer {li} " + "-" * 55)

        x_q, s_x = quantise(x)
        print(f"  input            cos {cos(x_q * s_x, x):.4f}   s_x={s_x:.5f}")
        write_tensor(mir, M.REG_X, M.TOK_STRIDE, x_q)

        s_qp = lay._project("self_attn.q_proj", x_q, s_x, M.REG_X, M.REG_Q, ntok)
        s_kp = lay._project("self_attn.k_proj", x_q, s_x, M.REG_X, M.REG_K, ntok)
        s_vp = lay._project("self_attn.v_proj", x_q, s_x, M.REG_X, M.REG_V, ntok)

        qf = read_act(fab, M.REG_Q, M.TOK_STRIDE, ntok, D) * s_qp \
            + w[f"{p}.self_attn.q_proj.bias"]
        kf = read_act(fab, M.REG_K, M.TOK_STRIDE, ntok, D) * s_kp \
            + w[f"{p}.self_attn.k_proj.bias"]
        vf = read_act(fab, M.REG_V, M.TOK_STRIDE, ntok, D) * s_vp \
            + w[f"{p}.self_attn.v_proj.bias"]
        qf = qf * (HD ** -0.5)
        print(f"  Q                cos {cos(qf, tr[f'enc{li}.q']):.4f}")
        print(f"  K                cos {cos(kf, tr[f'enc{li}.k']):.4f}")
        print(f"  V                cos {cos(vf, tr[f'enc{li}.v']):.4f}")

        q_q, s_q = quantise(qf)
        k_q, s_k = quantise(kf)
        v_q, s_v = quantise(vf)
        write_tensor(mir, M.REG_Q, M.TOK_STRIDE, q_q)
        write_tensor(mir, M.REG_K, M.TOK_STRIDE, k_q)
        write_tensor(mir, M.REG_V, M.TOK_STRIDE, v_q)

        mult_score = s_q * s_k * (2.0 ** 12)
        print(f"  score multiplier {mult_score:.4f}"
              + ("   <-- OVER 18-BIT SCALE RANGE" if mult_score >= 3.999 else ""))

        s_attn = lay._attention(ntok, s_q, s_k, s_v)

        # scores as the fabric stored them, back in natural-log units
        sc = np.array([[fab.score[h * M.MAX_TOK * M.SCORE_ROW_STRIDE
                                  + t * M.SCORE_ROW_STRIDE + j]
                        for j in range(ntok)]
                       for h in range(H) for t in range(ntok)],
                      dtype=np.float64) / 4096.0
        ref_sc = tr[f'enc{li}.scores'].reshape(H * ntok, ntok)
        print(f"  scores           cos {cos(sc, ref_sc):.4f}   "
              f"max|s| {np.abs(sc).max():.2f}")

        af = read_act(fab, M.REG_ATTN, M.TOK_STRIDE, ntok, D) * s_attn
        print(f"  attn ctx         cos {cos(af, tr[f'enc{li}.ctx']):.4f}")

        s_o = lay._project(
            "self_attn.out_proj",
            read_act(fab, M.REG_ATTN, M.TOK_STRIDE, ntok, D).astype(np.int64),
            s_attn, M.REG_ATTN, M.REG_OUT, ntok)
        of = read_act(fab, M.REG_OUT, M.TOK_STRIDE, ntok, D) * s_o \
            + w[f"{p}.self_attn.out_proj.bias"]
        print(f"  attn out         cos {cos(of, tr[f'enc{li}.attn_out']):.4f}")

        x1 = FP.layer_norm(x + of, w[f"{p}.self_attn_layer_norm.weight"],
                           w[f"{p}.self_attn_layer_norm.bias"])
        print(f"  post-attn LN     cos {cos(x1, tr[f'enc{li}.post_attn_ln']):.4f}")

        x1_q, s_x1 = quantise(x1)
        write_tensor(mir, M.REG_X, M.TOK_STRIDE, x1_q)
        s_h = lay._ffn1(x1_q, s_x1, ntok)
        hf = read_act(fab, M.REG_FFN_H, F, ntok, F) * s_h + w[f"{p}.fc1.bias"]
        hf = FP.silu(hf)
        print(f"  ffn hidden       cos {cos(hf, tr[f'enc{li}.ffn_hidden']):.4f}")

        h_q, s_h2 = quantise(hf)
        write_tensor(mir, M.REG_FFN_H, F, h_q)
        s_y = lay._ffn2(h_q, s_h2, ntok)
        yf = read_act(fab, M.REG_OUT, M.TOK_STRIDE, ntok, D) * s_y \
            + w[f"{p}.fc2.bias"]
        print(f"  ffn out          cos {cos(yf, tr[f'enc{li}.ffn_out']):.4f}")

        x = FP.layer_norm(x1 + yf, w[f"{p}.final_layer_norm.weight"],
                          w[f"{p}.final_layer_norm.bias"])
        print(f"  layer output     cos {cos(x, tr[f'enc{li}.out']):.4f}")

    mir.close()
    print(f"\nfinal encoder     cos {cos(x, ref):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
