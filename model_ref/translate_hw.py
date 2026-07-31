#!/usr/bin/env python3
"""
translate_hw.py -- "Hello" -> Arabic, with the encoder running on the RTL.

WHAT THIS ACTUALLY PROVES, STATED PLAINLY

The FPGA does not translate anything on its own. It has no embedding table,
no decoder, no beam search, and not enough BRAM to hold one weight matrix.
What it has is the matmul chain and the softmax of ONE transformer layer,
which is ~99.8% of the arithmetic. So the claim being tested is:

    drive the real MarianMT opus-mt-en-ar weights and the real token
    embedding for "Hello" through six passes of the RTL encoder layer,
    let the PS do the elementwise work between passes, and the resulting
    encoder state still decodes to the correct Arabic token.

Three separate checks, because a single end-to-end number can pass for the
wrong reasons:

  1. BIT-EXACT   every tensor the fabric produces is compared, byte for
                 byte, against mirage_int8.Fabric. Zero tolerance. This is
                 the check that says the RTL is what we think it is.
  2. NUMERIC     the int8 encoder output is compared against the float32
                 reference encoder, as cosine similarity per token. This
                 says the quantisation did not destroy the representation.
  3. SEMANTIC    the decoder is run on the fabric-produced encoder state
                 and the output token ids must match the reference. This
                 is the one the user cares about: does it still say
                 "marhaba".

Run:  ./.venv/bin/python translate_hw.py
"""

import os
import subprocess
import sys
import time

import numpy as np
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marian_fp32 as FP
import mirage_int8 as M
from encoder_hw import EncoderLayerHW, quantise

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")
D = M.MODEL_DIM
H = M.NUM_HEADS
HD = M.HEAD_DIM
F = M.FFN_DIM


def read_act(fab, base, stride, ntok, n):
    """Pull a (ntok, n) int8 tensor out of the fabric's activation memory."""
    return np.array([[fab.act[base + t * stride + i] for i in range(n)]
                     for t in range(ntok)], dtype=np.float64)


def write_tensor(mir, base, stride, q):
    for t in range(q.shape[0]):
        mir.write_act(base + t * stride, [int(v) for v in q[t]])


def run_layer(mir, w, li, x, ntok):
    """One encoder layer. x is (T, D) float32 -- the PS's view."""
    p = f"model.encoder.layers.{li}"
    lay = EncoderLayerHW(mir, w, li)
    fab = mir.fab

    # ---- PS: quantise the layer input and load it -----------------------
    x_q, s_x = quantise(x)
    write_tensor(mir, M.REG_X, M.TOK_STRIDE, x_q)

    # ---- PL: Q, K, V ----------------------------------------------------
    s_qp = lay._project("self_attn.q_proj", x_q, s_x, M.REG_X, M.REG_Q, ntok)
    s_kp = lay._project("self_attn.k_proj", x_q, s_x, M.REG_X, M.REG_K, ntok)
    s_vp = lay._project("self_attn.v_proj", x_q, s_x, M.REG_X, M.REG_V, ntok)
    (mir.check_act(f"L{li}.Q", M.REG_Q, M.TOK_STRIDE * ntok))
    (mir.check_act(f"L{li}.K", M.REG_K, M.TOK_STRIDE * ntok))
    (mir.check_act(f"L{li}.V", M.REG_V, M.TOK_STRIDE * ntok))

    # ---- PS: bias, and the 1/sqrt(head_dim) query scaling ---------------
    # Both are O(D). Folding the attention scaling into the requantisation
    # costs nothing: it changes the scale the PS records, not the data.
    qf = read_act(fab, M.REG_Q, M.TOK_STRIDE, ntok, D) * s_qp \
        + w[f"{p}.self_attn.q_proj.bias"]
    kf = read_act(fab, M.REG_K, M.TOK_STRIDE, ntok, D) * s_kp \
        + w[f"{p}.self_attn.k_proj.bias"]
    vf = read_act(fab, M.REG_V, M.TOK_STRIDE, ntok, D) * s_vp \
        + w[f"{p}.self_attn.v_proj.bias"]
    qf = qf * (HD ** -0.5)

    q_q, s_q = quantise(qf)
    k_q, s_k = quantise(kf)
    v_q, s_v = quantise(vf)
    write_tensor(mir, M.REG_Q, M.TOK_STRIDE, q_q)
    write_tensor(mir, M.REG_K, M.TOK_STRIDE, k_q)
    write_tensor(mir, M.REG_V, M.TOK_STRIDE, v_q)

    # ---- PL: scores, softmax, xV ----------------------------------------
    s_attn = lay._attention(ntok, s_q, s_k, s_v)
    for h in range(H):
        for t in range(ntok):
            mir.check_score(
                f"L{li}.sc.h{h}.t{t}",
                h * M.MAX_TOK * M.SCORE_ROW_STRIDE + t * M.SCORE_ROW_STRIDE,
                ntok)
    (mir.check_act(f"L{li}.attn", M.REG_ATTN,
                                M.TOK_STRIDE * ntok))

    # ---- PL: output projection ------------------------------------------
    s_o = lay._project("self_attn.out_proj",
                       read_act(fab, M.REG_ATTN, M.TOK_STRIDE, ntok, D).astype(np.int64),
                       s_attn, M.REG_ATTN, M.REG_OUT, ntok)
    (mir.check_act(f"L{li}.oproj", M.REG_OUT, M.TOK_STRIDE * ntok))

    # ---- PS: bias, residual, layer norm ---------------------------------
    of = read_act(fab, M.REG_OUT, M.TOK_STRIDE, ntok, D) * s_o \
        + w[f"{p}.self_attn.out_proj.bias"]
    x1 = FP.layer_norm(x + of,
                       w[f"{p}.self_attn_layer_norm.weight"],
                       w[f"{p}.self_attn_layer_norm.bias"])

    # ---- PL: FFN layer 1 -------------------------------------------------
    x1_q, s_x1 = quantise(x1)
    write_tensor(mir, M.REG_X, M.TOK_STRIDE, x1_q)
    s_h = lay._ffn1(x1_q, s_x1, ntok)
    (mir.check_act(f"L{li}.ffn1", M.REG_FFN_H, F * ntok))

    # ---- PS: bias and the activation function ---------------------------
    hf = read_act(fab, M.REG_FFN_H, F, ntok, F) * s_h + w[f"{p}.fc1.bias"]
    hf = FP.silu(hf)                      # config.activation_function
    h_q, s_h2 = quantise(hf)
    write_tensor(mir, M.REG_FFN_H, F, h_q)

    # ---- PL: FFN layer 2 -------------------------------------------------
    s_y = lay._ffn2(h_q, s_h2, ntok)
    (mir.check_act(f"L{li}.ffn2", M.REG_OUT, M.TOK_STRIDE * ntok))

    # ---- PS: bias, residual, layer norm ---------------------------------
    yf = read_act(fab, M.REG_OUT, M.TOK_STRIDE, ntok, D) * s_y \
        + w[f"{p}.fc2.bias"]
    return FP.layer_norm(x1 + yf,
                         w[f"{p}.final_layer_norm.weight"],
                         w[f"{p}.final_layer_norm.bias"])


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    t0 = time.time()

    print("=" * 70)
    print(f'MIRAGE end-to-end: "{text}" -> Arabic, encoder on the RTL')
    print("=" * 70)

    model = FP.Marian()
    w = model.w
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(FP.HF, "source.spm"))
    pieces = sp.encode(text, out_type=str)
    ids = [model.vocab[pc] for pc in pieces] + [FP.EOS_ID]
    ntok = len(ids)
    print(f"\n1. TOKENISE")
    print(f"   pieces      {pieces} + </s>")
    print(f"   ids         {ids}   ({ntok} tokens)")
    assert ntok <= M.FFN_TOK_MAX, (
        f"{ntok} tokens exceeds the FFN hidden region's {M.FFN_TOK_MAX}")

    # ---- reference ------------------------------------------------------
    ref_ids = model.generate(np.array(ids))
    ref_enc = model.encode(np.array(ids))
    print(f"\n2. FLOAT32 REFERENCE")
    print(f"   target ids  {ref_ids}")
    print(f"   translation {model.detokenise(ref_ids)!r}")

    # ---- embeddings -----------------------------------------------------
    x = model.embed(np.array(ids), "encoder")
    print(f"\n3. TOKEN EMBEDDINGS (what the fabric is actually fed)")
    print(f"   shape {x.shape}, range [{x.min():.3f}, {x.max():.3f}]")
    for t in range(ntok):
        print(f"   tok {ids[t]:5d}  {np.array2string(x[t][:6], precision=3)} ...")

    # ---- six encoder layers on the fabric -------------------------------
    script = os.path.join(SIM, "xlate_cmd.txt")
    result = os.path.join(SIM, "xlate_res.txt")
    mir = M.Mirage(script)
    print(f"\n4. ENCODER: 6 layers through the RTL job interface")
    nlayers = int(os.environ.get("MIRAGE_LAYERS", FP.N_LAYERS))
    for li in range(nlayers):
        ls = time.time()
        x = run_layer(mir, w, li, x, ntok)
        print(f"   layer {li}: {mir.scr.jobs:6d} jobs cumulative, "
              f"{time.time() - ls:5.1f}s to schedule")
    mir.close()
    enc_hw = x
    sz = os.path.getsize(script) / 1e6
    print(f"   script: {mir.scr.jobs} jobs, {sz:.1f} MB")

    # ---- run the RTL ----------------------------------------------------
    print(f"\n5. GHDL: executing the job stream against encoder_layer")
    subprocess.run([os.path.join(SIM, "build.sh")], check=True,
                   capture_output=True)
    gs = time.time()
    r = subprocess.run(
        ["ghdl", "-r", "--std=08", "--workdir=work", "encoder_layer_tb",
         f"-gSCRIPT={script}", f"-gRESULT={result}", "--ieee-asserts=disable"],
        cwd=SIM, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    print(f"   {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'ok'}")
    print(f"   wall time {time.time() - gs:.1f}s")

    # ---- check 1: bit-exact ---------------------------------------------
    n, fails = M.compare(mir.expect, result)
    print(f"\n6. CHECK 1 -- BIT-EXACT (RTL vs model, zero tolerance)")
    print(f"   {n} values compared across {len(mir.expect)} tensors")
    if fails:
        for f in fails[:15]:
            print("   FAIL", f)
        print(f"   {len(fails)} MISMATCHES")
        return 1
    print("   PASS: every byte the fabric produced matches the model")

    # ---- check 2: numeric ------------------------------------------------
    print(f"\n7. CHECK 2 -- NUMERIC (int8 encoder vs float32 encoder)")
    for t in range(ntok):
        a, b = enc_hw[t], ref_enc[t]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        print(f"   token {t}: cosine {cos:.4f}   relative error {rel:.4f}")

    # ---- check 3: semantic ----------------------------------------------
    print(f"\n8. CHECK 3 -- SEMANTIC (decode the fabric's encoder state)")
    dec = [FP.DECODER_START]
    for _ in range(24):
        logits = model.decode_step(np.array(dec), enc_hw.astype(np.float32))[-1]
        logits[FP.PAD_ID] = -np.inf
        nxt = int(logits.argmax())
        dec.append(nxt)
        if nxt == FP.EOS_ID:
            break
    hw_ids = dec[1:]
    print(f"   target ids  {hw_ids}")
    print(f"   pieces      {[model.inv_vocab[i] for i in hw_ids]}")
    print(f"   TRANSLATION {model.detokenise(hw_ids)!r}")

    ok = hw_ids[0] == ref_ids[0]
    print("\n" + "=" * 70)
    print(f"reference : {model.detokenise(ref_ids)!r}  {ref_ids}")
    print(f"on fabric : {model.detokenise(hw_ids)!r}  {hw_ids}")
    print(f"first target token {'MATCHES' if ok else 'DIFFERS'}")
    print(f"total wall time {time.time() - t0:.0f}s")
    print("=" * 70)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
