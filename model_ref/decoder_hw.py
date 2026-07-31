"""
decoder_hw.py -- a MarianMT decoder layer on the SAME fabric.

THE REUSE CLAIM, TESTED

The encoder layer is half the model. The design's whole premise is that the
decoder reuses the same compute path, and this file is what makes that
checkable rather than asserted. It adds NO new job type, NO new config
register and NO RTL change. A decoder layer is three sublayers, and each one
decomposes into jobs the fabric already has:

  self-attention   identical to the encoder's, except that query row t may
                   only see keys 0..t. That mask is free: the softmax job
                   already takes `sm_len`, so issuing sm_len = t+1 IS the
                   causal mask. No -inf, no compare, no extra hardware.

  cross-attention  Q comes from the decoder state, K and V from the encoder
                   output. Same projections, same staging, same softmax --
                   only the base addresses differ. K and V are computed once
                   per encoder output rather than per decoding step.

  feed-forward     byte for byte the encoder's, including the 2048-wide
                   reduction split into two chunks.

So the answer to "can we reuse it" is yes, and the reason is that the fabric
was given a job interface instead of a hardwired op sequence -- the ordering
that makes a decoder a decoder lives entirely on the PS.

MEMORY MAP WHILE DECODING

  REG_X      decoder state for the current step   (T_dec x 512)
  REG_Q/K/V  self-attention projections, then reused for cross-attention
  REG_FFN_H  doubles as scratch for the encoder output during cross-attn,
             because the FFN has not run yet at that point
"""

import numpy as np

import mirage_int8 as M
from encoder_hw import quantise, window_rows

D, H, HD, F = M.MODEL_DIM, M.NUM_HEADS, M.HEAD_DIM, M.FFN_DIM


class DecoderLayerHW:
    def __init__(self, mir, w, li):
        self.m = mir
        self.w = w
        self.p = f"model.decoder.layers.{li}"

    # -- shared with the encoder in everything but the weight name --------
    def _project(self, name, x_q, s_x, act_base, out_base, ntok):
        W = self.w[f"{self.p}.{name}.weight"]
        Wq, s_w = quantise(W)
        Wt = Wq.T
        acc = x_q @ Wt
        mult = 127.0 / max(1, int(np.abs(acc).max()))
        for ct in range(D // M.N_LANES):
            self.m.load_window(window_rows(Wt, ct, 0, D))
            for t in range(ntok):
                self.m.gemm(k_len=D,
                            act_base=act_base + t * M.TOK_STRIDE,
                            out_base=out_base + t * M.TOK_STRIDE + ct * M.N_LANES,
                            mult=mult)
        return s_x * s_w / mult

    def _attend(self, n_q, n_kv, s_q, s_k, s_v, causal):
        """Scores, softmax and xV. `causal` truncates each row's key range,
        which is the entire cost of masking on this design."""
        mult_score = s_q * s_k * (2.0 ** 12)
        for h in range(H):
            self.m.stage(base=M.REG_K + h * HD, rows=HD, lanes=n_kv,
                         stride=M.TOK_STRIDE, xpose=True, row_ofs=0)
            for t in range(n_q):
                self.m.gemm(k_len=HD,
                            act_base=M.REG_Q + t * M.TOK_STRIDE + h * HD,
                            out_base=t * M.SCORE_ROW_STRIDE,
                            mult=mult_score, wb_score=True,
                            sc_wbase=h * M.MAX_TOK * M.SCORE_ROW_STRIDE)

            for t in range(n_q):
                # A causal row attends to keys 0..t; a cross-attention row
                # attends to every source token.
                klen = (t + 1) if causal else n_kv
                for ct in range(HD // M.N_LANES):
                    self.m.stage(base=M.REG_V + h * HD + ct * M.N_LANES,
                                 rows=klen, lanes=M.N_LANES,
                                 stride=M.TOK_STRIDE, xpose=False,
                                 row_ofs=ct * klen)
                self.m.smax(
                    sm_base=h * M.MAX_TOK * M.SCORE_ROW_STRIDE
                            + t * M.SCORE_ROW_STRIDE,
                    sm_len=klen)
                self.m.gemm(k_len=klen, act_base=M.REG_EXP,
                            out_base=M.REG_ATTN + t * M.TOK_STRIDE + h * HD,
                            rq_shift=7, use_recip=True,
                            n_coltile=HD // M.N_LANES)
        return s_v * 128.0 / 127.0

    def _ffn(self, x_q, s_x, ntok):
        W1 = self.w[f"{self.p}.fc1.weight"]
        W1q, s_w1 = quantise(W1)
        W1t = W1q.T
        mult1 = 127.0 / max(1, int(np.abs(x_q @ W1t).max()))
        for ct in range(F // M.N_LANES):
            self.m.load_window(window_rows(W1t, ct, 0, D))
            for t in range(ntok):
                self.m.gemm(k_len=D, act_base=M.REG_X + t * M.TOK_STRIDE,
                            out_base=M.REG_FFN_H + t * F + ct * M.N_LANES,
                            mult=mult1)
        return s_x * s_w1 / mult1

    def _ffn2(self, h_q, s_h, ntok):
        W = self.w[f"{self.p}.fc2.weight"]
        Wq, s_w = quantise(W)
        Wt = Wq.T
        mult = 127.0 / max(1, int(np.abs(h_q @ Wt).max()))
        CH = M.BANK_DEPTH
        for ct in range(D // M.N_LANES):
            for t in range(ntok):
                for c in range(F // CH):
                    self.m.load_window(window_rows(Wt, ct, c * CH, CH))
                    self.m.gemm(
                        k_len=CH, act_base=M.REG_FFN_H + t * F + c * CH,
                        out_base=M.REG_OUT + t * M.TOK_STRIDE + ct * M.N_LANES,
                        mult=mult, acc_clr=(c == 0),
                        acc_wb=(c == F // CH - 1))
        return s_h * s_w / mult
