"""
encoder_hw.py -- one MarianMT encoder layer, split between PL and PS.

THE SPLIT, AND WHY IT IS WHERE IT IS

    PL (fabric)   every O(D^2) operation: the six projections, QK^T, the
                  softmax, and xV. ~4.7 M multiply-accumulates per layer.
    PS (here)     every O(D) operation: bias, the FFN activation function,
                  residual add, layer normalisation, and choosing the
                  quantisation scales. ~9 K operations per layer.

That is 0.2% of the arithmetic on the PS side, and it is deliberately the
0.2% where int8 hurts most. LayerNorm needs a mean, a variance and a
reciprocal square root over the feature axis; doing that in int8 costs
accuracy out of proportion to the silicon it saves, and the PS has to touch
every tensor at the layer boundary anyway -- it is already streaming the
weights in. The round trip is ~20 us of DMA against ~5 ms of fabric compute
per layer, so it is not on the critical path in any meaningful sense.

QUANTISATION

Per-tensor symmetric int8 throughout. For a matmul with activation scale
s_a and weight scale s_w, the fabric accumulates the exact integer product
and then applies

    out = sat8( ((acc * scale) >> 16) >> rq_shift )

so the output represents the true value with scale s_a * s_w / M, where
M = scale * 2^-(16+rq_shift). The PS picks M to put the result at full
int8 range and remembers the resulting scale. No calibration data is
needed because the PS computes the actual tensor first and reads the true
maximum off it -- this is a verification harness, not a deployment path.

THE ONE SCALE THAT IS NOT FREE

Score memory feeds exp_rom, which is indexed as (max - s) >> EXP_ROM_SH
over a table of exp(-i/256). A score LSB therefore MEANS 2^-12 in
natural-log units. The QK^T multiplier is chosen to make that true rather
than to maximise range; get it wrong and the attention silently runs at the
wrong temperature.
"""

import numpy as np

import mirage_int8 as M

D = M.MODEL_DIM
H = M.NUM_HEADS
HD = M.HEAD_DIM
F = M.FFN_DIM


def quantise(x, bits=8):
    """Symmetric per-tensor int8. Returns (int array, scale)."""
    amax = float(np.abs(x).max())
    if amax == 0.0:
        return np.zeros(x.shape, dtype=np.int64), 1.0
    s = amax / (2 ** (bits - 1) - 1)
    q = np.clip(np.rint(x / s), -128, 127).astype(np.int64)
    return q, s


def window_rows(wq_t, ct, k0, k_len):
    """Weight-window rows for column tile `ct`, reduction rows k0..k0+k_len.

    wq_t is the weight already transposed to (in_features, out_features),
    which is the orientation the MAC array reads: window row k, lane j
    carries the weight connecting input k to output ct*32 + j.
    """
    blk = wq_t[k0:k0 + k_len, ct * M.N_LANES:(ct + 1) * M.N_LANES]
    return [[int(v) for v in row] for row in blk]


class EncoderLayerHW:
    """Runs one encoder layer, emitting jobs into a Mirage driver."""

    def __init__(self, mir, w, li):
        self.m = mir
        self.w = w
        self.p = f"model.encoder.layers.{li}"
        self.li = li

    # -- one projection: x_q (T,D) -> (T,D), all 16 column tiles ----------
    def _project(self, name, x_q, s_x, act_base, out_base, ntok):
        W = self.w[f"{self.p}.{name}.weight"]          # (out, in)
        Wq, s_w = quantise(W)
        Wt = Wq.T                                       # (in, out)

        # The true accumulator maximum, so the output uses full int8 range.
        acc = x_q @ Wt
        amax = max(1, int(np.abs(acc).max()))
        mult = 127.0 / amax
        s_out = s_x * s_w / mult

        for ct in range(D // M.N_LANES):
            self.m.load_window(window_rows(Wt, ct, 0, D))
            for t in range(ntok):
                self.m.gemm(k_len=D,
                            act_base=act_base + t * M.TOK_STRIDE,
                            out_base=out_base + t * M.TOK_STRIDE + ct * M.N_LANES,
                            mult=mult)
        return s_out

    # -- attention -------------------------------------------------------
    def _attention(self, ntok, s_q, s_k, s_v):
        """Scores, softmax and xV for every head. Returns the attn scale."""
        # A score LSB must mean 2^-12 in natural-log units; see the header.
        mult_score = s_q * s_k * (2.0 ** 12)
        for h in range(H):
            # K^T for this head: window row = reduction dim, lane = key token
            self.m.stage(base=M.REG_K + h * HD, rows=HD, lanes=ntok,
                         stride=M.TOK_STRIDE, xpose=True, row_ofs=0)
            for t in range(ntok):
                self.m.gemm(k_len=HD,
                            act_base=M.REG_Q + t * M.TOK_STRIDE + h * HD,
                            out_base=t * M.SCORE_ROW_STRIDE,
                            mult=mult_score,
                            wb_score=True,
                            sc_wbase=h * M.MAX_TOK * M.SCORE_ROW_STRIDE)

            # V for this head, both column tiles of HEAD_DIM = 2 x N_LANES
            for ct in range(HD // M.N_LANES):
                self.m.stage(base=M.REG_V + h * HD + ct * M.N_LANES,
                             rows=ntok, lanes=M.N_LANES,
                             stride=M.TOK_STRIDE, xpose=False,
                             row_ofs=ct * ntok)

            for t in range(ntok):
                self.m.smax(
                    sm_base=h * M.MAX_TOK * M.SCORE_ROW_STRIDE
                            + t * M.SCORE_ROW_STRIDE,
                    sm_len=ntok)
                self.m.gemm(k_len=ntok, act_base=M.REG_EXP,
                            out_base=M.REG_ATTN + t * M.TOK_STRIDE + h * HD,
                            rq_shift=7, use_recip=True,
                            n_coltile=HD // M.N_LANES)
        # recip is 127*2^16/sum8 and the shift is 2^7, so the weighted mean
        # of V comes back scaled by 127/128.
        return s_v * 128.0 / 127.0

    # -- FFN -------------------------------------------------------------
    def _ffn1(self, x_q, s_x, ntok):
        W = self.w[f"{self.p}.fc1.weight"]              # (2048, 512)
        Wq, s_w = quantise(W)
        Wt = Wq.T                                        # (512, 2048)
        acc = x_q @ Wt
        amax = max(1, int(np.abs(acc).max()))
        mult = 127.0 / amax
        for ct in range(F // M.N_LANES):                 # 64 tiles
            self.m.load_window(window_rows(Wt, ct, 0, D))
            for t in range(ntok):
                self.m.gemm(k_len=D,
                            act_base=M.REG_X + t * M.TOK_STRIDE,
                            out_base=M.REG_FFN_H + t * F + ct * M.N_LANES,
                            mult=mult)
        return s_x * s_w / mult

    def _ffn2(self, h_q, s_h, ntok):
        W = self.w[f"{self.p}.fc2.weight"]              # (512, 2048)
        Wq, s_w = quantise(W)
        Wt = Wq.T                                        # (2048, 512)
        acc = h_q @ Wt
        amax = max(1, int(np.abs(acc).max()))
        mult = 127.0 / amax

        # FFN_DIM = 2048 exceeds BANK_DEPTH = 1024, so the reduction is split
        # in two chunks with the accumulator carried across them. Only the
        # second chunk writes back.
        CH = M.BANK_DEPTH
        nch = F // CH
        for ct in range(D // M.N_LANES):
            for t in range(ntok):
                for c in range(nch):
                    self.m.load_window(window_rows(Wt, ct, c * CH, CH))
                    self.m.gemm(
                        k_len=CH,
                        act_base=M.REG_FFN_H + t * F + c * CH,
                        out_base=M.REG_OUT + t * M.TOK_STRIDE + ct * M.N_LANES,
                        mult=mult,
                        acc_clr=(c == 0), acc_wb=(c == nch - 1))
        return s_h * s_w / mult
