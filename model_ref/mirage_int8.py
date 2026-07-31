"""
mirage_int8.py -- a bit-exact model of the Mirage v2 datapath, and the PS
driver that schedules jobs onto it.

TWO THINGS LIVE HERE AND THEY MUST NOT BE CONFUSED

  Fabric  models what the RTL DOES. Every operation is integer, in the same
          order and with the same truncations as the VHDL. It is the oracle
          the GHDL run is compared against, so if it "helpfully" rounds
          where the hardware truncates, the comparison proves nothing.

  Script  records the same job sequence as a command file for
          sim/encoder_layer_tb. Fabric and Script are driven from the SAME
          calls, so the RTL and the model provably see identical stimulus.

WHERE THE TRUNCATIONS ARE (gemm_sequencer, stages 2-5)

    wb2_prod = acc * scale                       51 bits
    wb3_nar  = resize(wb2_prod >> 16, 32)        truncates to ACC_W
    wb4_sh   = wb3_nar >> rq_shift               arithmetic, floor
    out_data = sat8(wb4_sh)                      saturating
    out_wide = resize(wb4_sh, 21)                WRAPS, does not saturate

`>>` on a negative number floors in both VHDL's shift_right(signed) and
Python's `>>`, which is the one place the two languages agree for free.
"""

import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "..", "rtl_v2")

# ---- geometry, mirrored from rtl_v2/pkg/mirage_v2_pkg.vhd -----------------
MODEL_DIM = 512
NUM_HEADS = 8
HEAD_DIM = 64
FFN_DIM = 2048
MAX_TOK = 16
N_LANES = 32
W_BANKS = 8
BANK_BYTES = 4
BANK_DEPTH = 1024
ACT_AW = 16
SCORE_W = 21
SCORE_AW = 12
SCORE_ROW_STRIDE = 32
ACC_W = 32
RECIP_FRAC = 16
EXP_ROM_AW = 11
EXP_ROM_SH = 4
MAG_MAX = 2 ** (EXP_ROM_AW + EXP_ROM_SH) - 1
INT8_MAX = 127
TOK_STRIDE = MODEL_DIM

REG_X = 0x0000
REG_Q = 0x2000
REG_K = 0x4000
REG_V = 0x6000
REG_ATTN = 0x8000
REG_OUT = 0xA000
REG_FFN_H = 0xC000
REG_EXP = 0xFF00
FFN_TOK_MAX = 7

JOB_GEMM, JOB_STAGE, JOB_SMAX = 0, 1, 2

# config register indices
CFG_KLEN, CFG_FLAGS, CFG_ABASE, CFG_OBASE = 0, 1, 2, 3
CFG_STBASE, CFG_STGEO, CFG_STSTRIDE, CFG_SM = 4, 5, 6, 7
CFG_BUFSEL, CFG_SCWBASE, CFG_SCALE = 8, 9, 0xA


# --------------------------------------------------------------------------
# ROM contents, read from the generated VHDL rather than recomputed.
#
# Recomputing them in Python would mean two independent implementations of
# the same formula, and the whole point of this file is to be the reference
# for the hardware. Parsing the actual table removes that class of drift.
# --------------------------------------------------------------------------
def _load_rom(path):
    txt = open(path, encoding="utf-8").read()
    body = txt[txt.index("architecture"):]
    return [int(m, 2) for m in re.findall(r'"([01]{8,})"', body)]


EXP_ROM = _load_rom(os.path.join(RTL, "core", "exp_rom.vhd"))
RECIP_ROM = _load_rom(os.path.join(RTL, "core", "recip_rom.vhd"))
assert len(EXP_ROM) == 2048, len(EXP_ROM)
assert len(RECIP_ROM) == 2048, len(RECIP_ROM)


# --------------------------------------------------------------------------
def sat8(v):
    return 127 if v > 127 else (-128 if v < -128 else v)


def wrap(v, bits):
    """Two's-complement truncation, which is what VHDL's resize() does when
    it narrows a signed value."""
    m = 1 << bits
    v &= m - 1
    return v - m if v >= (m >> 1) else v


def split_scale(mult):
    """Represent a real multiplier as (scale, rq_shift) with
    mult ~= scale / 2**(16 + rq_shift), scale kept in [2^16, 2^17) so the
    18-bit field carries its full precision."""
    assert mult > 0, mult
    rq = 0
    while mult * (1 << (16 + rq)) < (1 << 16) and rq < 63:
        rq += 1
    scale = int(round(mult * (1 << (16 + rq))))
    if scale > (1 << 18) - 1:
        raise ValueError(f"multiplier {mult} too large for an 18-bit scale")
    return max(1, scale), rq


# ==========================================================================
class Fabric:
    """Bit-exact model of encoder_layer's state and job semantics."""

    def __init__(self):
        self.act = [0] * (1 << ACT_AW)          # int8, signed
        self.score = [0] * (1 << SCORE_AW)      # int21, signed
        self.win = [[0] * N_LANES for _ in range(BANK_DEPTH)]
        self.acc = [0] * N_LANES                # survives between jobs
        self.recip = 1 << RECIP_FRAC
        self.exp_row = []

    # -- weight window load (models the DDR stream) ----------------------
    def load_row(self, row, lanes):
        self.win[row] = list(lanes)

    def load_block(self, arr):
        """arr is (N, N_LANES); rows 0..N-1 of the window."""
        for r in range(arr.shape[0]):
            self.win[r] = arr[r]

    # -- JOB_STAGE -------------------------------------------------------
    def job_stage(self, base, rows, lanes, stride, xpose, row_ofs):
        for r in range(rows):
            out = []
            for lane in range(N_LANES):
                if lane < lanes:
                    addr = base + (lane * stride + r if xpose else r * stride + lane)
                    out.append(self.act[addr & ((1 << ACT_AW) - 1)])
                else:
                    out.append(0)               # zero fill, never stale
            self.win[row_ofs + r] = out

    # -- JOB_GEMM --------------------------------------------------------
    def job_gemm(self, k_len, n_tok, n_coltile, act_base, out_base,
                 scale, rq_shift, acc_clr, acc_wb, wb_score, sc_wbase):
        import numpy as np
        out_stride = n_coltile * N_LANES
        for ct in range(n_coltile):
            w_tile = ct * k_len
            # int64 keeps this EXACT: |acc| <= k_len * 128 * 128 < 2^31 for
            # every k_len this design issues, so vectorising costs no
            # fidelity -- it is the same integer sum, not a float dot.
            wmat = np.array(self.win[w_tile:w_tile + k_len], dtype=np.int64)
            for t in range(n_tok):
                act_row = act_base + t * k_len
                avec = np.array(
                    self.act[act_row:act_row + k_len], dtype=np.int64)
                part = avec @ wmat
                for j in range(N_LANES):
                    self.acc[j] = int(part[j]) + (0 if acc_clr else self.acc[j])
                if acc_wb:
                    out_row = out_base + t * out_stride
                    for j in range(N_LANES):
                        prod = self.acc[j] * scale
                        nar = wrap(prod >> RECIP_FRAC, ACC_W)
                        sh = nar >> rq_shift
                        addr = out_row + ct * N_LANES + j
                        if wb_score:
                            self.score[(sc_wbase + addr) & ((1 << SCORE_AW) - 1)] = \
                                wrap(sh, SCORE_W)
                        else:
                            self.act[addr & ((1 << ACT_AW) - 1)] = sat8(sh)

    # -- JOB_SMAX --------------------------------------------------------
    def job_smax(self, sm_base, sm_len):
        row = [self.score[sm_base + i] for i in range(sm_len)]
        mx = max(row)
        e = []
        for s in row:
            mag = mx - s
            if mag < 0:
                idx = 0
            elif mag > MAG_MAX:
                idx = (1 << EXP_ROM_AW) - 1
            else:
                idx = mag >> EXP_ROM_SH
            eq = EXP_ROM[idx]
            e.append(min(INT8_MAX, (eq * INT8_MAX) >> 16))
        for i, v in enumerate(e):
            self.act[REG_EXP + i] = v
        self.exp_row = e
        self.recip = RECIP_ROM[sum(e)]


# ==========================================================================
class Script:
    """Emits the command file consumed by sim/encoder_layer_tb."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.jobs = 0

    def cfg(self, addr, data):
        self.f.write(f"CFG {addr:x} {data & 0xFFFFFFFF:08x}\n")

    def act(self, addr, val):
        self.f.write(f"ACT {addr:04x} {val & 0xFF:02x}\n")

    def wrow(self, row, lanes):
        words = []
        for b in range(W_BANKS):
            w = 0
            for i in range(BANK_BYTES):
                w |= (lanes[b * BANK_BYTES + i] & 0xFF) << (8 * i)
            words.append(f"{w:08x}")
        self.f.write(f"WROW {row:03x} " + " ".join(words) + "\n")

    def wrows(self, arr):
        """Bulk form of wrow for an (N, N_LANES) integer array.

        A layer emits ~131,000 weight rows. Packing them one lane at a time
        in Python made script generation slower than the GHDL run it feeds.
        """
        import numpy as np
        a = np.asarray(arr, dtype=np.int64).reshape(-1, W_BANKS, BANK_BYTES) & 0xFF
        w = (a[:, :, 0] | (a[:, :, 1] << 8)
             | (a[:, :, 2] << 16) | (a[:, :, 3] << 24))
        fmt = "WROW %x " + " ".join(["%08x"] * W_BANKS) + "\n"
        self.f.writelines(fmt % ((r,) + tuple(row)) for r, row in enumerate(w))

    def job(self, op):
        self.f.write(f"JOB {op:x}\n")
        self.jobs += 1

    def dump(self, addr, count):
        self.f.write(f"DUMP {addr:04x} {count}\n")

    def dumps(self, addr, count):
        self.f.write(f"DUMPS {addr:04x} {count}\n")

    def mark(self, text):
        self.f.write(f"MARK {text}\n")

    def close(self):
        self.f.write("END\n")
        self.f.close()


# ==========================================================================
class Mirage:
    """The PS: drives Fabric and Script together so they cannot diverge."""

    def __init__(self, script_path):
        self.fab = Fabric()
        self.scr = Script(script_path)
        self.expect = []      # (label, [values]) in DUMP order

    # -- primitive writes -------------------------------------------------
    def write_act(self, addr, values):
        for i, v in enumerate(values):
            v = int(v)
            assert -128 <= v <= 127, v
            self.fab.act[addr + i] = v
            self.scr.act(addr + i, v)

    def load_window(self, rows):
        """rows[r] is a list of N_LANES int8 for window row r.

        weight_window writes to the buffer NOT selected by rd_buf, so a
        serial caller -- load then compute from the same buffer -- must flip
        bufsel around the load. Buffer A is filled with bufsel=1 and read
        with bufsel=0.

        This still serialises the load against the compute. Real ping-pong,
        filling B while computing from A and then flipping, is what the
        corrected hardware permits and what a PS driver should do; it needs
        to know which half of the layer it is on, so it belongs in the
        scheduling layer rather than here.
        """
        import numpy as np
        arr = np.asarray(rows, dtype=np.int64)
        self.fab.load_block(arr)
        self.scr.cfg(CFG_BUFSEL, 1)     # writes land in buffer A
        self.scr.wrows(arr)
        self.scr.cfg(CFG_BUFSEL, 0)     # compute reads buffer A

    # -- jobs -------------------------------------------------------------
    def gemm(self, k_len, act_base, out_base, mult=None, rq_shift=None,
             n_tok=1, n_coltile=1, acc_clr=True, acc_wb=True,
             use_recip=False, wb_score=False, sc_wbase=0):
        # Two ways to set the output scale, and they are not interchangeable:
        #   use_recip=False  the PS supplies a real multiplier and it is
        #                    split into an 18-bit scale and a shift
        #   use_recip=True   the scale comes from recip_rom[sum8] inside the
        #                    fabric, so only the shift is ours to choose
        if use_recip:
            assert rq_shift is not None, "xV needs an explicit rq_shift"
            scale, rq = None, rq_shift
            eff_scale = self.fab.recip
        else:
            assert mult is not None, "a plain GEMM needs a multiplier"
            scale, rq = split_scale(mult)
            eff_scale = scale

        flags = ((n_tok & 0x1F)
                 | ((n_coltile & 0x1F) << 8)
                 | ((rq & 0x3F) << 16)
                 | ((1 if acc_clr else 0) << 24)
                 | ((1 if acc_wb else 0) << 25)
                 | ((1 if wb_score else 0) << 26)
                 | ((1 if use_recip else 0) << 27))
        self.scr.cfg(CFG_KLEN, k_len)
        self.scr.cfg(CFG_FLAGS, flags)
        self.scr.cfg(CFG_ABASE, act_base)
        self.scr.cfg(CFG_OBASE, out_base)
        if not use_recip:
            self.scr.cfg(CFG_SCALE, scale)
        if wb_score:
            self.scr.cfg(CFG_SCWBASE, sc_wbase)
        self.scr.job(JOB_GEMM)

        self.fab.job_gemm(k_len, n_tok, n_coltile, act_base, out_base,
                          eff_scale, rq, acc_clr, acc_wb, wb_score, sc_wbase)

    def stage(self, base, rows, lanes, stride, xpose, row_ofs=0):
        """Stage K^T or V from activation memory into the weight window.

        This is the OTHER writer to the window -- encoder_layer muxes
        stage_engine and the DDR stream onto one write port with
        `stage_busy` -- so it is subject to the same rule: writes go to the
        buffer NOT selected by bufsel. Staging into A therefore runs with
        bufsel=1, and the GEMM that consumes it reads A with bufsel=0.

        A staged window and the job that consumes it are strictly
        sequential, so this pair can never overlap. The ping-pong the
        corrected hardware allows is for the DDR stream, not for staging.
        """
        geo = ((rows & 0x7F) | ((lanes & 0x7F) << 8)
               | ((1 if xpose else 0) << 16) | ((row_ofs & 0x3FF) << 18))
        self.scr.cfg(CFG_STBASE, base)
        self.scr.cfg(CFG_STGEO, geo)
        self.scr.cfg(CFG_STSTRIDE, stride)
        self.scr.cfg(CFG_BUFSEL, 1)     # stage writes land in buffer A
        self.scr.job(JOB_STAGE)
        self.scr.cfg(CFG_BUFSEL, 0)     # compute reads buffer A
        self.fab.job_stage(base, rows, lanes, stride, xpose, row_ofs)

    def smax(self, sm_base, sm_len):
        self.scr.cfg(CFG_SM, (sm_base & 0xFFF) | ((sm_len & 0x1F) << 16))
        self.scr.job(JOB_SMAX)
        self.fab.job_smax(sm_base, sm_len)

    # -- readback ---------------------------------------------------------
    def check_act(self, label, addr, count):
        self.scr.mark(label)
        self.scr.dump(addr, count)
        self.expect.append((label, [self.fab.act[addr + i] for i in range(count)]))

    def check_score(self, label, addr, count):
        self.scr.mark(label)
        self.scr.dumps(addr, count)
        self.expect.append((label, [self.fab.score[addr + i] for i in range(count)]))

    def close(self):
        self.scr.close()


# ==========================================================================
def compare(expect, result_path, tol=0):
    """Compare the GHDL result file against the model's expectations.

    Returns (n_checked, [failures]). A failure is
    (label, index, want, got).
    """
    lines = [l.strip() for l in open(result_path, encoding="utf-8")
             if l.strip()]
    got = {}
    label = None
    for l in lines:
        if l.startswith("MARK "):
            label = l[5:].strip()
            got[label] = []
        elif l.startswith("DONE"):
            break
        elif label is not None:
            got[label].append(int(l))

    fails = []
    n = 0
    for label, want in expect:
        have = got.get(label)
        if have is None:
            fails.append((label, -1, "block missing from GHDL output", None))
            continue
        if len(have) != len(want):
            fails.append((label, -1, f"{len(want)} values", f"{len(have)}"))
            continue
        for i, (w, g) in enumerate(zip(want, have)):
            n += 1
            if abs(w - g) > tol:
                fails.append((label, i, w, g))
    return n, fails
