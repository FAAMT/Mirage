#!/usr/bin/env python3
"""
test_ops.py -- op-level co-simulation: every job type, RTL against model.

This is the rung of the verification ladder between the VHDL unit
testbenches (which prove one module in isolation) and the full layer
(which proves the whole chain but is slow and hard to bisect). Each case
here exercises one job type through the REAL top level, driven by the real
config register file, and compares every output byte against
mirage_int8.Fabric.

Run:  ./.venv/bin/python test_ops.py
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mirage_int8 as M

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")


def run_ghdl(script, result):
    subprocess.run([os.path.join(SIM, "build.sh")], check=True,
                   capture_output=True)
    r = subprocess.run(
        ["ghdl", "-r", "--std=08", "--workdir=work", "encoder_layer_tb",
         f"-gSCRIPT={script}", f"-gRESULT={result}",
         "--ieee-asserts=disable", "--stop-time=2sec"],
        cwd=SIM, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit("GHDL run failed")
    return r


def main():
    random.seed(20260731)
    script = os.path.join(SIM, "ops_cmd.txt")
    result = os.path.join(SIM, "ops_res.txt")
    m = M.Mirage(script)

    # ---------------------------------------------------------------
    # 1. Plain GEMM: one 32-wide output tile, reduction over 64.
    # ---------------------------------------------------------------
    k = 64
    acts = [random.randint(-128, 127) for _ in range(k)]
    m.write_act(M.REG_X, acts)
    rows = [[random.randint(-128, 127) for _ in range(M.N_LANES)]
            for _ in range(k)]
    m.load_window(rows)
    # worst-case |acc| = 64 * 128 * 128 = 2^20; aim the output at int8
    m.gemm(k_len=k, act_base=M.REG_X, out_base=M.REG_Q, mult=1.0 / 4096)
    m.check_act("gemm_basic", M.REG_Q, M.N_LANES)

    # ---------------------------------------------------------------
    # 2. Multi-token, multi-column-tile GEMM. Exercises out_stride and
    #    the w_tile = ct * k_len stepping together.
    # ---------------------------------------------------------------
    k = 32
    ntok, nct = 3, 2
    m.write_act(0x0100, [random.randint(-128, 127) for _ in range(k * ntok)])
    m.load_window([[random.randint(-128, 127) for _ in range(M.N_LANES)]
                   for _ in range(k * nct)])
    m.gemm(k_len=k, act_base=0x0100, out_base=0x0300, mult=1.0 / 1024,
           n_tok=ntok, n_coltile=nct)
    m.check_act("gemm_tiled", 0x0300, ntok * nct * M.N_LANES)

    # ---------------------------------------------------------------
    # 3. Reduction chunking: the same dot product, split in two jobs
    #    with the accumulator carried across. This is what the FFN
    #    output layer needs, since it reduces over 2048 > BANK_DEPTH.
    # ---------------------------------------------------------------
    k = 128
    a2 = [random.randint(-128, 127) for _ in range(k)]
    w2 = [[random.randint(-128, 127) for _ in range(M.N_LANES)]
          for _ in range(k)]
    m.write_act(0x0500, a2)
    mult = 1.0 / 8192
    m.load_window(w2[:64])
    m.gemm(k_len=64, act_base=0x0500, out_base=0x0600, mult=mult,
           acc_clr=True, acc_wb=False)
    m.load_window(w2[64:])
    m.gemm(k_len=64, act_base=0x0500 + 64, out_base=0x0600, mult=mult,
           acc_clr=False, acc_wb=True)
    m.check_act("gemm_chunked", 0x0600, M.N_LANES)

    # ---------------------------------------------------------------
    # 4. STAGE + GEMM: the QK^T shape. Stage K^T out of activation
    #    memory at the real token stride, then multiply Q by it.
    # ---------------------------------------------------------------
    ntok = 3
    for t in range(ntok):                       # K, token-major
        m.write_act(M.REG_K + t * M.TOK_STRIDE,
                    [random.randint(-128, 127) for _ in range(M.HEAD_DIM)])
    for t in range(ntok):                       # Q
        m.write_act(M.REG_Q + t * M.TOK_STRIDE,
                    [random.randint(-128, 127) for _ in range(M.HEAD_DIM)])
    m.stage(base=M.REG_K, rows=M.HEAD_DIM, lanes=ntok,
            stride=M.TOK_STRIDE, xpose=True, row_ofs=0)
    for t in range(ntok):
        m.gemm(k_len=M.HEAD_DIM, act_base=M.REG_Q + t * M.TOK_STRIDE,
               out_base=t * M.SCORE_ROW_STRIDE, mult=1.0 / 16,
               wb_score=True, sc_wbase=0)
    m.check_score("scores", 0, ntok * M.SCORE_ROW_STRIDE)

    # ---------------------------------------------------------------
    # 5. SMAX on those scores, then the xV matmul that consumes them.
    #    This is the path that could not work at all before: the exp
    #     stream is produced by one job and read back by the next.
    # ---------------------------------------------------------------
    for t in range(ntok):                       # V, token-major
        m.write_act(M.REG_V + t * M.TOK_STRIDE,
                    [random.randint(-128, 127) for _ in range(M.HEAD_DIM)])
    for ct in range(2):
        m.stage(base=M.REG_V + ct * M.N_LANES, rows=ntok, lanes=M.N_LANES,
                stride=M.TOK_STRIDE, xpose=False, row_ofs=ct * ntok)
    for t in range(ntok):
        m.smax(sm_base=t * M.SCORE_ROW_STRIDE, sm_len=ntok)
        m.check_act(f"exp_row{t}", M.REG_EXP, ntok)
        m.gemm(k_len=ntok, act_base=M.REG_EXP,
               out_base=M.REG_ATTN + t * M.TOK_STRIDE,
               rq_shift=7, use_recip=True, n_coltile=2)
    m.check_act("attn", M.REG_ATTN, M.TOK_STRIDE * (ntok - 1) + M.HEAD_DIM)

    m.close()
    print(f"script: {m.scr.jobs} jobs")
    run_ghdl("ops_cmd.txt", "ops_res.txt")

    n, fails = M.compare(m.expect, result)
    print(f"checked {n} values across {len(m.expect)} blocks")
    if fails:
        for f in fails[:25]:
            print("  FAIL", f)
        print(f"{len(fails)} mismatches")
        return 1
    print("PASS: every job type matches the bit-exact model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
