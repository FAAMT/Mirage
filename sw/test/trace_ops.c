/* =====================================================================
 * trace_ops.c -- drive the C driver through model_ref/test_ops.py's
 * exact call sequence and emit the resulting command script
 *
 * test_ops.py is the op-level co-simulation: every job type, driven
 * through the real top level, compared against the bit-exact model. Its
 * script is committed as sim/ops_cmd.txt and passes in GHDL. So if this
 * driver, given the same calls, produces the same script, the driver
 * issues a register sequence that has already been verified against the
 * RTL -- without a board, without Vivado, and without re-running GHDL.
 *
 * The activation and weight VALUES here are zero. They are test data,
 * not driver behaviour, and reproducing Python's Mersenne Twister in C
 * to match them would prove nothing about the driver. sw/test/
 * check_ops.sh therefore compares every line except those payload
 * bytes: all CFG/JOB/MARK/DUMP/DUMPS lines in full, and the ADDRESSES
 * of every ACT and the INDEX of every WROW. What that leaves unchecked
 * is exactly the set of bytes this file made up.
 * ===================================================================== */
#include <stdio.h>
#include <string.h>

#include "mirage.h"
#include "mirage_io.h"
#include "mirage_trace.h"

/* Address map constants, from model_ref/mirage_int8.py */
#define REG_X          0x0000u
#define REG_Q          0x2000u
#define REG_K          0x4000u
#define REG_V          0x6000u
#define REG_ATTN       0x8000u
#define REG_EXP        0xFF00u
#define TOK_STRIDE     512u
#define HEAD_DIM       64u
#define SCORE_ROW_STR  32u

static int8_t  acts[4096];
static int8_t  window[MIRAGE_BANK_DEPTH * MIRAGE_N_LANES];
static int8_t  rdbuf[4096];
static int32_t scbuf[512];

/* check_act(): the reference marks the block, then dumps it. */
static void check_act(const char *label, uint32_t addr, size_t count)
{
    mirage_mark(label);
    mirage_read_act(addr, rdbuf, count);
}

static void check_score(const char *label, uint32_t addr, size_t count)
{
    mirage_mark(label);
    mirage_read_scores(addr, scbuf, count);
}

static int fail(const char *what, int rc)
{
    fprintf(stderr, "trace_ops: %s returned %d\n", what, rc);
    return 1;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "ops_trace.txt";
    mirage_gemm_t g;
    uint32_t t, ct;
    int rc;

    if (mirage_trace_open(path) != 0) {
        fprintf(stderr, "trace_ops: cannot write %s\n", path);
        return 1;
    }

    memset(acts,   0, sizeof acts);
    memset(window, 0, sizeof window);

    /* -----------------------------------------------------------------
     * 1. Plain GEMM: one 32-wide output tile, reduction over 64.
     * ----------------------------------------------------------------- */
    mirage_write_act(REG_X, acts, 64);
    mirage_load_window(window, 64);

    g = (mirage_gemm_t)MIRAGE_GEMM_DEFAULTS;
    g.k_len = 64; g.act_base = REG_X; g.out_base = REG_Q;
    g.mult = 1.0 / 4096.0;
    if ((rc = mirage_gemm(&g)) != MIRAGE_OK) return fail("gemm_basic", rc);
    check_act("gemm_basic", REG_Q, MIRAGE_N_LANES);

    /* -----------------------------------------------------------------
     * 2. Multi-token, multi-column-tile GEMM.
     * ----------------------------------------------------------------- */
    mirage_write_act(0x0100u, acts, 32 * 3);
    mirage_load_window(window, 32 * 2);

    g = (mirage_gemm_t)MIRAGE_GEMM_DEFAULTS;
    g.k_len = 32; g.act_base = 0x0100u; g.out_base = 0x0300u;
    g.mult = 1.0 / 1024.0; g.n_tok = 3; g.n_coltile = 2;
    if ((rc = mirage_gemm(&g)) != MIRAGE_OK) return fail("gemm_tiled", rc);
    check_act("gemm_tiled", 0x0300u, 3 * 2 * MIRAGE_N_LANES);

    /* -----------------------------------------------------------------
     * 3. Reduction chunking: one dot product split across two jobs with
     *    the accumulator carried between them.
     * ----------------------------------------------------------------- */
    mirage_write_act(0x0500u, acts, 128);

    mirage_load_window(window, 64);
    g = (mirage_gemm_t)MIRAGE_GEMM_DEFAULTS;
    g.k_len = 64; g.act_base = 0x0500u; g.out_base = 0x0600u;
    g.mult = 1.0 / 8192.0; g.acc_clr = 1; g.acc_wb = 0;
    if ((rc = mirage_gemm(&g)) != MIRAGE_OK) return fail("chunk lo", rc);

    mirage_load_window(window, 64);
    g = (mirage_gemm_t)MIRAGE_GEMM_DEFAULTS;
    g.k_len = 64; g.act_base = 0x0500u + 64u; g.out_base = 0x0600u;
    g.mult = 1.0 / 8192.0; g.acc_clr = 0; g.acc_wb = 1;
    if ((rc = mirage_gemm(&g)) != MIRAGE_OK) return fail("chunk hi", rc);
    check_act("gemm_chunked", 0x0600u, MIRAGE_N_LANES);

    /* -----------------------------------------------------------------
     * 4. STAGE + GEMM: the QK^T shape.
     * ----------------------------------------------------------------- */
    for (t = 0u; t < 3u; t++)
        mirage_write_act(REG_K + t * TOK_STRIDE, acts, HEAD_DIM);
    for (t = 0u; t < 3u; t++)
        mirage_write_act(REG_Q + t * TOK_STRIDE, acts, HEAD_DIM);

    rc = mirage_stage(REG_K, HEAD_DIM, 3u, TOK_STRIDE, 1, 0u);
    if (rc != MIRAGE_OK) return fail("stage K^T", rc);

    for (t = 0u; t < 3u; t++) {
        g = (mirage_gemm_t)MIRAGE_GEMM_DEFAULTS;
        g.k_len = HEAD_DIM; g.act_base = REG_Q + t * TOK_STRIDE;
        g.out_base = t * SCORE_ROW_STR;
        g.mult = 1.0 / 16.0; g.wb_score = 1; g.sc_wbase = 0u;
        if ((rc = mirage_gemm(&g)) != MIRAGE_OK) return fail("qk^t", rc);
    }
    check_score("scores", 0u, 3u * SCORE_ROW_STR);

    /* -----------------------------------------------------------------
     * 5. SMAX on those scores, then the xV matmul that consumes them.
     * ----------------------------------------------------------------- */
    for (t = 0u; t < 3u; t++)
        mirage_write_act(REG_V + t * TOK_STRIDE, acts, HEAD_DIM);

    for (ct = 0u; ct < 2u; ct++) {
        rc = mirage_stage(REG_V + ct * MIRAGE_N_LANES, 3u, MIRAGE_N_LANES,
                          TOK_STRIDE, 0, ct * 3u);
        if (rc != MIRAGE_OK) return fail("stage V", rc);
    }

    for (t = 0u; t < 3u; t++) {
        char label[16];

        if ((rc = mirage_smax(t * SCORE_ROW_STR, 3u)) != MIRAGE_OK)
            return fail("smax", rc);

        snprintf(label, sizeof label, "exp_row%u", t);
        check_act(label, REG_EXP, 3);

        g = (mirage_gemm_t)MIRAGE_GEMM_DEFAULTS;
        g.k_len = 3; g.act_base = REG_EXP;
        g.out_base = REG_ATTN + t * TOK_STRIDE;
        g.rq_shift = 7; g.use_recip = 1; g.n_coltile = 2;
        if ((rc = mirage_gemm(&g)) != MIRAGE_OK) return fail("xV", rc);
    }
    check_act("attn", REG_ATTN, TOK_STRIDE * 2u + HEAD_DIM);

    mirage_trace_close();
    return 0;
}
