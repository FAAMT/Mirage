/* =====================================================================
 * mirage.c -- driver implementation
 *
 * Every register write here has a counterpart in
 * model_ref/mirage_int8.py. Where the order looks arbitrary it is not:
 * it is the order the RTL was verified with, and sw/test/check_ops.sh
 * diffs this file's register traffic against the reference script to
 * keep it that way.
 * ===================================================================== */
#include <math.h>

#include "mirage.h"
#include "mirage_io.h"

/* How long to wait for a job. The longest job in a layer is a GEMM
 * reducing over BANK_DEPTH with several column tiles -- tens of
 * thousands of cycles at 100 MHz, i.e. under a millisecond. A poll
 * budget this large is not a real time limit, it is a guarantee that a
 * hung fabric returns an error instead of wedging the CPU. */
#define MIRAGE_POLL_LIMIT  20000000u

/* ===================================================================== */
int mirage_probe(void)
{
    if (mirage_rd32(MIRAGE_REG_ID) != MIRAGE_ID_MAGIC)
        return MIRAGE_E_ID;
    return MIRAGE_OK;
}

void mirage_reset_status(void)
{
    mirage_wr32(MIRAGE_REG_ISR, MIRAGE_ISR_DONE | MIRAGE_ISR_CFG_BUSY);
}

/* ===================================================================== */
int mirage_split_scale(double mult, uint32_t *scale, uint32_t *shift)
{
    uint32_t rq = 0u;
    double   s;

    if (!(mult > 0.0))
        return MIRAGE_E_RANGE;

    /* ldexp rather than (1 << (16 + rq)): the shift overflows a 32-bit
     * int well before rq reaches its 63 limit, and the reference does
     * this in Python where integers do not overflow at all. */
    while (mult * ldexp(1.0, (int)(16u + rq)) < 65536.0 && rq < 63u)
        rq++;

    /* nearbyint under the default rounding mode is round-half-to-even,
     * which is what Python's round() does. The distinction only bites
     * on exact halves, but "only on exact halves" is exactly where a
     * scale factor lands when someone picks a power of two. */
    s = nearbyint(mult * ldexp(1.0, (int)(16u + rq)));

    if (s > (double)((1u << 18) - 1u))
        return MIRAGE_E_RANGE;

    *scale = (uint32_t)(s < 1.0 ? 1.0 : s);
    *shift = rq;
    return MIRAGE_OK;
}

/* ===================================================================== */
int mirage_run_job(uint32_t op)
{
    uint32_t n = 0u;
    uint32_t status;

    mirage_wr32(MIRAGE_REG_CTRL,
                MIRAGE_CTRL_START |
                ((op & MIRAGE_CTRL_OP_MASK) << MIRAGE_CTRL_OP_SHIFT));

    /* STATUS[1], never the fabric's `done`. done is one cycle wide at
     * 100 MHz and software cannot see it; done_sticky exists precisely
     * so that a poll loop is a legitimate way to wait. */
    for (;;) {
        status = mirage_rd32(MIRAGE_REG_STATUS);
        if (status & MIRAGE_STATUS_DONE)
            break;
        if (++n >= MIRAGE_POLL_LIMIT)
            return MIRAGE_E_TIMEOUT;
    }

    if (mirage_rd32(MIRAGE_REG_ISR) & MIRAGE_ISR_CFG_BUSY) {
        /* A CFG write landed while the fabric was busy and was dropped
         * rather than corrupting a running job. The job that just
         * finished ran with stale configuration, so its output is not
         * trustworthy and saying so is more use than returning OK. */
        mirage_wr32(MIRAGE_REG_ISR, MIRAGE_ISR_DONE | MIRAGE_ISR_CFG_BUSY);
        return MIRAGE_E_CFG_BUSY;
    }

    mirage_wr32(MIRAGE_REG_ISR, MIRAGE_ISR_DONE);
    return MIRAGE_OK;
}

/* ===================================================================== */
int mirage_write_act(uint32_t addr, const int8_t *values, size_t count)
{
    size_t i;

    mirage_wr32(MIRAGE_REG_ACT_PTR, addr);
    for (i = 0u; i < count; i++)
        mirage_wr32(MIRAGE_REG_ACT_DATA, (uint32_t)(values[i] & 0xFF));

    return MIRAGE_OK;
}

void mirage_read_act(uint32_t addr, int8_t *out, size_t count)
{
    size_t i;

    mirage_wr32(MIRAGE_REG_ACT_PTR, addr);
    for (i = 0u; i < count; i++)
        out[i] = (int8_t)(mirage_rd32(MIRAGE_REG_ACT_DATA) & 0xFFu);
}

void mirage_read_scores(uint32_t addr, int32_t *out, size_t count)
{
    size_t i;

    mirage_wr32(MIRAGE_REG_SC_PTR, addr);
    for (i = 0u; i < count; i++)
        out[i] = (int32_t)mirage_rd32(MIRAGE_REG_SC_DATA);
}

/* ===================================================================== */
void mirage_load_window(const int8_t *rows, uint32_t n_rows)
{
    uint32_t r, b, i;
    uint32_t total = n_rows * (uint32_t)MIRAGE_W_BANKS;
    uint32_t sent  = 0u;

    /* Writes land in the buffer NOT selected by bufsel: fill with 1,
     * compute with 0. See the note in mirage.h. */
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_BUFSEL), 1u);

    for (r = 0u; r < n_rows; r++) {
        for (b = 0u; b < (uint32_t)MIRAGE_W_BANKS; b++) {
            uint32_t word = 0u;

            /* Bank b carries lanes 4b..4b+3, little-endian within the
             * word -- the packing wgt_stream reverses when it fans a
             * beat out across the banks. */
            for (i = 0u; i < (uint32_t)MIRAGE_BANK_BYTES; i++) {
                uint32_t lane = b * (uint32_t)MIRAGE_BANK_BYTES + i;
                uint32_t byte = (uint32_t)
                    (rows[r * (uint32_t)MIRAGE_N_LANES + lane] & 0xFF);
                word |= byte << (8u * i);
            }

            /* TLAST returns wgt_stream's counter to zero, so it marks
             * the end of the WINDOW, not the end of a row. Sending it
             * per row would restart addressing every 8 words and stack
             * the whole window into row 0. */
            sent++;
            mirage_wr32(sent == total ? MIRAGE_REG_WGT_LAST
                                      : MIRAGE_REG_WGT_DATA, word);
        }
    }

    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_BUFSEL), 0u);
}

/* ===================================================================== */
int mirage_gemm(const mirage_gemm_t *g)
{
    uint32_t flags, scale = 0u, rq;
    int      rc;

    if (g->n_tok     > MIRAGE_FLAG_NTOK_MASK ||
        g->n_coltile > MIRAGE_FLAG_NCOLTILE_MASK)
        return MIRAGE_E_RANGE;

    if (g->use_recip) {
        rq = g->rq_shift;
        if (rq > MIRAGE_FLAG_RQSHIFT_MASK)
            return MIRAGE_E_RANGE;
    } else {
        rc = mirage_split_scale(g->mult, &scale, &rq);
        if (rc != MIRAGE_OK)
            return rc;
    }

    flags = ((g->n_tok     & MIRAGE_FLAG_NTOK_MASK)
                                  << MIRAGE_FLAG_NTOK_SHIFT)
          | ((g->n_coltile & MIRAGE_FLAG_NCOLTILE_MASK)
                                  << MIRAGE_FLAG_NCOLTILE_SHIFT)
          | ((rq           & MIRAGE_FLAG_RQSHIFT_MASK)
                                  << MIRAGE_FLAG_RQSHIFT_SHIFT)
          | (g->acc_clr   ? MIRAGE_FLAG_ACC_CLR   : 0u)
          | (g->acc_wb    ? MIRAGE_FLAG_ACC_WB    : 0u)
          | (g->wb_score  ? MIRAGE_FLAG_WB_SCORE  : 0u)
          | (g->use_recip ? MIRAGE_FLAG_USE_RECIP : 0u);

    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_KLEN),  g->k_len);
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_FLAGS), flags);
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_ABASE), g->act_base);
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_OBASE), g->out_base);
    if (!g->use_recip)
        mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_SCALE), scale);
    if (g->wb_score)
        mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_SCWBASE), g->sc_wbase);

    return mirage_run_job(MIRAGE_JOB_GEMM);
}

/* ===================================================================== */
int mirage_stage(uint32_t base, uint32_t rows, uint32_t lanes,
                 uint32_t stride, int xpose, uint32_t row_ofs)
{
    uint32_t geo;
    int      rc;

    if (rows    > MIRAGE_STGEO_ROWS_MASK  ||
        lanes   > MIRAGE_STGEO_LANES_MASK ||
        row_ofs > MIRAGE_STGEO_ROWOFS_MASK)
        return MIRAGE_E_RANGE;

    geo = ((rows    & MIRAGE_STGEO_ROWS_MASK)   << MIRAGE_STGEO_ROWS_SHIFT)
        | ((lanes   & MIRAGE_STGEO_LANES_MASK)  << MIRAGE_STGEO_LANES_SHIFT)
        | (xpose ? MIRAGE_STGEO_XPOSE : 0u)
        | ((row_ofs & MIRAGE_STGEO_ROWOFS_MASK) << MIRAGE_STGEO_ROWOFS_SHIFT);

    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_STBASE),   base);
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_STGEO),    geo);
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_STSTRIDE), stride);

    /* Same bracketing as a window load, for the same reason: the stage
     * engine is the other writer to the window. */
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_BUFSEL), 1u);
    rc = mirage_run_job(MIRAGE_JOB_STAGE);
    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_BUFSEL), 0u);

    return rc;
}

/* ===================================================================== */
int mirage_smax(uint32_t base, uint32_t len)
{
    if (len > MIRAGE_SM_LEN_MASK)
        return MIRAGE_E_RANGE;

    mirage_wr32(MIRAGE_CFG(MIRAGE_CFG_SM),
                (base & MIRAGE_SM_BASE_MASK)
                | ((len & MIRAGE_SM_LEN_MASK) << MIRAGE_SM_LEN_SHIFT));

    return mirage_run_job(MIRAGE_JOB_SMAX);
}
