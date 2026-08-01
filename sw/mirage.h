/* =====================================================================
 * mirage.h -- bare-metal driver for the Mirage int8 encoder layer
 *
 * This is a transliteration of the Mirage class in
 * model_ref/mirage_int8.py, which is the reference the RTL was verified
 * against bit-exactly. Same operations, same order, same register
 * writes -- deliberately, because the register sequence is the part
 * that has to be right and it has already been proven once. Deviating
 * from it to be more idiomatic would throw away that proof.
 *
 * C has no keyword arguments, so the GEMM parameters arrive in a struct
 * with a defaults initialiser rather than as a dozen positional
 * arguments in an order nobody can remember:
 *
 *     mirage_gemm_t g = MIRAGE_GEMM_DEFAULTS;
 *     g.k_len = 64; g.act_base = 0x0000; g.out_base = 0x2000;
 *     g.mult  = 1.0 / 4096;
 *     mirage_gemm(&g);
 * ===================================================================== */
#ifndef MIRAGE_H
#define MIRAGE_H

#include <stdint.h>
#include <stddef.h>
#include "mirage_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- return codes --------------------------------------------------
 * Negative on failure so callers can test < 0. */
#define MIRAGE_OK            0
#define MIRAGE_E_ID        (-1)   /* ID register did not read "MIRG"    */
#define MIRAGE_E_TIMEOUT   (-2)   /* job did not raise done_sticky      */
#define MIRAGE_E_CFG_BUSY  (-3)   /* a CFG write was dropped (ISR[1])   */
#define MIRAGE_E_RANGE     (-4)   /* a parameter does not fit its field */

/* =====================================================================
 * Bring-up
 * ===================================================================== */

/* Read ID and check it against "MIRG". The first thing to run on the
 * board: it proves the bitstream loaded, the PS-PL clock is running,
 * GP0 is enabled and the base address is right. Every other failure is
 * easier to reason about once this passes. */
int mirage_probe(void);

/* Reset sticky state. Clears both ISR bits and leaves the window buffer
 * selection at 0. Does not reset the fabric. */
void mirage_reset_status(void);

/* =====================================================================
 * Activation memory (PIO, auto-incrementing)
 * ===================================================================== */

/* Write `count` signed bytes starting at `addr`. Values are truncated
 * to int8 by the hardware port; passing anything outside [-128,127] is
 * a caller bug and returns MIRAGE_E_RANGE without writing. */
int mirage_write_act(uint32_t addr, const int8_t *values, size_t count);

/* Read `count` bytes back, sign-extended. */
void mirage_read_act(uint32_t addr, int8_t *out, size_t count);

/* Read `count` score words, sign-extended from SCORE_W bits. */
void mirage_read_scores(uint32_t addr, int32_t *out, size_t count);

/* =====================================================================
 * Weight window
 * ===================================================================== */

/* Load `rows` window rows of MIRAGE_N_LANES int8 each, starting at
 * window row 0, over the PIO weight path.
 *
 * BUFSEL IS BRACKETED AROUND THE LOAD, and that is not decoration.
 * weight_window writes to the buffer NOT selected by bufsel, so a
 * caller that loads and then computes from the same buffer must flip
 * bufsel around the load: fill with bufsel=1, compute with bufsel=0.
 * Getting this backwards is self-consistent -- load and compute agree
 * with each other -- so it passes every test while silently removing
 * the ability to overlap them. See the note in weight_window.vhd.
 *
 * This is the slow path by design; it is for bring-up, not production.
 * The DMA over s_axis is the real one. */
void mirage_load_window(const int8_t *rows, uint32_t n_rows);

/* =====================================================================
 * Jobs
 * ===================================================================== */

typedef struct {
    uint32_t k_len;      /* reduction length                            */
    uint32_t act_base;   /* activation base address                     */
    uint32_t out_base;   /* output base address                         */

    /* Two ways to set the output scale, and they are NOT
     * interchangeable:
     *   use_recip = 0   `mult` is a real multiplier, split here into an
     *                   18-bit scale and a shift
     *   use_recip = 1   the scale comes from recip_rom[sum8] inside the
     *                   fabric and only `rq_shift` is ours to choose */
    double   mult;
    uint32_t rq_shift;
    int      use_recip;

    uint32_t n_tok;      /* tokens in this job          (default 1)     */
    uint32_t n_coltile;  /* output column tiles         (default 1)     */
    int      acc_clr;    /* clear the accumulator first (default 1)     */
    int      acc_wb;     /* write the accumulator back  (default 1)     */
    int      wb_score;   /* write to score memory       (default 0)     */
    uint32_t sc_wbase;   /* score writeback base                        */
} mirage_gemm_t;

#define MIRAGE_GEMM_DEFAULTS                                        \
    { 0u, 0u, 0u, 0.0, 0u, 0, 1u, 1u, 1, 1, 0, 0u }

int mirage_gemm(const mirage_gemm_t *g);

/* Stage K^T or V out of activation memory into the weight window.
 *
 * The other writer to the window, and subject to the same bufsel rule
 * as mirage_load_window -- staging into a buffer runs with bufsel=1 and
 * the job that consumes it reads with bufsel=0. A staged window and its
 * consumer are strictly sequential, so this pair can never overlap; the
 * ping-pong the corrected hardware allows is for the DDR stream. */
int mirage_stage(uint32_t base, uint32_t rows, uint32_t lanes,
                 uint32_t stride, int xpose, uint32_t row_ofs);

/* Softmax over `len` scores at `base`. Leaves the exp stream where a
 * following GEMM with use_recip=1 can consume it. */
int mirage_smax(uint32_t base, uint32_t len);

/* =====================================================================
 * Internals, exposed because they are worth testing directly
 * ===================================================================== */

/* Represent a real multiplier as scale / 2**(16 + shift), with scale
 * kept in [2^16, 2^17) so the 18-bit field carries its full precision.
 * Returns MIRAGE_E_RANGE if the multiplier will not fit. */
int mirage_split_scale(double mult, uint32_t *scale, uint32_t *shift);

/* Start a job and wait for done_sticky. Exposed so a caller that has
 * written CFG itself can still use the correct completion protocol. */
int mirage_run_job(uint32_t op);

#ifdef __cplusplus
}
#endif

#endif /* MIRAGE_H */
