/* =====================================================================
 * smoke.c -- the first thing to run on the board
 *
 * Bare-metal Vitis application. Build with mirage.c and io_mmio.c.
 *
 * Three stages, in the order that makes a failure diagnosable:
 *
 *   1. ID       proves the bitstream loaded, FCLK_CLK0 is running, GP0
 *               is enabled and 0x43C00000 is where the design put the
 *               accelerator. Everything else assumes all four.
 *
 *   2. loopback proves the activation port works on its own, with no
 *               fabric involvement: write a pattern, read it back. If
 *               this fails the datapath is not the suspect, the AXI
 *               shim is.
 *
 *   3. GEMM     all-ones activations against an all-ones window,
 *               reducing over 64, with a unity multiplier. Every output
 *               byte must be 64.
 *
 * Stage 3 is a real end-to-end test in disguise: it exercises
 * activation memory, the weight window, both bufsel flips, the MAC
 * array, the requantiser and the writeback path. The expected value
 * follows from the reference arithmetic in model_ref/mirage_int8.py --
 * acc = 64, split_scale(1.0) = (65536, 0), (64 * 65536) >> 16 = 64 --
 * and not from having run it once and written down what came out.
 * ===================================================================== */
#include "mirage.h"
#include "mirage_io.h"

#if defined(__arm__) && !defined(MIRAGE_HOST_BUILD)
#include "xil_printf.h"
#define PRINT xil_printf
#else
#include <stdio.h>
#define PRINT printf
#endif

#define K_LEN     64u
#define OUT_BASE  0x2000u
#define ACT_BASE  0x0000u
#define EXPECT    64

static int8_t acts[K_LEN];
static int8_t window[K_LEN * MIRAGE_N_LANES];
static int8_t out[MIRAGE_N_LANES];
static int8_t back[K_LEN];

int main(void)
{
    mirage_gemm_t g;
    uint32_t i;
    int rc, bad;

    PRINT("\r\n=== Mirage bring-up ===\r\n");

    /* -- 1. ID -------------------------------------------------------- */
    rc = mirage_probe();
    PRINT("ID       : 0x%08x  %s\r\n",
          (unsigned)mirage_rd32(MIRAGE_REG_ID),
          rc == MIRAGE_OK ? "OK" : "MISMATCH (expected 0x4D495247)");
    if (rc != MIRAGE_OK) {
        PRINT("stopping: nothing below this can mean anything.\r\n");
        return 1;
    }
    mirage_reset_status();

    /* -- 2. activation loopback --------------------------------------- */
    for (i = 0u; i < K_LEN; i++)
        acts[i] = (int8_t)(i - 32);          /* spans negative and positive */

    mirage_write_act(ACT_BASE, acts, K_LEN);
    mirage_read_act(ACT_BASE, back, K_LEN);

    bad = 0;
    for (i = 0u; i < K_LEN; i++)
        if (back[i] != acts[i])
            bad++;
    PRINT("loopback : %s (%d of %u bytes wrong)\r\n",
          bad ? "FAIL" : "OK", bad, (unsigned)K_LEN);
    if (bad)
        return 1;

    /* -- 3. all-ones GEMM --------------------------------------------- */
    for (i = 0u; i < K_LEN; i++)
        acts[i] = 1;
    for (i = 0u; i < K_LEN * MIRAGE_N_LANES; i++)
        window[i] = 1;

    mirage_write_act(ACT_BASE, acts, K_LEN);
    mirage_load_window(window, K_LEN);

    g = (mirage_gemm_t)MIRAGE_GEMM_DEFAULTS;
    g.k_len = K_LEN;
    g.act_base = ACT_BASE;
    g.out_base = OUT_BASE;
    g.mult = 1.0;

    rc = mirage_gemm(&g);
    if (rc != MIRAGE_OK) {
        PRINT("gemm     : FAIL rc=%d  (STATUS=0x%08x)\r\n", rc,
              (unsigned)mirage_rd32(MIRAGE_REG_STATUS));
        return 1;
    }

    mirage_read_act(OUT_BASE, out, MIRAGE_N_LANES);

    bad = 0;
    for (i = 0u; i < (uint32_t)MIRAGE_N_LANES; i++)
        if (out[i] != EXPECT)
            bad++;

    PRINT("gemm     : %s  expected %d in all %d lanes\r\n",
          bad ? "FAIL" : "OK", EXPECT, MIRAGE_N_LANES);
    if (bad) {
        PRINT("  got:");
        for (i = 0u; i < (uint32_t)MIRAGE_N_LANES; i++)
            PRINT(" %d", out[i]);
        PRINT("\r\n");
        return 1;
    }

    PRINT("\r\nALL PASS -- datapath confirmed on silicon.\r\n");
    return 0;
}
