/* =====================================================================
 * mirage_regs.h -- the AXI4-Lite register map, and nothing else
 *
 * Transcribed from the header of rtl_v2/axi/mirage_axi_lite.vhd and the
 * cfg decode in rtl_v2/ctrl/layer_ctrl.vhd. Those two files are the
 * authority; if they disagree with this one, they are right.
 *
 * No logic lives here on purpose -- this header is shared by the driver,
 * the trace backend that decodes register writes, and anything on the
 * host side that needs to name a register.
 * ===================================================================== */
#ifndef MIRAGE_REGS_H
#define MIRAGE_REGS_H

/* Base address. Pinned in tcl/bd_cora.tcl rather than left to
 * assign_bd_address, so it is a fact about the design and not a
 * property of the last block-design edit. The block design declares a
 * 4K segment; the register file decodes 8 address bits, so 0x100..0xFFF
 * alias back onto the same 64 words. */
#define MIRAGE_BASE_ADDR   0x43C00000u
#define MIRAGE_ADDR_SPAN   0x1000u

/* ---- registers (byte offsets) -------------------------------------- */
#define MIRAGE_REG_ID        0x00u  /* R    0x4D495247 "MIRG"           */
#define MIRAGE_REG_CTRL      0x04u  /* W    [0] start, [2:1] job_op     */
#define MIRAGE_REG_STATUS    0x08u  /* R    [0] busy [1] done [2] buf   */
#define MIRAGE_REG_ISR       0x0Cu  /* RW1C [0] job done [1] cfg-busy   */
#define MIRAGE_REG_ACT_PTR   0x10u  /* RW   activation byte pointer     */
#define MIRAGE_REG_ACT_DATA  0x14u  /* RW   auto-incrementing byte port */
#define MIRAGE_REG_SC_PTR    0x18u  /* RW   score word pointer          */
#define MIRAGE_REG_SC_DATA   0x1Cu  /* R    auto-incrementing, signed   */
#define MIRAGE_REG_WGT_DATA  0x24u  /* W    push one weight word        */
#define MIRAGE_REG_WGT_LAST  0x28u  /* W    push one word with TLAST    */
#define MIRAGE_REG_CFG0      0x40u  /* W    CFG n is at 0x40 + 4n       */

#define MIRAGE_CFG(n)        (MIRAGE_REG_CFG0 + 4u * (unsigned)(n))
#define MIRAGE_CFG_COUNT     11u    /* CFG0 .. CFG10 (0x40 .. 0x68)     */

#define MIRAGE_ID_MAGIC      0x4D495247u   /* "MIRG" big-endian ASCII   */

/* ---- CTRL ---------------------------------------------------------- */
#define MIRAGE_CTRL_START       (1u << 0)
#define MIRAGE_CTRL_OP_SHIFT    1
#define MIRAGE_CTRL_OP_MASK     0x3u

/* ---- STATUS --------------------------------------------------------
 *
 * Poll DONE, never the fabric's `done`. layer_ctrl's done is a
 * one-cycle pulse at 100 MHz; software cannot observe it and must not
 * try. done_sticky latches it and is cleared by writing ISR[0]. */
#define MIRAGE_STATUS_BUSY      (1u << 0)
#define MIRAGE_STATUS_DONE      (1u << 1)
#define MIRAGE_STATUS_WBUF      (1u << 2)  /* wstream_buf, reflects CFG8 */

/* ---- ISR (write 1 to clear) ---------------------------------------- */
#define MIRAGE_ISR_DONE         (1u << 0)
#define MIRAGE_ISR_CFG_BUSY     (1u << 1)  /* a CFG write was dropped   */

/* ---- job opcodes (CTRL[2:1]) --------------------------------------- */
#define MIRAGE_JOB_GEMM   0u
#define MIRAGE_JOB_STAGE  1u
#define MIRAGE_JOB_SMAX   2u

/* ---- CFG register indices (layer_ctrl cfg_addr) -------------------- */
#define MIRAGE_CFG_KLEN     0x0u  /* reduction length                   */
#define MIRAGE_CFG_FLAGS    0x1u  /* packed, see below                  */
#define MIRAGE_CFG_ABASE    0x2u  /* activation base                    */
#define MIRAGE_CFG_OBASE    0x3u  /* output base                        */
#define MIRAGE_CFG_STBASE   0x4u  /* stage source base                  */
#define MIRAGE_CFG_STGEO    0x5u  /* stage geometry, packed             */
#define MIRAGE_CFG_STSTRIDE 0x6u  /* stage row stride                   */
#define MIRAGE_CFG_SM       0x7u  /* softmax base + len, packed         */
#define MIRAGE_CFG_BUFSEL   0x8u  /* [0] compute-side window buffer     */
#define MIRAGE_CFG_SCWBASE  0x9u  /* score writeback base               */
#define MIRAGE_CFG_SCALE    0xAu  /* 18-bit requantise multiplier       */

/* ---- CFG_FLAGS bit layout ------------------------------------------ */
#define MIRAGE_FLAG_NTOK_SHIFT      0
#define MIRAGE_FLAG_NTOK_MASK       0x1Fu
#define MIRAGE_FLAG_NCOLTILE_SHIFT  8
#define MIRAGE_FLAG_NCOLTILE_MASK   0x1Fu
#define MIRAGE_FLAG_RQSHIFT_SHIFT   16
#define MIRAGE_FLAG_RQSHIFT_MASK    0x3Fu
#define MIRAGE_FLAG_ACC_CLR         (1u << 24)
#define MIRAGE_FLAG_ACC_WB          (1u << 25)
#define MIRAGE_FLAG_WB_SCORE        (1u << 26)
#define MIRAGE_FLAG_USE_RECIP       (1u << 27)

/* ---- CFG_STGEO bit layout ------------------------------------------ */
#define MIRAGE_STGEO_ROWS_SHIFT     0
#define MIRAGE_STGEO_ROWS_MASK      0x7Fu
#define MIRAGE_STGEO_LANES_SHIFT    8
#define MIRAGE_STGEO_LANES_MASK     0x7Fu
#define MIRAGE_STGEO_XPOSE          (1u << 16)
#define MIRAGE_STGEO_ROWOFS_SHIFT   18
#define MIRAGE_STGEO_ROWOFS_MASK    0x3FFu

/* ---- CFG_SM bit layout --------------------------------------------- */
#define MIRAGE_SM_BASE_MASK         0xFFFu
#define MIRAGE_SM_LEN_SHIFT         16
#define MIRAGE_SM_LEN_MASK          0x1Fu

/* ---- geometry, from rtl_v2/pkg/mirage_v2_pkg.vhd ------------------- */
#define MIRAGE_N_LANES      32
#define MIRAGE_W_BANKS      8
#define MIRAGE_BANK_BYTES   4      /* int8 per 32-bit bank word         */
#define MIRAGE_BANK_DEPTH   1024   /* window rows per buffer            */

#endif /* MIRAGE_REGS_H */
