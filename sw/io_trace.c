/* =====================================================================
 * io_trace.c -- host backend: decode register traffic into the command
 * file format that model_ref/mirage_int8.py's Script class emits
 *
 * The point of decoding rather than instrumenting: the driver calls
 * mirage_wr32 with an offset and a word, exactly as it will on the
 * board, and this file works out what that meant. A wrong register
 * offset, a missing CFG write, weight words in the wrong order or a
 * bufsel flip on the wrong side of a load all come out as a diff
 * against the reference script. A tracing hook called at the driver's
 * own abstraction level would agree with every one of those mistakes.
 *
 * What it cannot check: anything the reference script does not encode.
 * ISR handling and the done-polling protocol produce no script lines,
 * so they are invisible here and are covered by sim/axi_lite_tb.vhd
 * instead.
 * ===================================================================== */
#include <stdio.h>
#include <stdint.h>

#include "mirage_io.h"
#include "mirage_regs.h"
#include "mirage_trace.h"

static FILE *out;

/* activation / score pointers, mirroring the hardware's auto-increment */
static uint32_t act_ptr;
static uint32_t sc_ptr;

/* a DUMP is a run of consecutive reads; it is only known to be over
 * when something else happens */
static int      dump_open;      /* 0 none, 1 DUMP (act), 2 DUMPS (score) */
static uint32_t dump_addr;
static uint32_t dump_count;

/* weight words accumulate into one WROW line per 8 */
static uint32_t wrow_words[MIRAGE_W_BANKS];
static uint32_t wrow_fill;
static uint32_t wrow_index;

static void flush_dump(void)
{
    if (dump_open == 1)
        fprintf(out, "DUMP %04x %u\n", dump_addr, dump_count);
    else if (dump_open == 2)
        fprintf(out, "DUMPS %04x %u\n", dump_addr, dump_count);
    dump_open  = 0;
    dump_count = 0u;
}

int mirage_trace_open(const char *path)
{
    out = fopen(path, "w");
    if (!out)
        return -1;
    act_ptr = sc_ptr = 0u;
    dump_open = 0; dump_count = 0u;
    wrow_fill = 0u; wrow_index = 0u;
    return 0;
}

void mirage_trace_close(void)
{
    if (!out)
        return;
    flush_dump();
    fprintf(out, "END\n");
    fclose(out);
    out = NULL;
}

/* ===================================================================== */
void mirage_wr32(uint32_t offset, uint32_t value)
{
    if (!out)
        return;

    if (offset >= MIRAGE_REG_CFG0 &&
        offset <  MIRAGE_REG_CFG0 + 4u * MIRAGE_CFG_COUNT) {
        flush_dump();
        fprintf(out, "CFG %x %08x\n",
                (offset - MIRAGE_REG_CFG0) / 4u, value);
        return;
    }

    switch (offset) {
    case MIRAGE_REG_CTRL:
        if (value & MIRAGE_CTRL_START) {
            flush_dump();
            fprintf(out, "JOB %x\n",
                    (value >> MIRAGE_CTRL_OP_SHIFT) & MIRAGE_CTRL_OP_MASK);
        }
        break;

    case MIRAGE_REG_ACT_PTR:
        act_ptr = value;
        break;

    case MIRAGE_REG_ACT_DATA:
        flush_dump();
        fprintf(out, "ACT %04x %02x\n", act_ptr, value & 0xFFu);
        act_ptr++;
        break;

    case MIRAGE_REG_SC_PTR:
        sc_ptr = value;
        break;

    case MIRAGE_REG_WGT_DATA:
    case MIRAGE_REG_WGT_LAST:
        flush_dump();
        if (wrow_fill < (uint32_t)MIRAGE_W_BANKS)
            wrow_words[wrow_fill] = value;
        wrow_fill++;

        if (wrow_fill == (uint32_t)MIRAGE_W_BANKS) {
            uint32_t i;
            fprintf(out, "WROW %x", wrow_index);
            for (i = 0u; i < (uint32_t)MIRAGE_W_BANKS; i++)
                fprintf(out, " %08x", wrow_words[i]);
            fprintf(out, "\n");
            wrow_fill = 0u;
            wrow_index++;
        }

        /* TLAST returns wgt_stream's beat counter to zero, so the next
         * window starts at row 0 again. Modelling that here is what
         * makes a driver that sends TLAST per row -- instead of per
         * window -- show up as every row landing at index 0. */
        if (offset == MIRAGE_REG_WGT_LAST)
            wrow_index = 0u;
        break;

    case MIRAGE_REG_ISR:
        /* Interrupt bookkeeping has no counterpart in the script. */
        break;

    default:
        flush_dump();
        fprintf(out, "# unexpected write: off=0x%02x val=0x%08x\n",
                offset, value);
        break;
    }
}

/* ===================================================================== */
uint32_t mirage_rd32(uint32_t offset)
{
    switch (offset) {
    case MIRAGE_REG_ID:
        return MIRAGE_ID_MAGIC;

    case MIRAGE_REG_STATUS:
        /* Jobs complete instantly and the window buffer reads back as
         * whatever CFG8 last set. Returning DONE unconditionally is the
         * only sane thing for a host trace -- there is no fabric to
         * wait for -- and it is why the polling protocol itself is not
         * under test here. */
        return MIRAGE_STATUS_DONE;

    case MIRAGE_REG_ISR:
        return MIRAGE_ISR_DONE;

    case MIRAGE_REG_ACT_DATA:
        if (out) {
            if (dump_open != 1) {
                flush_dump();
                dump_open  = 1;
                dump_addr  = act_ptr;
                dump_count = 0u;
            }
            dump_count++;
        }
        act_ptr++;
        return 0u;

    case MIRAGE_REG_SC_DATA:
        if (out) {
            if (dump_open != 2) {
                flush_dump();
                dump_open  = 2;
                dump_addr  = sc_ptr;
                dump_count = 0u;
            }
            dump_count++;
        }
        sc_ptr++;
        return 0u;

    default:
        return 0u;
    }
}

/* ===================================================================== */
void mirage_mark(const char *label)
{
    if (!out)
        return;
    flush_dump();
    fprintf(out, "MARK %s\n", label);
}
