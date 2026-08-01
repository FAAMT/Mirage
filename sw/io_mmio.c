/* =====================================================================
 * io_mmio.c -- the real backend: a volatile pointer at the base address
 *
 * Link this for anything that runs on the board. It is deliberately the
 * shorter of the two backends; all the cleverness is in io_trace.c,
 * where it cannot affect what the hardware sees.
 * ===================================================================== */
#include "mirage_io.h"
#include "mirage_regs.h"

#ifndef MIRAGE_MMIO_BASE
#define MIRAGE_MMIO_BASE MIRAGE_BASE_ADDR
#endif

/* volatile: every access is a bus transaction with a side effect.
 * ACT_DATA in particular auto-increments its pointer on each access, so
 * a compiler that decided two reads of the same address could be folded
 * into one would silently halve a readback. */
static volatile uint32_t *const mirage_regs =
    (volatile uint32_t *)(uintptr_t)MIRAGE_MMIO_BASE;

void mirage_wr32(uint32_t offset, uint32_t value)
{
    mirage_regs[offset / 4u] = value;
}

uint32_t mirage_rd32(uint32_t offset)
{
    return mirage_regs[offset / 4u];
}

void mirage_mark(const char *label)
{
    (void)label;   /* a trace annotation; nothing to do on hardware */
}
