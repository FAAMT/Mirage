/* =====================================================================
 * mirage_io.h -- the two calls everything else is built on
 *
 * The driver touches the accelerator only through these. Swapping the
 * implementation swaps what "the accelerator" means:
 *
 *   io_mmio.c    a volatile pointer at MIRAGE_BASE_ADDR -- the board
 *   io_trace.c   a decoder that turns register traffic back into the
 *                command-file format model_ref/mirage_int8.py emits
 *
 * The second one is why the driver can be checked before any hardware
 * exists. Note that the trace backend decodes REGISTER WRITES rather
 * than being called by the driver at a higher level: the driver has no
 * idea it is being traced, so a wrong offset or a wrong write order
 * shows up in the diff instead of being papered over by a tracing hook
 * that agrees with the mistake.
 * ===================================================================== */
#ifndef MIRAGE_IO_H
#define MIRAGE_IO_H

#include <stdint.h>

void     mirage_wr32(uint32_t offset, uint32_t value);
uint32_t mirage_rd32(uint32_t offset);

/* Test annotation. No register behind it: it exists so a trace can be
 * labelled the way model_ref's check_act() labels its dumps, and it
 * compiles to nothing on hardware. */
void     mirage_mark(const char *label);

#endif /* MIRAGE_IO_H */
