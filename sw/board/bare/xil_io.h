/* =====================================================================
 * xil_io.h -- a stub, deliberately empty of anything real
 *
 * ps7_init_gpl.c ends with `#include "xil_io.h"`, but it never uses a
 * single thing from it: the file defines its own mask_write() and
 * mask_poll() a few lines further down, both plain volatile pointer
 * writes. The include is vestigial -- Vivado emits it whether or not the
 * generated code needs it.
 *
 * So this header exists to satisfy the preprocessor and nothing else.
 * That is the entire reason the PS bring-up can be built with a stock
 * arm-none-eabi-gcc and no Xilinx BSP: the 12,500 lines of ps7_init are
 * a table of register writes, not a client of the Xilinx runtime.
 *
 * If a future XSA emits a ps7_init that genuinely calls Xil_In32 or
 * Xil_Out32, the build breaks here with an implicit-declaration error
 * rather than silently linking against nothing. Implement them at that
 * point -- they are one volatile access each.
 * ===================================================================== */
#ifndef XIL_IO_H
#define XIL_IO_H

#endif /* XIL_IO_H */
