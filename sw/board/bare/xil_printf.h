/* =====================================================================
 * xil_printf.h -- the name board/smoke.c already expects
 *
 * smoke.c selects its output function like this:
 *
 *     #if defined(__arm__) && !defined(MIRAGE_HOST_BUILD)
 *     #include "xil_printf.h"
 *     #define PRINT xil_printf
 *
 * That was written for a Vitis build. Keeping the name means smoke.c --
 * which is verified code -- compiles unmodified against this standalone
 * build too. The alternative was editing smoke.c to call something else,
 * which would have meant the thing running on the board was no longer
 * the thing that was checked.
 *
 * The implementation in uart.c is a real subset, not a wrapper around
 * newlib's printf: %s %c %d %u %x %% with optional zero-padded width.
 * That is exactly what smoke.c uses and nothing more. Pulling in
 * newlib's vfprintf would cost ~30 KB and drag in _write, _sbrk and the
 * rest of the syscall surface for no gain on a UART with no filesystem
 * behind it.
 * ===================================================================== */
#ifndef XIL_PRINTF_H
#define XIL_PRINTF_H

void xil_printf(const char *fmt, ...);

/* Used by crt0 before the C environment is fully up, and by the panic
 * path when even the formatter is suspect. */
void uart_putc(char c);
void uart_puts(const char *s);
void uart_init(void);

/* Blocks until the 64-byte TX FIFO has actually drained. Needed before
 * the CPU parks, or the last line of a run is still in flight when
 * everything stops and never appears on the terminal. */
void uart_flush(void);

#endif /* XIL_PRINTF_H */
