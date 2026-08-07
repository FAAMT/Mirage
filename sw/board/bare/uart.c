/* =====================================================================
 * uart.c -- console on Zynq UART1, and the printf subset smoke.c needs
 *
 * The USB-UART is bridged by the same FT2232H that carries JTAG:
 * channel A is JTAG, channel B is this console, which is why the board
 * shows up as two ttyUSB nodes and only the second one is a terminal.
 *
 * WHICH PS UART, AND HOW TO CHECK
 *
 * UART0 at 0xE0000000, on MIO14/15. Not UART1 -- this block design
 * leaves UART1 disabled entirely, so a driver pointed at 0xE0001000
 * writes to a peripheral that is not clocked and produces exactly
 * nothing, with no error anywhere. That is a genuinely nasty failure
 * mode, so do not take this comment's word for it. The block design is
 * the authority, and it is readable straight out of the XSA:
 *
 *   unzip -p build_cora/mirage_cora.xsa mirage_cora.hwh \
 *     | grep -oE 'PCW_UART[0-9]_(PERIPHERAL_ENABLE|BASEADDR|[A-Z0-9]+_IO)[^/]*VALUE="[^"]*"'
 *
 *   PCW_UART0_PERIPHERAL_ENABLE = "1"     <- this one
 *   PCW_UART0_BASEADDR          = "0xE0000000"
 *   PCW_UART0_UART0_IO          = "MIO 14 .. 15"
 *   PCW_UART1_PERIPHERAL_ENABLE = "0"
 *
 * BAUD RATE IS NOT SET HERE, ON PURPOSE.
 *
 * ps7_init() already programmed BAUDGEN and BAUDDIV from the PS
 * configuration captured in the XSA, and it knows the UART reference
 * clock frequency that came out of the PLL settings. Recomputing the
 * divisors here would mean hardcoding a reference clock that the block
 * design is free to change, and getting it wrong yields a console that
 * emits plausible-looking garbage rather than an obvious failure. The
 * one number this file would need is the one number it cannot see.
 *
 * So: 115200 8N1 comes from the block design. If the console is garbled,
 * the fix is in the PS configuration, not here.
 * ===================================================================== */
#include <stdarg.h>
#include <stdint.h>
#include "xil_printf.h"

/* Overridable so a design that moves the console does not need a source
 * edit: make CFLAGS+=-DMIRAGE_UART_BASE=0xE0001000 */
#ifndef MIRAGE_UART_BASE
#define MIRAGE_UART_BASE 0xE0000000u
#endif

#define UART_CR      0x00u    /* control                                */
#define UART_SR      0x2Cu    /* status                                 */
#define UART_FIFO    0x30u    /* transmit / receive holding             */

#define CR_RXRST     (1u << 0)
#define CR_TXRST     (1u << 1)
#define CR_RXEN      (1u << 2)
#define CR_TXEN      (1u << 4)

#define SR_TXFULL    (1u << 4)
#define SR_TXEMPTY   (1u << 3)

static volatile uint32_t *const uart =
    (volatile uint32_t *)(uintptr_t)MIRAGE_UART_BASE;

static inline uint32_t rd(uint32_t off) { return uart[off / 4u]; }
static inline void     wr(uint32_t off, uint32_t v) { uart[off / 4u] = v; }

void uart_init(void)
{
    /* Flush whatever the BootROM or a previous run left in the FIFOs,
     * then enable both directions. Baud is left exactly as ps7_init
     * programmed it. */
    wr(UART_CR, CR_RXRST | CR_TXRST);
    while (rd(UART_CR) & (CR_RXRST | CR_TXRST))
        ;                                  /* self-clearing; wait it out */
    wr(UART_CR, CR_RXEN | CR_TXEN);
}

void uart_putc(char c)
{
    while (rd(UART_SR) & SR_TXFULL)
        ;
    wr(UART_FIFO, (uint32_t)(unsigned char)c);
}

void uart_puts(const char *s)
{
    while (*s)
        uart_putc(*s++);
}

/* Drain before returning from main or parking forever: the FIFO is 64
 * bytes deep, so without this the last line of a passing run can still
 * be in flight when the CPU stops and never reaches the terminal. */
void uart_flush(void)
{
    while (!(rd(UART_SR) & SR_TXEMPTY))
        ;
}

/* --- the formatter -------------------------------------------------- */

/* Longest output is a 32-bit value in binary-free bases: 10 digits for
 * decimal, 8 for hex. 12 is comfortable. */
#define NUMBUF 12

static void emit_num(unsigned long v, unsigned base, int upper,
                     int width, int zero_pad, int negative)
{
    char buf[NUMBUF];
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int n = 0;

    if (v == 0ul)
        buf[n++] = '0';
    while (v != 0ul && n < NUMBUF) {
        buf[n++] = digits[v % base];
        v /= base;
    }

    /* A negative sign counts against the field width, and when padding
     * with zeroes it has to be printed BEFORE them -- "-0042", never
     * "0-042". Getting this wrong is the classic mini-printf bug. */
    if (negative && zero_pad) {
        uart_putc('-');
        if (width > 0)
            width--;
    }

    {
        int len = n + ((negative && !zero_pad) ? 1 : 0);
        while (width > len) {
            uart_putc(zero_pad ? '0' : ' ');
            width--;
        }
    }

    if (negative && !zero_pad)
        uart_putc('-');

    while (n > 0)
        uart_putc(buf[--n]);
}

void xil_printf(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);

    while (*fmt) {
        int width = 0, zero_pad = 0;

        if (*fmt != '%') {
            uart_putc(*fmt++);
            continue;
        }
        fmt++;                                   /* consume '%' */

        if (*fmt == '0') { zero_pad = 1; fmt++; }
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt - '0');
            fmt++;
        }

        switch (*fmt) {
        case 'd': {
            long v = (long)va_arg(ap, int);
            int neg = v < 0;
            unsigned long m = neg ? (unsigned long)(-(v + 1)) + 1ul
                                  : (unsigned long)v;
            /* -(v) on LONG_MIN is undefined; the +1/-1 dance above is
             * why this is not simply -v. */
            emit_num(m, 10u, 0, width, zero_pad, neg);
            break;
        }
        case 'u':
            emit_num((unsigned long)va_arg(ap, unsigned int),
                     10u, 0, width, zero_pad, 0);
            break;
        case 'x':
            emit_num((unsigned long)va_arg(ap, unsigned int),
                     16u, 0, width, zero_pad, 0);
            break;
        case 'X':
            emit_num((unsigned long)va_arg(ap, unsigned int),
                     16u, 1, width, zero_pad, 0);
            break;
        case 'c':
            uart_putc((char)va_arg(ap, int));
            break;
        case 's': {
            const char *s = va_arg(ap, const char *);
            uart_puts(s ? s : "(null)");
            break;
        }
        case '%':
            uart_putc('%');
            break;
        case '\0':
            /* Trailing '%' with nothing after it. Stop rather than
             * running off the end of the string. */
            va_end(ap);
            return;
        default:
            /* Unknown conversion: show it rather than swallowing it, so
             * a format this subset does not implement is visible on the
             * console instead of silently dropping its argument. */
            uart_putc('%');
            uart_putc(*fmt);
            break;
        }
        fmt++;
    }

    va_end(ap);
}
