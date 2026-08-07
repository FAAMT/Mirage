/* =====================================================================
 * start.c -- the first C the board runs
 *
 * crt0.S hands over here with the two PS bring-up status codes, because
 * assembly is a bad place to decide what to print and the console does
 * not exist until ps7_init() has run anyway.
 *
 * The ordering below is the point of this file: bring the UART up FIRST,
 * then report on ps7_init. A bring-up failure you cannot see is
 * indistinguishable from a dead board, and ps7_init failing while the
 * UART still works is a real and diagnosable state -- it means a mask
 * poll timed out, usually DDR, which this smoke test does not even use.
 * ===================================================================== */
#include "xil_printf.h"

extern int main(void);

/* ps7_init_gpl.c's own success code. Declared here rather than pulling
 * in ps7_init_gpl.h, which drags the whole EMIT_* macro apparatus in
 * for one integer. */
#define PS7_INIT_SUCCESS 0

void _cstart(int ps7_status, int post_status)
{
    uart_init();

    if (ps7_status != PS7_INIT_SUCCESS) {
        xil_printf("\r\nps7_init FAILED: %d\r\n", ps7_status);
        xil_printf("Clocks or DDR did not come up. FCLK_CLK0 feeds the\r\n");
        xil_printf("PL, so the accelerator is probably not clocked.\r\n");
        /* Deliberately continue: the ID register read in smoke.c is a
         * better diagnostic than stopping here, and it is safe -- a
         * dead bus reads as zeroes, not as a fault. */
    }

    if (post_status != PS7_INIT_SUCCESS)
        xil_printf("\r\nps7_post_config FAILED: %d "
                   "(PS-PL level shifters)\r\n", post_status);

    (void)main();

    /* Let the last line reach the terminal before crt0 parks the CPU.
     * The TX FIFO is 64 bytes deep and a passing run ends with a line
     * longer than that. */
    uart_flush();
}
