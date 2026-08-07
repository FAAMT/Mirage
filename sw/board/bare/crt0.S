/* =====================================================================
 * crt0.S -- reset vector to main(), with no BSP underneath
 *
 * Replaces what a Vitis standalone platform would have generated. The
 * sequence matters more than the length:
 *
 *   1. SVC mode, interrupts masked
 *   2. MMU and caches OFF          <- see the note below, this is load bearing
 *   3. stack, then .bss cleared
 *   4. ps7_init()                  <- PLLs, MIO, DDR, FCLK_CLK0
 *   5. ps7_post_config()           <- PS-PL level shifters
 *   6. _cstart()                   <- C from here on
 *
 * WHY CACHES STAY OFF
 *
 * The accelerator is memory-mapped at 0x43C00000 and several of its
 * registers have side effects on access -- ACT_DATA auto-increments its
 * pointer every time it is touched. With the D-cache enabled and no MMU
 * to mark that range as Device memory, a write could sit in a dirty line
 * and never reach the PL, and two reads could be folded into one. The
 * `volatile` in io_mmio.c stops the compiler reordering things; it says
 * nothing about the cache. Running uncached costs performance this smoke
 * test does not care about and removes a whole class of bring-up
 * failures that look like datapath bugs.
 *
 * WHY ps7_post_config() IS NOT OPTIONAL
 *
 * It raises the PS-PL level shifters. Until it runs, AXI transactions
 * from the CPU do not reach the fabric at all: reads of the ID register
 * return garbage or hang the bus. On a normal boot the FSBL calls it
 * after loading the bitstream. There is no FSBL here, so we call it
 * ourselves -- and because the bitstream is programmed over JTAG before
 * this code runs, the ordering works out the same.
 * ===================================================================== */
    .syntax unified
    .arm

/* --- exception vectors ------------------------------------------------
 * Only reset is implemented. Anything else parks, which on this design
 * means a data abort shows up as a hung board rather than a silent
 * reset loop -- easier to recognise with a JTAG attached.
 */
    .section .vectors, "ax"
    .global _vectors
_vectors:
    b       _start          /* 0x00  reset                  */
    b       _hang           /* 0x04  undefined instruction  */
    b       _hang           /* 0x08  supervisor call        */
    b       _hang           /* 0x0C  prefetch abort         */
    b       _hang           /* 0x10  data abort             */
    b       _hang           /* 0x14  reserved               */
    b       _hang           /* 0x18  IRQ                    */
    b       _hang           /* 0x1C  FIQ                    */

    .section .text.startup, "ax"
    .global _start
    .type   _start, %function
_start:
    /* SVC mode, IRQ and FIQ masked. */
    msr     cpsr_c, #0xD3

    /* Point VBAR at our table. Default is 0, which is correct for the
     * OCM build, but the DDR build links elsewhere and the BootROM's
     * value would be stale. */
    ldr     r0, =_vectors
    mcr     p15, 0, r0, c12, c0, 0

    /* SCTLR: clear M (MMU), C (D-cache), I (I-cache). */
    mrc     p15, 0, r0, c1, c0, 0
    bic     r0, r0, #(1 << 0)
    bic     r0, r0, #(1 << 2)
    bic     r0, r0, #(1 << 12)
    mcr     p15, 0, r0, c1, c0, 0
    isb

    /* Invalidate what the BootROM left behind. */
    mov     r0, #0
    mcr     p15, 0, r0, c7, c5, 0       /* ICIALLU */
    mcr     p15, 0, r0, c8, c7, 0       /* TLBIALL */
    dsb
    isb

    ldr     sp, =__stack_top

    /* Clear .bss. The linker script aligns both ends to 8, so a word
     * loop cannot overrun. */
    ldr     r0, =__bss_start
    ldr     r1, =__bss_end
    mov     r2, #0
1:  cmp     r0, r1
    strlo   r2, [r0], #4
    blo     1b

    /* PS bring-up. Both return a status; keep them for _cstart to
     * report, because at this point there is no console to print on --
     * ps7_init is what makes the UART work. */
    bl      ps7_init
    mov     r4, r0
    bl      ps7_post_config
    mov     r5, r0

    mov     r0, r4
    mov     r1, r5
    bl      _cstart

    .global _hang
    .type   _hang, %function
_hang:
    wfi
    b       _hang

    .size   _start, . - _start
