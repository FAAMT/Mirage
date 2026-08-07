/* =====================================================================
 * zynq.ld -- one script, two homes
 *
 * LOAD_BASE and STACK_SIZE arrive from the Makefile via --defsym, so the
 * OCM and DDR builds share a single layout instead of drifting apart as
 * two near-identical scripts.
 *
 * The default is OCM at 0x00000000, and that is a deliberate choice for
 * bring-up: the on-chip RAM is alive out of reset, so the smoke test has
 * no dependency on DDR training having succeeded. If ps7_init's DDR
 * sequence fails, an OCM-resident test still runs and still tells you
 * whether the accelerator answers -- which is the question being asked.
 * A DDR-resident test would simply not load.
 *
 * OCM is 192 KB mapped low (three of four banks; the fourth sits at
 * 0xFFFF0000). smoke.c's statics come to about 2.2 KB and ps7_init's
 * register tables to a few tens of KB, so the fit is comfortable. Should
 * an app outgrow it, switch with:  make REGION=ddr
 * ===================================================================== */
ENTRY(_start)

SECTIONS
{
    . = DEFINED(LOAD_BASE) ? LOAD_BASE : 0x00000000;

    /* VBAR requires 32-byte alignment, and crt0 points it here. */
    .vectors : ALIGN(32) {
        KEEP(*(.vectors))
    }

    .text : ALIGN(8) {
        *(.text.startup)
        *(.text .text.*)
        *(.glue_7) *(.glue_7t)
        *(.rodata .rodata.*)
        . = ALIGN(8);
    }

    /* Loaded straight to its run address over JTAG, so LMA == VMA and
     * crt0 has no .data copy loop to get wrong. */
    .data : ALIGN(8) {
        *(.data .data.*)
        . = ALIGN(8);
    }

    .bss (NOLOAD) : ALIGN(8) {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    }

    . = ALIGN(16);
    __stack_bottom = .;
    . = . + (DEFINED(STACK_SIZE) ? STACK_SIZE : 0x4000);
    __stack_top = .;

    _end = .;

    /* No unwinder, no exceptions, and nothing reads the build notes on a
     * board with no filesystem. */
    /DISCARD/ : {
        *(.ARM.exidx*)
        *(.ARM.extab*)
        *(.comment)
        *(.note.*)
    }
}
