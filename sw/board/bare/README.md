# bare — running smoke.c on the Cora Z7 without Vitis

A standalone platform for the Zynq-7007S, assembled from what the XSA
already contains. Builds `sw/board/smoke.c` into an ELF, programs the
bitstream, and runs it over JTAG.

## Why not Vitis

Vitis 2022.2 is 60–100 GB and does not support Ubuntu 24.04, and this
machine has 7.7 GB of RAM. That is a lot of friction for what turns out
to be a small job — because the XSA already carries the part that looks
hard:

| what a Vitis platform generates | where it comes from here |
|---|---|
| `ps7_init.c/h` — PLLs, MIO, DDR, FCLK | `ps7_init_gpl.c`, extracted from the XSA |
| linker script | `zynq.ld`, 60 lines |
| `boot.S` / crt0 | `crt0.S`, 40 instructions |
| `xil_printf` | `uart.c`, a real subset |
| BSP drivers | not used — the driver talks to one AXI slave |

`ps7_init_gpl.c` is 12,500 lines of register writes that depend on
nothing: it defines its own `mask_write()` and `mask_poll()` as volatile
pointer accesses. Its `#include "xil_io.h"` is vestigial, which is why
`xil_io.h` here is an empty stub. That single fact is what makes this
approach work at all.

**This does not replace Vivado.** Changing the hardware still needs a
synthesis run. This builds software against a bitstream that exists.

## Build

```sh
make                    # build/mirage_smoke.elf
make REGION=ddr         # if it ever outgrows 192 KB of OCM
```

Needs `gcc-arm-none-eabi`. The XSA and `.bit` are read from
`build_cora/`, which is gitignored — regenerate with `tcl/bd_cora.tcl`
if they are missing.

### Where it runs, and why that matters

Default is **OCM at 0x00000000** — on-chip RAM, alive out of reset. The
smoke test then has no dependency on DDR training having worked. If
ps7_init's DDR sequence fails, an OCM-resident test still runs and still
answers the question being asked, which is *does the accelerator
respond*. A DDR-resident test would simply fail to load, and a load
failure and a dead accelerator look far too similar at 2am.

## Run

```sh
make load               # bitstream + ELF over JTAG, then resume
make console            # 115200 8N1 on /dev/ttyUSB1
```

Open the console first — the test prints and finishes in milliseconds.

Both need the board's permissions set up once:

```sh
sudo usermod -aG dialout $USER          # then log out and back in
printf 'SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", MODE="0660", GROUP="plugdev", TAG+="uaccess"\n' \
  | sudo tee /etc/udev/rules.d/52-digilent.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## What a pass looks like

```
=== Mirage bring-up ===
ID       : 0x4d495247  OK
loopback : OK (0 of 64 bytes wrong)
gemm     : OK  expected 64 in all 32 lanes

ALL PASS -- datapath confirmed on silicon.
```

The three stages are ordered so a failure localises itself: ID means
clock or address, loopback means the AXI shim, GEMM means the datapath.

## The startup sequence, and the two parts of it that are load-bearing

`crt0.S` → `ps7_init()` → `ps7_post_config()` → `_cstart()` → `main()`

**Caches stay off.** The accelerator is memory-mapped and several
registers have side effects on access — `ACT_DATA` auto-increments its
pointer every time it is touched. With the D-cache on and no MMU marking
0x43C00000 as Device memory, a write could sit in a dirty line and never
reach the PL, and two reads could fold into one. `volatile` in
`io_mmio.c` constrains the compiler; it says nothing about the cache.

**`ps7_post_config()` is not optional.** It raises the PS–PL level
shifters. Until it runs, AXI transactions do not reach the fabric at
all. On a normal boot the FSBL calls it after loading the bitstream;
there is no FSBL here, so `crt0.S` calls it directly.

## smoke.c is not modified

`bare/xil_printf.h` supplies the name `smoke.c` already expects, rather
than `smoke.c` being edited to suit this build. It is verified code —
editing it would mean the thing running on the board is no longer the
thing that was checked. Same for `mirage.c` and `io_mmio.c`, which
compile here unchanged.

## What is verified

The JTAG chain is confirmed on hardware. At 5 MHz, both TAPs match and
both cores enumerate:

```
zynq_pl.bs   0x13723093    part 0x3723 = XC7Z007S, revision 1
zynq.cpu     0x4ba00477    ARM DAP
zynq.cpu0 / zynq.cpu1      6 breakpoints, 4 watchpoints each
```

The Digilent HS1 layout mask was right first time; that had been the one
guess in this directory.

`0x13723093` is worth writing down, because openocd 0.12.0's stock
`zynq_7000.cfg` does **not** list it — the XC7Z007S is missing from its
table of four. Sourcing that file yields "UNEXPECTED", four errors, and
"Bypassing JTAG setup events due to errors". It limps on with setup
events skipped, which is the kind of half-working that costs an evening
later. `cora.cfg` therefore declares the chain itself, with the stock
config's remaining contents reproduced verbatim.

The build is confirmed too: 22 KB, pure ARM (zero Thumb mapping symbols
— the thumb-libc objects the linker warned about are dropped by
`--gc-sections`), soft-float through libgcc's `__aeabi_d*` helpers,
vectors at `0x0`, `.bss` correctly beyond FileSiz, stack top at
`0x9600`.

What remains untested is the only thing that needs the accelerator:
whether it answers.
