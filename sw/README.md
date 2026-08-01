# sw — bare-metal driver for the Mirage accelerator

A transliteration of `model_ref/mirage_int8.py`'s `Mirage` class onto the
AXI4-Lite register map. Same operations, same order, same register writes
— deliberately, because that sequence is the part that has to be right and
it has already been proven bit-exact against the RTL.

## Layout

| file | what it is |
|---|---|
| `mirage_regs.h` | the register map, transcribed from `mirage_axi_lite.vhd` and `layer_ctrl.vhd`. No logic. |
| `mirage.h` / `mirage.c` | the driver |
| `mirage_io.h` | the two calls the driver is built on |
| `io_mmio.c` | real backend — a volatile pointer at `0x43C00000` |
| `io_trace.c` | host backend — decodes register traffic into the reference command-file format |
| `board/smoke.c` | first thing to run on hardware |
| `test/` | the host checks |

## Checking it without hardware

```sh
make check
```

Builds the driver against the tracing backend, replays
`model_ref/test_ops.py`'s call sequence through it, and diffs the register
traffic against `sim/ops_cmd.txt` — the script that `test_ops.py` already
runs against the RTL in GHDL.

The trace backend **decodes writes** rather than being called at the
driver's own level. That distinction is the whole point: a wrong register
offset, a missing CFG write, weight words in the wrong order, or a bufsel
flip on the wrong side of a load all show up as a diff. A tracing hook
placed at the driver's abstraction level would agree with every one of
those mistakes.

Compared: all CFG/JOB/MARK/DUMP/DUMPS lines in full and in order, the
address of every ACT write, and the index of every WROW. Not compared: the
payload bytes, which come from a seeded Python RNG and are test data rather
than driver behaviour.

`make check` also cross-checks `mirage_split_scale` against the reference's
`split_scale` over `test/scale_vectors.txt`. That test exists because the
ops trace *cannot* catch requantise-arithmetic bugs — every multiplier in
`test_ops.py` is a power of two, so the scale lands on an exact integer and
the rounding mode never matters. Swapping `nearbyint` for `floor(0.5 + x)`
passes the entire ops check untouched and fails three of these vectors.

### What these checks are known to catch

Each was confirmed by mutating the driver and watching the check fail:

| mutation | caught by |
|---|---|
| bufsel bracketing inverted around a window load | control sequence |
| `TLAST` sent per row instead of per window | WROW indices |
| `CFG_SCALE` and `CFG_OBASE` write order swapped | control sequence |
| round-half-away-from-zero in `split_scale` | split_scale vectors |

### What they do not cover

The done-polling protocol and ISR handling produce no lines in the
reference script, so they are invisible to the trace. `sim/axi_lite_tb.vhd`
covers those against the real register file.

## On the board

Build `mirage.c`, `io_mmio.c` and `board/smoke.c` as a Vitis application
against the XSA that `tcl/bd_cora.tcl` writes. `smoke.c` checks the ID
register, then activation loopback, then an all-ones GEMM whose expected
output — 64 in every lane — follows from the reference arithmetic rather
than from having run it once.

## The two rules worth knowing before changing anything

**Poll `STATUS[1]`, never the fabric's `done`.** `done` is a one-cycle
pulse at 100 MHz. Software cannot observe it. `done_sticky` exists so that
a poll loop is a legitimate way to wait.

**Writes land in the buffer `bufsel` does *not* select.** So a caller that
loads and then computes from the same buffer must flip `bufsel` around the
load — fill with 1, compute with 0. Getting this backwards is
self-consistent, so it passes every test while silently removing the
ability to overlap load and compute, which on a design that is
weight-bandwidth bound about 4.5 : 1 is the largest speedup available. See
the note in `rtl_v2/core/weight_window.vhd`.
