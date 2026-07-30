# Known issues

**Sections 7 and 8 are the ones that matter most.** §8 is why the measured DSP
count is 0 rather than the predicted 60.

**Section 7 is the memory blocker.** The design declares 212% of the
device's BRAM and reads its weight arrays 60-wide, which no FPGA memory
primitive supports. Until that is resolved, no whole-design resource number
exists — not because of tooling, but because the design cannot be mapped.

Section 0 is the first Vivado 2022.2 blocker, now fixed. Sections 1–3 are analysis
blockers. Sections 4–6 are the integration gaps that
decide how far a top-level can honestly be built — read those before you trust
any whole-design number. Diagnostics are from GHDL 4.1.0 `--std=08`; Vivado's
analyser will reject the same constructs.

---

## 0. Readable `out` ports — FIXED 2026-07-29

The Vivado 2022.2 run in `runs/2026-07-29-vivado-2022.2/` failed with nine
instances of:

```
ERROR: [Synth 8-10557] cannot read from 'out' object 'clear_mac';
       use 'buffer' or 'inout' instead   [ffn.vhd:152]
```

Reading an `out` port is legal VHDL-2008 and GHDL accepts it. **Vivado's
synthesiser does not implement it.** This is a tool support gap, not a design
fault — the RTL is correct VHDL.

Three files were affected, 55 references across 8 ports:

| file | ports read as outputs | refs |
|---|---|---|
| `ffn.vhd` | `start_mac`, `clear_mac` | 33 |
| `qkv_score_calc.vhd` | `start_gemm`, `clear_gemm`, `qk_iter`, `tok_q_iter`, `tok_k_iter` | 50 |
| `weight_stream_ctrl.vhd` | `mem_addr` | 4 |

**Vivado only reported `ffn.vhd`** — it aborts after the first failing unit, so
the other two would have surfaced on the next run, and the one after that.

**Fix:** each port is driven and read through an internal `<name>_i` signal and
mirrored to the port with a concurrent assignment. Entity interfaces are
unchanged, so nothing that instantiates these modules is affected — including
the container and every testbench. Identical synthesised logic.

Vivado's own suggestion (`buffer` mode) was rejected: it changes the port mode,
which ripples into every instantiation and testbench.

Originals preserved as `reference/*.readable-out-ports.vhd`.

**Guard:** `tools/vivado_lint.sh` now catches this class locally in about a
second, via `ghdl -a --std=93` — VHDL-93 forbids readable out ports too.
Validated against Vivado's own output: the 93 pass flagged the same nine lines
in `ffn.vhd` that Vivado did. Run it before spending another Vivado cycle.

---

## 1. `qk_score_bram.vhd` — one-line type error, PATCH APPLIED

```
qk_score_bram.vhd:63:49: error: can't match type conversion with type array type "UNRESOLVED_SIGNED"
    bram_data_out(i) <= std_logic_vector(bram_score(addr_off_0 + i));
```

`score_io_type` elements are `signed` (`types_pkg.vhd:48`), so casting the
source to `std_logic_vector` before assigning to a `signed` target is a type
error. The cast is spurious — dropping it is the whole fix.

**This module matters for the resource numbers.** It is a `4×4×8`-element score
BRAM and contributes directly to the RAMB36 count the W11 report puts at ~84%.
Worth enabling.

**This is now applied.** The patched file is in `rtl/mirage/` and in the synth
run; it analyses and elaborates cleanly. The untouched original is kept at
`reference/qk_score_bram.unpatched.vhd`, and the patch itself at
`patches/qk_score_bram-drop-slv-cast.patch`, so it is fully reversible:

```bash
cp reference/qk_score_bram.unpatched.vhd rtl/mirage/qk_score_bram.vhd
```

---

## 2. `softmax.vhd` — incomplete, do not try to fix here

49 analysis errors: `bram_qkv`, `bram_qkv_addr`, `offset`, and `mac_out_reg`
are used throughout the architecture but never declared, which then cascades
into operator-resolution failures on every expression touching them.

This is not a typo — it is a half-finished rewrite. The Mirage history says so
directly: `37b9584 "Last commit before full revamp of the multi head attention
core and the softmax core"`, then `ed7dd77 "Restarting MIRAGE accelerator..."`.

**It is also superseded.** The working softmax path is
`rtl/outputs/smx_core.vhd` + `smx_cache.vhd` + `smx_v_calc.vhd`, which are in
the run. Use those figures for the softmax block; ignore `softmax.vhd`.

One gap to be aware of: `smx_core.vhd:74` has the `taylor_exp` instantiation
**commented out**, so the synthesised `smx_core` contains the cache and the ×V
datapath but not the exponential. `taylor_exp` is synthesised separately in the
same run, so you have its cost — just remember that `smx_core` alone is not the
complete softmax, and adding the two is the closer estimate.

---

## 3. No top-level entity exists

`tb/system_tb.vhd` is the only thing resembling a system-level integration, and
it is a testbench. There is no `encoder_layer` entity anywhere in either repo —
which is why the original `outputs/synth.tcl` (kept at
`reference/synth.tcl.original`) could never have run: it calls
`synth_design -top encoder_layer`. Its own header admits as much
("upstream fixes required before this elaborates"), and
`reference/w11-postsynth-timing.md` is explicit that no synthesis was ever run.

`rtl/integration/mirage_sizing_container.vhd` stands in for it — see below for
exactly what it does and does not claim.

---

## 4. Two modules have no result-return port

`multi_head` and `qkv_proj_calc` both drive `input_a`/`input_b` at the MAC tile
and accept `done_mac`/`done_gemm` back — but neither entity has an `output_d`
input. The MAC result has nowhere to land. Compare `qkv_score_calc` and `ffn`,
which do carry `output_d`.

So two of the four MAC clients cannot be wired to `gemm_array` faithfully. Any
top-level that appears to connect them has had glue invented for it.

---

## 5. `taylor_exp` → `ffn` is a width mismatch

| signal | width |
|---|---|
| `taylor_exp.data_out` | `TILE_H * SCORE_BRAM_DATA_W` = 12 × 21 = **252** |
| `ffn.data_in` | `16 * DATA_W` = 16 × 17 = **272** |

The pairing is clearly intended — `taylor_exp.fetch_attn` matches
`ffn.fetch_attn` exactly — but the literal `16` and `DATA_W := 17` in `ffn` are
stale against the package's `TILE_H = 12` and `SCORE_BRAM_DATA_W = 21`. These
two ports do not connect as written. This is the W7 fix the phase reports refer
to; fixing it is a design decision (which constant is wrong?), not a packaging
one.

---

## 6. Four MAC clients, one `gemm_array`, no arbiter

`qkv_proj_calc`, `qkv_score_calc`, `multi_head`, and `ffn` all drive a MAC
interface. There is one `gemm_array` and nothing that arbitrates between them.
An arbiter is real design work with real resource cost, and inventing one here
would put LUTs in your utilisation report that the architecture has not
committed to.

---

## What the container does about all this

`rtl/integration/mirage_sizing_container.vhd` instantiates each **root** module
exactly once and wires only the connections the source already defines
unambiguously and type-exactly:

- `qkv_score_calc` ↔ `gemm_array` — the one complete MAC pairing (it has
  `output_d`)
- `qkv_score_calc.qk_iter/tok_q_iter/tok_k_iter` → `qk_score_bram` — name- and
  type-exact
- `taylor_exp.bram_rd/addr_off_0` → `qk_score_bram`, and
  `qk_score_bram.bram_data_out` → `taylor_exp.data_in` and `smx_core.score_row`
  — all `score_io_type`, an explicit pairing

Everything from sections 4–6 is brought out to a container port instead of
being wired. That is deliberate. Exposing a port in out-of-context mode
preserves the logic behind it; inventing a control path that never toggles lets
Vivado constant-propagate and **delete** the block, which under-reports — a
worse failure than over-reporting, because it looks like good news.

**What the container total is:** every root module present once, nothing
double-counted, nothing pruned. A sound upper bound and the best available
device figure.

**What it is not:** a functioning encoder layer. It has no arbiter, no width
adapters, and no result path for `multi_head`/`qkv_proj_calc`. Once those
design decisions are made, the real total will move — arbitration and adapters
add logic, shared-resource merging removes it.

---

## 7. Weight memories cannot be mapped — THE BLOCKING ISSUE

The second Vivado 2022.2 run
(`runs/2026-07-29-vivado-2022.2-run2/`) got past the readable-`out`-port fix and
died on:

```
ERROR: [Synth 8-3391] Unable to infer a block/distributed RAM for 'bram_qkv_reg'
  because the memory pattern used is not supported. Failed to dissolve the memory
  into bits because the number of bits (786440) is too large
WARNING: [Synth 8-4767] Trying to implement RAM 'bram_qkv_reg' in registers.
  1: RAM has too many ports (16). Maximum supported = 16.
```

`786440` identifies it exactly: `multi_head.vhd:43`,
`array (0 to BASE_OFFSET) of std_logic_vector(7 downto 0)` = 98,305 × 8.

### The pattern

Every weight array is read inside a `for i in 0 to TILE_H-1 / for j in 0 to
TILE_W-1` nest to fill the 12×5 MAC tile in one cycle. That is **60 concurrent
read ports on a single array**. BRAM provides 2. LUTRAM provides up to 16. No
memory primitive on any Xilinx device provides 60, so Vivado falls back to
registers.

`tools/memory_audit.py` classifies every array. Current state:

| module | array | bits | rd ports | verdict |
|---|---|---|---|---|
| `ffn` | `bram_hlw` | 1,048,576 | 60 | hard error |
| `ffn` | `bram_olw` | 1,048,576 | 60 | hard error |
| `multi_head` | `bram_qkv` | 786,440 | 60 | hard error |
| `qkv_proj_calc` | `bram_qkv` | 786,440 | 60 | hard error |
| `ffn` | `bram_proj` | 131,072 | 60 | hard error |
| `qkv_proj_bram` | `bram_qkv` | 49,152 | 120 | **quiet** → 49,152 FF |
| `ffn` | `bram_hl_in` | 16,384 | 60 | quiet → 16,384 FF |
| `ffn` | `bram_ol_in` | 8,192 | 60 | quiet → 8,192 FF |

Total declared memory: **3,903,904 bits = 212% of the device's 1,843,200**.

### Two failure modes, and the quiet one is worse

Above Vivado's default `dissolveMemorySizeLimit` (65,536 bits) you get the hard
error above. **Below it, Vivado silently converts the array into flip-flops and
carries on.** `qkv_proj_bram` becomes 49,152 FF on a 28,800 FF device — it will
report over 100% and it is not a logic cost, it is a memory that failed to
become BRAM. That number looks like data but is not.

### Do not use Vivado's suggested workaround

```
set_param synth.elaboration.rodinMoreOptions {rt::set_parameter dissolveMemorySizeLimit 786440}
```

This raises the dissolve limit so the largest array becomes 1,048,576 individual
flip-flops on a device with 28,800. It does not make the design fit. It makes
Vivado grind instead of erroring.

### Two independent problems

1. **Capacity.** 464 KB of weights against 225 KB of BRAM. `reference/w11-resource-map.md`
   already knew this — it marks the FFN hl/ol weights "**DOES NOT FIT** — DDR-streamed
   (W9 §3)". `weight_stream_ctrl` was written for exactly that job. **Nothing
   instantiates it.** The plan exists; the RTL still declares the weights resident.
2. **Bandwidth/pattern.** Even with capacity solved, a 12×5 int8 tile needs 60
   bytes per cycle. That requires roughly 30 dual-port BRAMs banked in parallel
   holding a working window, not one flat array. This is a real memory-hierarchy
   design, not a code fix.

### What was changed here, and what was not

Changed: `rtl/integration/mirage_sizing_subset.vhd` synthesises the 6 root
modules that *can* be mapped, so the run produces usable numbers instead of
nothing. `ffn`, `multi_head` and `qkv_proj_calc` appear as `BLOCKED` rows in the
summary — excluded, not silently dropped.

Not changed: the memory hierarchy. Banking the weights and wiring
`weight_stream_ctrl` is the core architecture of the accelerator, it changes
latency and the DDR interface, and the right shape depends on numbers we do not
have yet. That is a design decision, and it is now clearly the dominant one —
ahead of the MAC arbiter (§6) and the 252/272 precision question (§5), both of
which are downstream of how weights reach the tile.

---

## 8. `gemm_array` inferred 0 DSPs and a latch — FIXED 2026-07-30

Run 3 (`runs/2026-07-30-vivado-2022.2-run3/`) cleared phase 0 and produced the
first real numbers. `gemm_array`, the 12×5 int8 MAC tile, synthesised to:

| cell | count |
|---|---|
| LUT1–LUT6 | **6,818** |
| CARRY4 | 1,056 |
| FDCE (flip-flops) | 385 |
| **LD (transparent latches)** | **240** |
| **DSP48E1** | **0** |

`reference/w11-resource-map.md` predicted **60 DSPs** and "LUT/FF comfortably
under 30%". Measured: **zero DSPs and ~47% of device LUTs in one module.** No DSP
was inferred anywhere in the design.

### Root cause — one defect, both symptoms

```vhdl
accum_proc : process (input_a, input_b, start)   -- reads `computing`, which is
                                                 -- NOT in the sensitivity list
begin
  -- if rising_edge(clk) then                    -- commented out by the author
  if computing = '1' then                        -- no else -> retains value
      ... tmp_val := tmp_val + resize(input_b(i,j) * input_a(i,j), 32); ...
      accum_val(i) <= tmp_val;
  end if;
end process;
```

A conditional assignment with no `else` inside a **combinational** process makes
`accum_val` a transparent latch — Vivado emitted 12× `[Synth 8-327] inferring
latch for 'accum_val_reg[n]'` and 240 `LD` cells.

That latch is also why there are no DSPs. **DSP48E1 inference requires the
multiply to feed a clocked register.** Feeding a transparent latch instead, all
60 int8 multiplies were mapped to LUT and CARRY4 fabric.

The commented-out `-- if rising_edge(clk) then` shows this was meant to be
clocked. Two further tells: `computing` is read but missing from the sensitivity
list (simulation/synthesis mismatch), and `accum(i) <= accum(i) + accum_val(i)`
in the main process samples a transparently-updating latch — a race between latch
propagation and the clock edge, the classic works-in-simulation failure.

### Fix

`accum_proc` is now `process (clk)` with `if rising_edge(clk)`, exactly as the
commented-out line intended. Removes the latch and puts a register where DSP
inference wants one. Original at `reference/gemm_array.latched-accum.vhd`.

**This costs one pipeline cycle.** `accum_val` is now registered, so the
accumulate sees the previous cycle's product sum. The latched version had no such
delay — but it was a race, so it could not stay. If the MAC's cycle count is
load-bearing for `perf-report-v1`'s latency budget, that budget needs one extra
cycle per accumulate step.

**Run 4 verdict: the latch fix worked, the DSP hypothesis was WRONG.**

| | run 3 (latched) | run 4 (clocked) |
|---|---|---|
| Register as Latch | 240 | **0** |
| Register as Flip Flop | 385 | 625 |
| Slice LUTs | 6,818 | **4,850** (−29%) |
| DSP48E1 | 0 | **still 0** |

The 240 latches became 240 flip-flops (385 + 240 = 625, exactly) and LUTs fell
29%. But clocking the accumulator did **not** unlock DSP inference, so the stated
hypothesis is disproved.

**The actual reason there are no DSPs: an 8×8 multiply is below Vivado's DSP
inference threshold.** A DSP48E1 is 25×18; Vivado maps small multipliers to
fabric because it is cheaper than spending a DSP. Direct evidence from the same
run: `smx_v_calc` does `w_buf(v_key) * v_data` — 32×8, wide enough to cross the
threshold — and it got **2 DSPs**. `gemm_array`'s int8×int8 got none.

So W11's "60 DSP" estimate was wrong twice over. Vivado will not infer DSPs for
int8 unasked, *and* spending one DSP48E1 per int8 MAC would waste most of a
25×18 multiplier anyway. Getting int8 onto DSPs takes deliberate work: either
force it with a `use_dsp` attribute, or pack two int8 products per DSP using the
standard 25×18 packing trick. Both are design decisions with real trade-offs.

### Two smaller notes from the same run

**`quantizer` is empty, and that is correct.** `[Synth 8-3330] design quantizer
has an empty top module`, zero cells. It is a constant right-shift from 16 bits
to 8 — pure bit selection, no logic. Zero is the right answer, not a failure.

**`a_vec`/`w_mat` are dead.** `gemm_array`'s first process copies `input_a`/
`input_b` into them, but the multiply reads the ports directly. Harmless — Vivado
strips them — but it is why some of the 100 `[Synth 8-6014] unused sequential
element removed` messages appear.

### Run 3 also stalled, and the script now survives that

Run 3 stopped inside phase-1 `qkv_proj_bram` — the module §7 flags as degrading
to a 49,152-bit register file — and because the summary was written only at the
end, it produced **no table at all**, despite `gemm_array` and `quantizer` having
already succeeded. Two changes:

- phase 1 now runs cheap/informative modules first and the memory-degrading ones
  (`qkv_proj_bram`, `qkv_score_calc`) last
- `reports/summary.csv` is flushed after every module, so a stall or a kill
  leaves everything completed up to that point
