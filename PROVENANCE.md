# PROVENANCE — every change from the code as received

This file exists to answer one question without anyone having to read a diff:
**what did we change, and why.**

The code was received as the `Mirage` repository at
`agentic-chip-design/Mirage`, branch `main`, commit `ed7dd77`
("Restarting MIRAGE accelerator..."). Nothing in that repository has been
modified — it is still at `ed7dd77` and is not a submodule of this package.
Every change described here was made inside the packaged working copy at
`~/Desktop/mirage-vivado-2026-07-29`, which has its own git history.

Changes fall into two clearly separated layers, and the separation is the
important part:

| Layer | What | Where it lives | Purpose |
|---|---|---|---|
| **A** | Minimal patches to the received RTL | `rtl/`, `rtl_excluded/` | Make it synthesise **at all**, so it could be *measured* rather than estimated |
| **B** | A ground-up re-architecture | `rtl_v2/` | Fix what the measurement showed |

Layer A never changes behaviour on purpose. Layer B is a new design.
**The received RTL is still present, unmodified apart from Layer A, and is
still buildable.** v2 did not overwrite it.

A byte-identical frozen copy of the v1 tree also exists outside this package
at `~/Desktop/mirage-v1-FROZEN-2026-07-30` (read-only, its own local git
repo, no remote).

---

## Layer A — the patches that made v1 measurable

Five files, ~237 changed lines. Every original is preserved in `reference/`.

The received design had never been through Vivado. It analysed under GHDL and
stopped at the first `synth_design`. These patches are the shortest path to a
number, and each is annotated in place with the Vivado error it clears.

### A1. Readable `out` ports — `ffn.vhd`, `qkv_score_calc.vhd`

*Vivado error:* `[Synth 8-10557]`, 9 sites in `ffn` alone.

Both architectures read their own `out` ports in FSM conditions. That is legal
VHDL-2008 and GHDL accepts it; Vivado's synthesiser does not implement it.

*Patch:* add an internal `_i` signal, drive and read that, mirror it to the
port. The entity interface is unchanged, so no instantiation or testbench is
affected. Vivado's own suggestion — change the port to `buffer` mode — was
rejected because it ripples into every instantiation.

Originals: `reference/ffn.readable-out-ports.vhd`,
`reference/qkv_score_calc.readable-out-ports.vhd`. A third instance in
`weight_stream_ctrl.vhd` was found and preserved
(`reference/weight_stream_ctrl.readable-out-ports.vhd`) but that module is
instantiated by nothing, so it was left alone.

### A2. Double cast — `qk_score_bram.vhd`

One line. `std_logic_vector(bram_score(...))` where `bram_score` is already
`std_logic_vector`. Patch in `patches/qk_score_bram-drop-slv-cast.patch`,
original in `reference/qk_score_bram.unpatched.vhd`.

### A3. Missing type alias — `types_pkg.vhd`

Three lines. Upstream renamed `mac_array` → `int8_gemm_array` in the package
but never updated `ffn`, `softmax` or `multi_head`, which still reference the
old name. A `subtype mac_array is int8_gemm_array;` alias restores it. Without
this, three of the nine roots do not analyse.

This is upstream's own inconsistency, not something introduced here.

### A4. The latched accumulator — `gemm_array.vhd`

This is the one Layer A change that alters behaviour, and it was made
deliberately, to test a hypothesis.

The accumulator process was combinational, read a signal (`computing`) that
was not in its sensitivity list, and had a commented-out `rising_edge(clk)`
that the author had evidently intended:

```vhdl
accum_proc : process (input_a, input_b, start)
begin
  -- if rising_edge(clk) then
  if computing = '1' then
    ...
  -- end if;
end if;
```

`if computing = '1'` with no `else`, in a combinational process, is a
transparent latch. Vivado emitted 12 × `[Synth 8-327]` and built 240 `LD`
cells. The 60 int8 multiplies fed that latch, and DSP48E1 inference requires
a multiply to feed a **clocked** register — so all 60 multiplies went to
fabric.

*Patch:* uncomment the clock, as the author intended. Costs one pipeline
cycle. Original at `reference/gemm_array.latched-accum.vhd`.

**Measured result: LUT 6,818 → 4,850 (−29%), latches 240 → 0, DSP 0 → 0.**

That last number is the point. Clocking the accumulator removed the latches
and did *not* produce a single DSP, which **disproved the latch theory** and
sent the investigation to the real cause. A patch that changes behaviour is
justified when it is the experiment.

### A5. Excluded, not patched — `softmax.vhd`

Moved to `rtl_excluded/` untouched. It is a half-finished rewrite that
produces 49 analysis errors and is superseded by
`rtl/outputs/smx_core|smx_cache|smx_v_calc`. Excluding it is a statement about
which file is live, not a modification.

---

## What Layer A measured

Vivado 2022.2, `xc7z007sclg400-1`, out-of-context, 5 of 9 roots (the other
four do not have a synthesisable top):

| | v1 measured | Device | Plan estimate (`reference/w11-resource-map.md`) |
|---|---|---|---|
| LUT | **18,182** | 14,400 | "comfortably under 30%" |
| BRAM | **0** | 50 | 16–20 |
| DSP48E1 | **2** | 66 | 60 |

**126% of the device, on 5 of 9 modules, with no top level.**
`qk_score_bram` alone spent **11,587 LUT — 80% of the device — on 2,688 bits
of storage.**

There is no top-level `encoder_layer` entity anywhere in the received code,
although `outputs/synth.tcl` calls `synth_design -top encoder_layer`. That is
why v1 could only ever be sized per module.

### The two root causes, both measured, not inferred

**1. 60-wide array reads.** Every weight array is read inside a
`for i in 0 to TILE_H-1 / for j in 0 to TILE_W-1` nest to fill the 12×5 MAC
tile in one cycle — **60 concurrent read ports on one array**. BRAM offers 2,
LUTRAM 16. No Xilinx primitive offers 60, so Vivado degraded every memory to
registers plus a combinational mux tree.

Total declared memory is **3,903,904 bits = 212%** of the device's 1,843,200.
Two independent problems: capacity (464 KB vs 225 KB) and bandwidth.

*Danger noted for anyone who returns to this:* below Vivado's default
`dissolveMemorySizeLimit` (65,536 bits), arrays degrade to flip-flops
**silently**. `qkv_proj_bram` became 49,152 FF on a 28,800 FF device — a
plausible-looking number that means nothing. Never raise that limit; it makes
Vivado grind instead of erroring.

**2. int8 × int8 is below Vivado's DSP inference threshold.** Proven by
contrast rather than by argument: `smx_v_calc`'s 32×8 multiply *did* infer 2
DSPs — the only 2 in the whole design — while every 8×8 multiply went to
fabric. This is why A4 removing all 240 latches still yielded zero DSPs.

### Three integration gaps, all downstream of the memory question

1. `multi_head` and `qkv_proj_calc` have **no `output_d` port** — the MAC
   result has nowhere to land.
2. `taylor_exp.data_out` is 12×21 = 252 bits; `ffn.data_in` is 16×17 = 272
   bits. The `16`/`17` in `ffn` are stale against the package constants.
3. Four MAC clients, one `gemm_array`, **no arbiter**.

`weight_stream_ctrl` was written for the DDR-streaming plan that the W9/W11
reports describe, but **nothing instantiates it**.

---

## Layer B — the v2 re-architecture

`rtl_v2/`, top `encoder_layer`. New code; no v1 file is edited by it.

v2 keeps v1's **logical** chain exactly as the plan defines it —
`QKV projection → QKᵀ → softmax → ×V → output projection → FFN → residual`.
What changes is how the arithmetic and the memories are realised.

### B1. Output-stationary MAC array — `prim/mac_lane.vhd`, `core/mac_array.vhd`

Lane *j* owns output element *j* and accumulates over the reduction dimension
serially. So the activation is **broadcast** (1 byte/cycle) and weights are 32
**consecutive** bytes/cycle — 8 BRAM banks × 32 bits, two ports each. Never 60.

This is what makes the memories inferable. It is the single change everything
else follows from.

### B2. Every multiply forced into a DSP, and the accumulator kept out

`mac_lane` is a canonical DSP48E1 MACC. The counter-intuitive part, and the
reason it took three Vivado runs: **the binding constraint was LUTs, not
DSPs**, so spending a whole DSP48E1 on an int8 multiply is correct here.

The attribute must be scoped **per signal**:

```vhdl
attribute use_dsp of mreg : signal is "yes";   -- multiply  -> DSP48E1
attribute use_dsp of preg : signal is "no";    -- accumulate -> fabric
```

`use_dsp` on an *architecture* applies to **every** arithmetic operation in
it, so it demanded a DSP for the accumulator as well: 32×2 + 2 = **66 DSP,
100% of the device**. Scoping it per signal is what took DSP to 34.

### B3. One canonical memory template — `prim/bram_sdp.vhd`

**No reset in the process.** BRAM contents cannot be reset, and that alone
disqualified every v1 array. One template, used everywhere.

### B4. One MAC client — `core/gemm_sequencer.vhd`

Every matmul in the encoder is the same triple loop with different bases and
lengths. So the four-client arbiter gap and the two missing writeback paths
**stop existing** rather than getting solved.

### B5. Softmax with no divider and no materialised probabilities

`core/softmax_unit.vhd`, `core/exp_rom.vhd`, `core/recip_rom.vhd`.
exp is a BRAM table, not a polynomial — a degree-2 Horner form was tried first
and measured **0.98 absolute error** over [−8, 0], because a parabola turns
back upward and cannot track exp. One reciprocal lookup scales each output
vector, bounded to [127, 2032] because the argmax always contributes exp(0).

### B6. A top level that exists — `top/encoder_layer.vhd`

Fully connected: every submodule port is driven or consumed. That is not
tidiness. v1's `smx_core` measured **smaller than its own children** (1,668
LUT vs 3,999) because its `score_row` input was unconnected and Vivado deleted
the logic behind it. An unconnected input is not a stub; it is a request to
delete the block.

### B7. Gates that did not exist before — `tools/`

| Tool | Catches |
|---|---|
| `vivado_lint.sh` | Vivado's readable-`out`-port errors, via `ghdl -a --std=93` |
| `memory_audit.py` | BRAM-inferability per array (**blind to v2** — see below) |
| `v2_budget.py` | The structural invariant: no memory or signal×signal multiply outside the primitives |
| `scrub_check.sh` | Hostnames, usernames and paths before anything is pushed |
| `pkg/addr_math_tb.vhd` | Address-generator **values and injectivity**, not widths |
| `pkg/sat8_tb.vhd` | Saturating narrow, 6,149 inputs |
| `pkg/exp_idx_tb.vhd` | exp index vs the integer form it replaced, 37,215 inputs |
| `core/mac_array_tb.vhd` | MAC array including a latency pin derived from `MAC_LAT` |

`memory_audit.py` reports a **false all-clear** on v2 (generics plus
`constant` ROMs defeat it). It is kept for v1 only.

---

## The bugs found during Layer B, and what each cost

These are recorded because each one cost a Vivado round-trip, and none is
visible from a file tree or a commit log.

### The `numeric_std` multiply trap — silent, total, and accepted by every tool

```vhdl
function "*" (L : UNSIGNED; R : NATURAL) return UNSIGNED
  is  return L * TO_UNSIGNED(R, L'LENGTH);
```

The natural takes the **left** operand's width. So `head * HEAD_DIM` with a
4-bit `head` is `TO_UNSIGNED(64, 4)` = **0**, and the product is 0 for *every*
head. GHDL analysis, GHDL elaboration and Vivado synthesis all accept it.
Measured: every head addressed head 0's K and V.

The same operator grows the result to `L'LENGTH + R'LENGTH`, so chained
multiplies hit 44 bits into an 11-bit target — that form *is* a hard error
(`Synth 8-690`) and it **masked** the silent one.

Fix: `mirage_v2_pkg.addr_mul` — widen, then multiply, then trim, with an
assert on the stride. Six call sites in `layer_ctrl` converted.

**Why the gate missed it:** `v2_budget.py` whitelisted "signal × power-of-two
constant" as a harmless wired shift, which is *exactly* the broken shape.
Gates that reason about **intent** miss operators that are wrong for reasons
of **typing**.

### `rom_style` on a `constant` is silently ignored

`Synth 8-5733`. ROMs must be `signal` arrays with initial values (UG901
template), or 36 Kb of table lands in LUTs.

### `integer` arithmetic in the datapath

VHDL's `integer` is 32 bits, so 21-bit arithmetic ran at 32, and `/ 2**k` is a
**truncating division** carrying sign-correction logic, not a shift. Run 8's
worst path was 17 CARRY4 levels from the score BRAM to `rom_addr`. Replaced by
`exp_index`, tested against the integer form it replaced on 37,215 inputs.

### A 48-bit clamp is two carry chains

`> 127 / < −128` on a wide accumulator synthesises to two carry chains.
`sat8` tests whether the high bits are all copies of the sign bit — an
OR-reduce and an AND-reduce, about two LUT levels.

---

## Measured trajectory

Vivado 2022.2, `xc7z007sclg400-1`, full flow to routing.

| | v1 | run 6 | run 8 | run 9 | run 10 | **run 11** | Budget |
|---|---|---|---|---|---|---|---|
| LUT | 18,182 (**126%**) | 1,969 | 2,303 | 2,203 (15%) | 2,337 | **2,150** (15%) | 12,960 |
| FF | 385 | 466 | 1,570 | 1,570 | 1,796 | **1,803** | 28,800 |
| DSP48E1 | 2 | **66** (100%) | 34 | 34 (52%) | 34 | **34** (52%) | 59 |
| RAMB36 | **0** | 35.5 | 35.5 | 35.5 (71%) | 36.5 | **36.5** (73%) | 47 |
| WNS | — | −9.485 | −6.226 | +0.058 ns | *n/a* | **+0.252 ns** | ≥ 0 |

Runs 9 and 11: *"All user specified timing constraints are met."* Run 11 has 0
failing endpoints of 6,409, hold met at +0.046 ns, 0 errors.

Run 10 is the Layer C RTL and reports no WNS because implementation never ran:
`synth_design` had never been given `-mode out_of_context`, so Vivado treated
`encoder_layer` as a whole chip and demanded a package pin per port bit. The
job interface takes the top level from 16 ports to 24, or 171 bits, against
100 sites on `clg400` — `[Place 30-58] IO placement is infeasible`. A flow
defect, not a design defect: this block is an AXI slave inside a Zynq block
design, where those ports are internal nets. Run 11 is the same netlist with
the flag set.

Run 11's critical path is unchanged from run 9 — `recip/q_reg` →
`seq/wb2_prod_reg/PCIN`, 8.264 ns of data path, 82.9% logic and 17.1% route
over two logic levels:

| stage | delay | share |
|---|---|---|
| RAMB36E1 clk→`DOPADOP` | 2.454 ns | 30% |
| route `rc_dout[16]` | 0.778 ns | |
| LUT3 — the `mac_scale` mux | **0.149 ns** | 2% |
| route `seq/scale[16]` | 0.637 ns | |
| DSP48E1 `A[16]`→`PCOUT` | 4.244 ns | 51% |
| DSP48E1 setup on `PCIN` | 1.400 ns | 17% |

The `mac_scale` mux was the change flagged as the timing risk of Layer C. It
costs 0.149 ns. **81% of this path is two hard-block propagation delays** — a
BRAM read and a 32×19 multiply crossing a DSP cascade — which no RTL
restructuring reduces. 200 MHz would need the DSP `MREG` and the `recip_rom`
output register, roughly 2 ns each, at a pipeline cycle apiece.

Per-module numbers confirm the intent rather than just the totals:
`mac_array` = 1,024 LUT (exactly 32 lanes × 32-bit fabric accumulators);
`act_mem` and `score_mem` = **0 LUT**, entirely in BRAM.

---

## Commit index

| Commit | What |
|---|---|
| `14c63b6` | v1 baseline — Layer A applied, measured state before v2 |
| `8f03703` | v2: re-architect for DSP and BRAM utilisation |
| `8fa8746` | Drop Vivado webtalk telemetry before publishing |
| `0aa9aad` | Fix `layer_ctrl` address arithmetic and ROM inference (run 5) |
| `3d1518c` | Fix DSP doubling and pipeline the writeback (run 6) |
| `35de294` | Confirm the timing diagnosis, cut two paths, add a scrub gate |
| `b1fb28c` | Remove a literal hostname from `scrub_check.sh`, close its blind spot |
| `b8409ef` | Scope `use_dsp` per signal: the attribute was the bug, not the pipeline |
| `5cc6fed` | Cut the softmax exp-index path; all three resource targets met |
| `ba8c15a` | Run 9: all user specified timing constraints are met |
| `670be3c` | Layer C — 13 defects fixed, four verification rungs added |
| `e8abe50` | Run the block out of context, so the placer stops asking for pins |

---

## What is NOT changed, and what is NOT done

**Not changed:** the upstream `Mirage` repository, at `ed7dd77`. The v1 RTL's
logic, apart from A4. The plan's logical dataflow — v2 implements the same
chain.

---

## Layer C — numerical correctness (2026-07-31)

Run 9 met every resource and timing target and **still computed wrong
numbers**; synthesis and static timing analysis cannot see a numerical error.
Layer C is what closed that. `TEST_PLAN.html` is the plan; `V2_DESIGN.md` has
the engineering detail. In outline:

**Thirteen defects found and fixed**, across `stage_engine` (5),
`layer_ctrl` (4), `gemm_sequencer` (2), `softmax_unit` (1) and
`encoder_layer` (1). Eight are alignment errors between a strobe and the data
it labels. None changes a width, so nothing warned.

**Two structural changes.** `layer_ctrl` became a job dispatcher rather than a
fixed op walker — the walker could not be made correct, because it assumed
weights were resident and no weight matrix in this model fits the 32 KB
window. `gemm_sequencer` gained reduction chunking (`acc_clr` / `acc_wb`) and a
programmable requantisation multiplier.

**New verification, all of it new to this package:** `sim/build.sh`,
`sim/stage_engine_tb.vhd`, `sim/softmax_unit_tb.vhd`,
`sim/encoder_layer_tb.vhd` (a command interpreter, so the PS model drives the
real RTL), and `model_ref/` — a float32 MarianMT reference in numpy with no
torch, a bit-exact int8 model of the datapath, and the co-simulation drivers.

**Result:** "Hello" through six encoder layers on the RTL, bit-exact against
the model, decoding to token 1466 مرحبا — identical to the float32 reference.
With the decoder on the fabric too, all 12 layers and 13,584 jobs, the output
ids match the reference exactly.

**Timing, now measured.** Layer C's timing claims were arguments when written.
Run 11 settled them: **WNS +0.252 ns**, better than run 9's +0.058 despite the
added job interface. Two variables moved at once — the RTL and the
out-of-context flag — so the gain cannot be attributed cleanly to either; the
critical path is the identical net in both runs and its logic is unchanged,
which makes placement freedom the likely source. The `mac_scale` mux singled
out as the risk measures 0.149 ns.

---

## Layer D — the PS interface (in progress, 2026-07-31)

Everything above characterises `encoder_layer` **out of context**: a block
with 24 raw ports and nothing attached. Layer D attaches it.

**Not yet started at the time of writing.** Scope, in order: an AXI4-Lite
slave over the existing config register file; an AXI-Stream path on a PS HP
port for the weight stream; a Cora Z7-07S block design; bare-metal bring-up
replaying the verified job stream over JTAG; then a PetaLinux image for the
full translation loop.

The bring-up asset that matters already exists: `sim/` holds a 2,352-job
command stream that GHDL proved bit-exact. Replaying it on hardware and
comparing byte-for-byte means any hardware mismatch localises to the wrapper
or the clocking, never to the datapath.

Out-of-context numbers do not survive integration unchanged. Run 11's +0.252 ns
excludes the interconnect delay the PS-side ports will pick up, so timing must
be re-measured in context once the block design exists.
