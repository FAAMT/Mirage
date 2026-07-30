# Mirage v2 — re-architecture for DSP and BRAM utilisation

Written 2026-07-30. v1 is frozen read-only at `~/Desktop/mirage-v1-FROZEN-2026-07-30`
and as commit 1 of this folder's local git repo.

## Why v2 exists

v1 measured, on Vivado 2022.2 / xc7z007sclg400-1 (run 4, 5 of 9 roots):

| | measured | device | |
|---|---|---|---|
| LUT | **18,182** | 14,400 | **126%** |
| RAMB36 | **0** | 50 | 0% |
| DSP48E1 | **2** | 66 | 3% |
| top level | **none existed** | | |

Every memory became flip-flops plus a combinational mux tree, and every
int8 multiply became LUT fabric. The design was inverted: everything that
belonged in hard blocks was in soft logic, and the soft logic ran out.

Two root causes, both traced to measurements not guesses:

1. **A 12×5 MAC tile fed by 60-wide array reads.** BRAM has 2 read ports,
   LUTRAM 16. 60 is not implementable by any memory primitive on any
   device, so Vivado degraded every weight array to registers. On the big
   arrays that was a hard error (`Synth 8-3391`); on the small ones it
   quietly built a mux tree — `qk_score_bram` spent **11,587 LUT (80% of
   the device) on 2,688 bits of storage**.
2. **int8×int8 is below Vivado's DSP inference threshold.** A DSP48E1 is
   25×18, so Vivado maps small multipliers to fabric because it is cheaper
   in area. Proven by contrast in the same run: `smx_v_calc`'s 32×8
   multiply *did* infer 2 DSPs, the only two in the design.

## What v2 changes, and what it keeps

Kept, exactly as the MVP plan specifies: the logical chain
*qkv projection → QK^T → exp/softmax → ×V → output projection → FFN →
residual*, int8 quantisation, and the Helsinki-class dimensions.

Changed: how the arithmetic and the memories are realised.

### 1. Output-stationary MAC array

Lane *j* owns output element *j* and accumulates over the reduction
dimension serially. The activation is **broadcast**; only weights differ
per lane.

| | v1 | v2 |
|---|---|---|
| activations per cycle | 60, arbitrary addresses | **1**, broadcast |
| weights per cycle | 60, arbitrary addresses | **32, consecutive** |
| accumulator location | LUT fabric (and a latch) | **DSP48E1 P register** |

32 consecutive weight bytes is 8 BRAM banks × 32 bits with **one read
address per bank per cycle** — which is what BRAM can actually serve.

### 2. Every multiply forced into a DSP

`prim/mac_lane.vhd` is the canonical fully-pipelined DSP48E1 MACC
(A→areg, B→breg, M=areg*breg, P+=M) with `USE_DSP = "yes"` on the
architecture to override the small-multiplier heuristic. One DSP per int8
MAC underuses the 25×18 multiplier — deliberately. LUTs were at 126% and
DSPs at 3%; trading the exhausted resource for the abundant one is the
entire point.

Two-int8-per-DSP packing (operands at A[7:0] and A[24:17]) would halve the
DSP count and is **not** done: it requires hand-instantiating DSP48E1 and
abandons the inference template. Revisit only if the array must exceed 66
lanes.

### 3. Every memory through one canonical template

`prim/bram_sdp.vhd`, used for all of them. Four rules, each one a v1
failure:

- synchronous read (async read forces LUTRAM)
- **no reset anywhere in the process** — BRAM contents have no reset in
  silicon, and v1's arrays all had `<= (others => (others => '0'))` in a
  reset branch, which alone disqualified them
- exactly one read and one write address
- `ram_style = "block"` stated explicitly

### 4. One MAC client instead of four

v1 had four modules driving one `gemm_array` with no arbiter
(`KNOWN_ISSUES` §6), and two of them had no result-return port at all
(§4). v2 has one `gemm_sequencer`; every matmul is the same triple loop
with different bases. **Both problems stop existing rather than getting
solved.**

### 5. Softmax with no divider and no probabilities

`exp` is a **BRAM lookup table**, not a polynomial. v2 tried degree-2
Horner first and measured **0.98 absolute error** over [-8,0] — a parabola
turns back upward and cannot track exp. The table is one RAMB36, and
accuracy is reported as the metric that matters:

- max absolute error in a softmax probability: **0.0027 (0.27 pp)**
- rows where the argmax moved: **33 / 20,000**

Probabilities are never materialised. Unnormalised exp values feed the MAC
array as int8 activations; one `recip_rom` lookup scales the whole output
vector at the end, because every element of a (head, query) output shares
one exp-sum. The table is small and accurate because the argmax element
always contributes `exp(0)`, so the exp-sum is bounded to [127, 2032] —
worst relative error **0.012%**.

### 6. A top level that exists

`rtl_v2/top/encoder_layer.vhd`. v1's `synth.tcl` called
`synth_design -top encoder_layer` and no such entity existed anywhere.

It is **fully connected** — every submodule port driven or consumed. That
is not tidiness: v1's `smx_core` measured *smaller* than its own children
(1,668 LUT vs 3,999) because an unconnected input let Vivado delete the
logic behind it. An unconnected input is a request to delete the block.

## Resource budget

From `tools/v2_budget.py`, derived from each instantiation's generics:

| | count | device | % |
|---|---|---|---|
| weight_window (8 banks × 2 buffers) | 16 RAMB36 | | |
| act_mem (8b × 64 K) | 15 RAMB36 | | |
| score_mem, exp_rom, recip_rom | 4 RAMB36 | | |
| **RAMB36 total** | **35** | 50 | **70%** |
| mac_array (32 × mac_lane) | 32 DSP | | |
| softmax scale multiply | ~3 DSP | | |
| **DSP48E1 total** | **35** | 66 | **53%** |

Against the plan's exit criteria (<90% LUT, <95% BRAM): BRAM and DSP both
pass. **LUT is not predictable from the RTL** — all storage and arithmetic
now sit in hard blocks, so what remains is control logic, and only
synthesis can measure it. v1's 18,182 LUT is the number to beat.

The choice that made it fit: the FFN hidden region is sized for **one
token**, not all `MAX_TOK`. FFN output for a token reduces over that
token's hidden values only, so token-by-token scheduling keeps the region
at 2 KB instead of 32 KB.

## Verification status — read this before trusting anything

**Verified by simulation:**
- `mac_array` — all 32 lanes match a full-precision golden model over 64
  reduction steps (`rtl_v2/core/mac_array_tb.vhd`, self-checking, fails
  loudly). This found a real bug: holding `en` high during the pipeline
  flush re-accumulates the last operands, which broke 31 of 32 lanes.
- exp and reciprocal table accuracy, in the generators, against float.

**Verified structurally:**
- all 12 modules analyse and `encoder_layer` elaborates under GHDL 4.1.0
  `--std=08`
- **zero** readable-out-port sites under `--std=93` (the proxy that
  reproduced Vivado's `Synth 8-10557` exactly on v1)
- `tools/v2_budget.py` passes: no memory or signal×signal multiply outside
  the approved primitives
- resource budget arithmetic above

**NOT verified — and I cannot verify it here:**
- **DSP and BRAM inference.** GHDL does no technology mapping. Whether
  `USE_DSP="yes"` and `ram_style="block"` actually take can only be
  confirmed by Vivado. The synthesis script warns explicitly if DSP or
  BRAM comes back 0.
- **Timing.** No static timing analysis exists outside Vivado. The design
  is pipelined for it (DSP M register, no multipliers in address paths,
  BRAM read latency explicitly aligned) and constrained at the plan's
  100 MHz fallback, but the WNS is unmeasured.
- **End-to-end numerics** against `outputs/*_model.py`. The MAC array is
  verified; the assembled layer is not.

**Known incomplete:**
- **Residual + layer normalisation.** The plan's Week 8 item. `O_RESID` is
  a wired pass-through hook, not an implementation.
- **Requantisation shifts** (`RQ_PROJ`, `RQ_SCORE`, `RQ_ATTN`, `RQ_FFN` in
  `layer_ctrl`) are reasoned from `log2(k_len)` plus headroom, not
  calibrated against the reference models. They are named constants in one
  place precisely so calibration touches nothing else.
- **FFN hidden tiling.** `FFN_DIM = 2048` needs 64 column tiles but
  `n_coltile` is 5 bits (max 31). The chunking loop is described in
  `layer_ctrl`'s comments but not implemented — FFN currently issues one
  `COL_TILES`-sized chunk.
- **`stage_engine` multi-tile interaction.** The staging address generator
  handles both access patterns, but its interaction with the two-column-tile
  ×V op needs a directed test.

## Running it

```bash
./tools/v2_budget.py                    # structural rules + budget
cd <package root> && mkdir -p reports_v2
vivado -mode batch -source tcl/synth_v2.tcl \
       -log reports_v2/vivado.log -journal reports_v2/vivado.jou
```

Full flow: elaborate → synth → opt/place/route → timing. Reports land as
soon as they exist, so a stall still leaves everything up to that point.
`reports_v2/summary_v2.csv` carries the pass/fail line.

**The first two numbers to look at are DSP and RAMB36.** If either is 0,
the corresponding attribute did not take and nothing else in the report
matters yet.
