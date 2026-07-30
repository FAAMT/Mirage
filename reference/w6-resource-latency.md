# W6 — mha_core Integration: Functional Baseline + Resource/Latency Profile

- **Task:** `p1/w6` — mha_core integration
- **Method:** model-level functional baseline (6 tokens, bit-accurate stages) + static resource/latency estimation (no simulator/synthesis per run directive)
- **Date:** 2026-07-24
- **Artifacts:** `reports/w6-e2e-6token.txt` (6-token model run), model: `outputs/smx_e2e_model.py`

## Acceptance criterion (from task graph)

> mha_core functional-correctness baseline — functional sim for 1 sentence (≥6 tokens) across all heads passes vs software.

## 1. Integration status (inspection)

- `Mirage/modules/mha_core/mha_core.vhd` **does not exist**. The closest upstream top is `mha_core/multi_head.vhd` — an early **projection-only** engine: it tiles token×W_Q/K/V through the MAC but has no score/softmax/attention stages, no head loop (relies on external weight swapping), declares an unused `softmax` state, and routes projection results nowhere observable (`output_d` isn't even a port).
- The real integration chain is the W1–W5 path: `qkv_proj_calc` → `qkv_proj_bram` → `qkv_score_calc` → `qk_score_bram` → taylor_exp → smx_cache → smx_v_calc. Missing integration piece upstream: a top FSM sequencing these + the score-writeback finalization (W1).
- **Parameter gap:** reference `TOK_CONTEXT=4` is hard-coded in `types_pkg`; the ≥6-token acceptance needs the parameterization already used in the `outputs/` drafts (`MAX_TOK=16`).

## 2. Functional baseline — 6 tokens × 8 heads (model level)

`python3 outputs/smx_e2e_model.py 8 6` (full output: `reports/w6-e2e-6token.txt`), reference = float softmax(s/√64)@V, int8 output units:

| Scenario | as-coded RTL | with W5 fixes (`<<6` + range-reduced exp) |
|---|---|---|
| full-range int8 Q,K | worst 100.0 / RMS 6.88 | 0.00 / 0.00 |
| ternary Q,K (soft regime) | worst 73.8 / RMS 10.48 | **worst 3.98 / RMS 0.59** |

Baseline verdict: **passes vs software with the W3/W5 fixes applied** (≤4 int8 LSB worst-case); as-coded RTL fails. All 8 heads checked per run.

## 3. Resource profile (estimate by inspection; synthesis out of scope)

Target: Cora Z7-07S (XC7Z007S) — datasheet ≈ 14.4k LUTs, 28.8k FFs, **66 DSP48E1**, **50 RAMB36** (verify against current datasheet).

| Block | Resource | Estimate |
|---|---|---|
| gemm_array (12×5 int8 MAC) | DSP | 60 mults → **~60 DSP = 91% of device** — dominant risk (W5 note). One int8 mult per DSP48E1; consider time-multiplexing to 2×30 or LUT-based mults for headroom |
| W_Q/K/V (1 layer) | BRAM | 3×64×512 int8 = 98,304 B → **~24 RAMB36 (48%)** |
| qkv_proj_bram | BRAM | 6,144 × 16 b = 12 KB → ~3 RAMB36 |
| qk_score_bram + smx_cache | BRAM/regs | 128×21 b + 128×48 b → <1 RAMB36 |
| taylor_exp (with W3/W5 fixes) | DSP | z², z³, ×174763, ×185 → ~4 DSP |
| smx_v_calc divider | LUT | restoring divider ~SUM_W×3 LUTs ≈ 150 LUTs |
| FSMs/glue | LUT/FF | est. 2–4k LUTs total |

Headline: fits, but **DSP (91%) and BRAM (48% for one layer's weights) leave little headroom** — multi-layer weight streaming from DDR is mandatory, and DSP sharing should be decided before W7+.

## 4. Latency model (per layer, 6 tokens, 8 heads, 200 MHz / 5 ns)

Deterministic FSM estimates (same method as W2):

| Stage | Formula | Cycles | Time |
|---|---|---|---|
| QKV projection | 6 tok × 3 × 8 heads × (6 h-passes × 103 w-passes) × ~3 cyc | ≈ 267k | ≈ 1.33 ms |
| Q·Kᵀ scores | 6×6×8 scores × ~7 cyc (W2 model) | ≈ 2.0k | ≈ 10 µs |
| softmax (max+exp+sum) | 8 heads × 6 rows × ~30 cyc | ≈ 1.4k | ≈ 7 µs |
| normalize + A·V | 8 heads × 6 rows × (39 div + 64 MAC) cyc | ≈ 30k | ≈ 0.15 ms |
| **Total / layer** | | **≈ 300k** | **≈ 1.5 ms** |

- **Projection dominates (~89%)** — any speedup must come from wider GEMM tiling or head parallelism, not the softmax path.
- Timing-risk note (task): the combinational 60-lane quantizer + gemm input mux is the likely critical path; if slack < 0, insert the pipeline register between quantizer and gemm_array (1-cycle cost per pass, ~2% latency).

## 5. Profiling scripts (for when tools exist)

- Functional: `python3 outputs/smx_e2e_model.py <heads> <tokens>` (deterministic, seeded).
- Sim: `cd Mirage/modules/mha_core/mha_core && bash run_sim.sh` (ghdl ≥ 4.1).
- Synth/P&R profiling (skeleton, requires Vivado + human): `vivado -mode batch -source synth.tcl` with `report_utilization`/`report_timing_summary` post-route → append to this report. Not run this run (no synthesis per contract).

## Disposition

Functional-correctness baseline established at model level for ≥6 tokens across all heads: PASS with the W3/W5 fixes (worst 3.98 int8 LSB). Resource/latency profile documented: ~60 DSP (91%) and ~24 RAMB36 (48%) are the binding constraints; ~1.5 ms/layer at 200 MHz, projection-dominated. Upstream `mha_core.vhd` top-level integration + sim/synth execution pending tools and the recorded RTL fixes.
