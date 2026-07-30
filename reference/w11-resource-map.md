# W11 — Resource Map (EXPECTED-REPORT SKELETON)

- **Task:** `p3/w11` — companion to `reports/w11-postsynth-timing.md`
- **Status:** NO SYNTHESIS RUN (contract). Pre-synth estimates from the W6 inspection-based profile; Vivado actuals to be pasted into §3.
- **Date:** 2026-07-24

## 1. Device budget (Cora Z7-07S, xc7z007sclg400-1 — verify against current datasheet)

| Resource | Total | Notes |
|---|---|---|
| LUT | ~14,400 | logic |
| FF | ~28,800 | |
| DSP48E1 | 66 | int8 MACs |
| RAMB36 | 50 (≈ 225 KB) | weights + activations |

## 2. Pre-synth ESTIMATE (W6 profile + W7/W9 updates)

| Block | LUT (est) | DSP | RAMB36 |
|---|---|---|---|
| gemm_array (12×5 int8) | 400 | **60** | 0 |
| taylor_exp (W3/W5 fix set: z², z³, ×174763, ×185) | 300 | 4 | 0 |
| quantizer ×2 (60-lane, comb) | 500 | 0 | 0 |
| smx_cache + smx_v_calc (drafts) | 600 | 1 | 1 |
| W_Q/K/V + Wo buffers (128 KB, W9) | 0 | 0 | 32 |
| qkv_proj_bram + score_bram | 100 | 0 | 4 |
| FFN proj weights (16 KB) + activation buffers | 200 | 0 | 5 |
| FFN hl/ol weights (262 KB) | — | 0 | **DOES NOT FIT — DDR-streamed (W9 §3)** |
| FSMs / glue / weight_stream_ctrl | 1,500 | 0 | 0 |
| **Total resident** | **~3,600 (25%)** | **65 (98%)** | **42 (84%)** |

## 3. EXPECTED `report_utilization` — TO BE FILLED BY VIVADO RUN

```
LUT:      ______ / 14400  (___%)
FF:       ______ / 28800  (___%)
DSP48:    ______ / 66     (___%)   <-- acceptance < 90%: expected FAIL at 60/66 (91-98%)
RAMB36:   ______ / 50     (___%)
```

## 4. < 90% acceptance analysis (pre-synth)

- **DSP: expected to EXCEED 90%** (60–65 of 66). Mitigations, in order of preference:
  1. Time-multiplex gemm tile 12×5 → 2 passes of 6×5 (30 DSP, ~2× score/proj latency — still within real-time budget, perf-report-v1 §3 headroom 24×).
  2. Map the taylor z³ mult to LUTs (frees 2).
  3. Keep 60-DSP config only if timing closes at 200 MHz with slack (unlikely).
- **RAMB36: ~84% resident** IF FFN hl/ol weights are DDR-streamed (mandatory per W9); otherwise impossible (>100%).
- LUT/FF: comfortable (<30%).

## 5. Floorplanning watch items (task risk note)

- DSP cascade locality for gemm_array (keep the 12×5 tile in one DSP column region).
- BRAM-to-DSP routing for weight fetch: place W_Q/K/V BRAMs adjacent to the DSP column.
- If routing congestion > threshold: floorplan gemm_array + quantizer + taylor_exp into one clock region; FFN weight stream near the MIG/DDR interface.

## Disposition

Resource-map skeleton with pre-synth estimates produced. DSP ~91–98% is the flagged acceptance risk with a ready mitigation (2×30 time-multiplex). Actuals pending human Vivado run.
