# W11 — Post-Synth Timing Report (EXPECTED-REPORT SKELETON)

- **Task:** `p3/w11` — Synthesis and initial P&R
- **Status:** NO SYNTHESIS RUN (contract: no synth/P&R without human-in-loop; Vivado not available in this environment). This is the plan + script + expected-report skeleton. Pre-synth estimates marked ESTIMATE; actuals pending a human Vivado run.
- **Date:** 2026-07-24

## 1. Run plan (human + Vivado ≥ 2023.1, Cora Z7-07S = xc7z007sclg400-1)

```bash
vivado -mode batch -source outputs/synth.tcl -journal logs/w11-vivado.jou
# then paste report_timing_summary + report_utilization into sections 3-4 below
```

`outputs/synth.tcl` (provided) does: read VHDL-2008 sources (types_pkg, gemm_array, quantizer, qkv_proj/smx/ffn set + outputs/ drafts), `synth_design -top encoder_layer -part xc7z007sclg400-1`, `opt/place/route`, then `report_timing_summary`, `report_utilization`, `report_utilization -hierarchical`, and (for the HW functional sim clause) `write_vhdl -mode funcsim netlist.vhd` + launch the module TBs against the netlist.

## 2. Expected timing-critical paths (pre-synth ESTIMATE)

| Rank | Path | Why | Mitigation ready |
|---|---|---|---|
| 1 | quantizer (comb, 60 lanes) → gemm_array input mux → DSP | longest comb chain (W6 risk) | pipeline register (drafted W6) |
| 2 | gemm accum adder chain 32b × 12 | DSP cascade routing | none needed (DSP48 cascade) |
| 3 | taylor_exp z³ multiply (42×21b) | wide mult, needs 2 DSPs + fabric | register between term stages (W3 fix set) |
| 4 | smx_cache readout mux (32:1 × 48b) | wide comb mux | optional output register (W4 note) |

Target: 200 MHz (5.0 ns). Fallback: 100 MHz still meets real-time NMT budget (perf-report-v1 §3).

## 3. EXPECTED `report_timing_summary` — TO BE FILLED BY VIVADO RUN

```
WNS (worst negative slack):   ______ ns   (acceptance: >= 0)
TNS (total negative slack):   ______ ns
Fmax (derived):               ______ MHz
Failing endpoints (top 10):   ______
```

## 4. Utilization vs acceptance (< 90% of critical resources)

See `reports/w11-resource-map.md` (companion). Pre-synth ESTIMATE flags DSP at ~91% — EXPECT this to fail the <90% bar at synthesis unless the gemm tile is time-multiplexed (2×30) or int8 mults mapped partly to LUTs. This is the key watch item.

## 5. HW functional sim with exported netlist — plan

1. `write_vhdl -mode funcsim` post-route netlist + `write_sdf`.
2. Re-run module TBs (qkv_score_calc_tb, qk_score_bram_tb, ffn_tb, system_tb) against the netlist in ghdl: same golden vectors as W1/W2/W7 model runs.
3. **Prerequisite upstream RTL fixes before this can pass:** W1 score writeback, W3/W5 exp scaling + range reduction, W7 mac_array/done/writeback — all recorded in the phase reports.

## Disposition

Skeleton + run script + estimates produced (deliverable). Actual post-synth numbers require a human Vivado run — flagged, not fabricated.
