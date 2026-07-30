# W12 — Timing Closure & Bitstream v1 (PLAN + EXPECTED-REPORT SKELETON)

- **Task:** `p3/w12` — Timing closure and bitstream v1
- **Status:** NO BITSTREAM BUILT (contract: no synth/P&R/board without human; Vivado/board unavailable). Driver implementation is real and complete; the bitstream/loopback parts are plan + skeleton.
- **Date:** 2026-07-24
- **Artifacts:** `driver/mirage_driver.c` (real), `outputs/mirage_driver.h`, `outputs/synth.tcl` (W11), this report. `outputs/mirage_v1.bit` — NOT PRODUCED (needs Vivado).

## 1. Timing-closure iteration plan (subtask: iteration 2)

Known risks going in (from W11 skeletons): DSP ~91–98% and the quantizer→gemm combinational path.

Iteration ladder (each step = one `outputs/synth.tcl` run by a human):
1. **Baseline** (60-DSP gemm, 200 MHz): record WNS/TNS/Fmax into `reports/w11-postsynth-timing.md` §3.
2. If WNS < 0 on quantizer→gemm: insert the drafted pipeline register (W6 note) — cost +1 cyc/pass (~2%).
3. If DSP > 90% or routing congestion: switch gemm to 2×30 time-multiplex (W11 resource map §4) — cost ~2× score/proj latency, still ~12× real-time headroom (perf-report-v1).
4. If still failing: clock divider fallback (task risk note) — 100 MHz bitstream; throughput ≈ 180 tok/s overlapped, still real-time.
5. Freeze the first config with WNS ≥ 0 as **bitstream v1 (non-optimized)**.

## 2. EXPECTED bitstream-v1 record — TO BE FILLED BY VIVADO RUN

```
bitstream:              outputs/mirage_v1.bit  (NOT PRODUCED this run)
config:                 [60-DSP | 2x30] @ [200|100] MHz
WNS/TNS:                ______ / ______
utilization LUT/DSP/BRAM: ______ / ______ / ______
sha256(.bit):           ______
```

## 3. Driver init & in-hardware loopback (acceptance clause)

- Driver init: `mirage_init()` (`driver/mirage_driver.c`) — reads STATUS, expects error bit clear, enables IRQ. Board-free dry run not possible (no PL); code reviewed against the register map in `outputs/mirage_driver.h`.
- Loopback minimal case (when a board exists): write a known token buffer to `TOK_ADDR`, run 1 layer, read `OUT_ADDR`, compare against the W8 model golden for the same input (pass = bit-identical int16 outputs). ISR path: `mirage_isr()` + `mirage_wait_done()`; poll fallback implemented.
- **Board availability:** none in this environment — clause "if board available" is recorded as NOT AVAILABLE.

## 4. What is real vs pending

| Item | Status |
|---|---|
| `driver/mirage_driver.c` + register map + ISR + poll fallback | **written** |
| Timing-closure ladder + fallbacks | planned, scripts ready (`outputs/synth.tcl`) |
| `outputs/mirage_v1.bit` | **pending human Vivado run** — explicitly not fabricated |
| Driver init on hardware / loopback | **pending board** |

## Disposition

Producible deliverables done (driver, closure plan, skeletons). The bitstream and on-board loopback require Vivado + Cora board with a human — flagged as pending, not fabricated.
