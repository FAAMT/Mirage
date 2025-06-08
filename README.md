# Mirage FPGA AI Accelerator 🚀⚡✨

Mirage is an open-source, low-footprint hardware accelerator aimed at bringing real-time Transformer inference to ultra-small FPGAs. It targets the Digilent Cora Z7-07S (Zynq-7007S) and is the main design for the upcoming Multicore Desktop Max demo at MDXB MIT 2025. �⚡✨

## Project Goals 🚀⚡✨

- **Edge-class Latency** – < 10 ms per token for a 256-dim encoder running at 100 MHz fabric.
- **Minimal Footprint** – Fit comfortably inside 60 DSPs <140 BRAMs, leaving 30 % slack for glue logic.
- **Quantization-Friendly Pipeline** – End-to-end INT8 / INT11 fixed-point; no floating point units.
- **Modular VHDL Cores** – Matrix-Multiply, Multi-Head Attention, Softmax, and Feed-Forward cores can be reused in other designs.
- **Commercial Viability** – Fabric-agnostic RTL with a path to ASIC tape-out via a fab-less flow by 2026.

## High-Level Architecture 🚀⚡✨

Mirage consists of tightly-coupled VHDL cores forming a minimal Transformer block: multi-head attention, fixed-point softmax, and a feed-forward MLP, all managed by a Cortex-A9 CPU for orchestration and token tiling.

<p align="center">
  <img src="/Users/faamt/Desktop/Nomadic Inc/Mirage/images/mirage-top-level-bd.png" alt="Mirage Top-Level Block Diagram" width="600"/>
</p>

| Core     | Function                                   | DSP  | BRAM | Thruput*             |
|----------|-------------------------------------------|------|------|----------------------|
| MMC      | 64 × 64 INT8 matrix multiply (time-multiplexed) | 60   | 16   | 1 tile / 5 cy        |
| MHA      | Q/K/V projection + dot-product            | –    | 12   | 4 toks / pass        |
| SMX      | 3rd-order Taylor softmax (INT11 → 2⁸)     | –    | 4    | 4 vecs / 2 cy        |
| FFN      | 256 → 128 → 256 MLP                       | 60†  | 20   | 1 tok / 12 cy        |
| Add-Norm | Residual add + layer-norm (INT8)          | –    | 2    | 1 tok / cy           |

\* Measured at 100 MHz fabric clock.  
† Re-uses MMC DSP stack.

An ARM Cortex-A9 on the PS side orchestrates DMA, tiling and token streaming. 🚀⚡✨

**Block diagram:** `docs/architecture.svg` (coming soon). 🚀⚡✨

## Development Timeline 🚀⚡✨

### 2024-2025 Highlights 🚀⚡✨

- **2024-03-28** – Ideation sprint; surveyed candidate projects.
- **2024-03-29** – Chose AlterEgo-inspired silent-speech use-case.
- **2025-03-28** – Mirage project officially kicked off.
- **2025-04-25** – ✅ Matrix Multiply Core passes unit tests.
- **2025-04-30** – ✅ Single-token Multi-Head Attention simulated.
- **2025-05-02** – Softmax (SMX) prototype completed.
- **2025-05-11** – Integrated Scores × V multiplication; Add-Norm spec written.
- **2025-05-23** – System-level skeleton committed; board bring-up scripts added.
- **2025-05-29** – SMX outputs finalized (INT8 span); FFN design phase begins.

A detailed day-by-day log lives in `docs/timeline.md`. 🚀⚡✨

## Repository Layout 🚀⚡✨
/boards/ Board files & XDC constraints  
/docs/ Specs, white-papers, dev diary  
/scripts/ Python verification helpers (no external libs)  
/sim/ GHDL & GtkWave testbenches  
/vhdl/ RTL cores (mmc/, mha/, smx/, ffn/, glue/)  
Makefile Convenience targets for sim & synth

## Quick Start 🚀⚡✨

### Prerequisites 🚀⚡✨

- Vivado 2023.1 (or compatible) WebPACK is enough  
- GHDL ≥ 4.0 for VHDL-2008 simulation  
- Python 3.11 (standard library only – numpy-free zone)

### Build & Simulate 🚀⚡✨

```bash
# clone
$ git clone https://github.com/<your-handle>/mirage.git
$ cd mirage

# unit-test Matrix-Multiply Core
$ make sim CORE=mmc           # generates wave.vcd
$ open -a gtkwave sim/wave.vcd

# full system bitstream (Cora Z7-07S)
$ make synth BOARD=cora-z7
