#!/usr/bin/env bash
# Analyse rtl_v2 and the testbenches, then optionally run one testbench.
#
#   ./build.sh                 analyse everything
#   ./build.sh <tb_name>       analyse, elaborate and run that testbench
#
# GHDL tracks staleness per design unit and refuses to elaborate against a
# package that has been reanalysed since. Rebuilding the whole library in
# dependency order every time is cheaper than chasing that error.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p work

RTL=(
  ../rtl_v2/pkg/mirage_v2_pkg.vhd
  ../rtl_v2/prim/bram_sdp.vhd
  ../rtl_v2/prim/mac_lane.vhd
  ../rtl_v2/core/mac_array.vhd
  ../rtl_v2/core/weight_window.vhd
  ../rtl_v2/core/exp_rom.vhd
  ../rtl_v2/core/recip_rom.vhd
  ../rtl_v2/core/softmax_unit.vhd
  ../rtl_v2/core/stage_engine.vhd
  ../rtl_v2/core/gemm_sequencer.vhd
  ../rtl_v2/ctrl/layer_ctrl.vhd
  ../rtl_v2/top/encoder_layer.vhd
  ../rtl_v2/axi/mirage_axi_lite.vhd
  ../rtl_v2/axi/wgt_stream.vhd
  ../rtl_v2/top/mirage_axi_top.vhd
  # Nothing simulates the shim -- it is here so a port-map mistake in it
  # is caught by GHDL on this machine rather than by Vivado five hours
  # later on the machine that has Vivado. It is VHDL-93 but legal 2008,
  # so analysing it under --std=08 with everything else is fine.
  ../rtl_v2/top/mirage_axi_wrap.vhd
)

TB=(
  ../rtl_v2/pkg/addr_math_tb.vhd
  ../rtl_v2/pkg/sat8_tb.vhd
  ../rtl_v2/pkg/exp_idx_tb.vhd
  ../rtl_v2/core/mac_array_tb.vhd
  stage_engine_tb.vhd
  softmax_unit_tb.vhd
  encoder_layer_tb.vhd
  axi_lite_tb.vhd
)

A="ghdl -a --std=08 --workdir=work"
for f in "${RTL[@]}"; do $A "$f"; done
for f in "${TB[@]}"; do [ -f "$f" ] && $A "$f"; done

if [ $# -ge 1 ]; then
  unit="$1"; shift
  ghdl -e --std=08 --workdir=work "$unit"
  # GHDL wants simulation options AFTER the top-level unit name.
  exec ghdl -r --std=08 --workdir=work "$unit" "$@"
fi
echo "analyse OK"
