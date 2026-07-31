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
)

TB=(
  ../rtl_v2/pkg/addr_math_tb.vhd
  ../rtl_v2/pkg/sat8_tb.vhd
  ../rtl_v2/pkg/exp_idx_tb.vhd
  ../rtl_v2/core/mac_array_tb.vhd
  stage_engine_tb.vhd
  softmax_unit_tb.vhd
  encoder_layer_tb.vhd
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
