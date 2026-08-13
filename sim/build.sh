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
  ww_probe_tb.vhd
  # stage_engine_asis_tb.vhd is deliberately absent: it predates the
  # stride/row_ofs ports Layer C added to stage_engine, so it no longer
  # compiles, and the addressing contract it asserts (addr = base +
  # row*lanes + lane) has been superseded by the strided one. Reviving it
  # means deciding what contract it should assert now -- not a mechanical fix.
)

A="ghdl -a --std=08 --workdir=work"
for f in "${RTL[@]}"; do $A "$f"; done
for f in "${TB[@]}"; do [ -f "$f" ] && $A "$f"; done

# --all: run every testbench and exit non-zero if any fails, so this can be a
# gate rather than something driven by hand. Every TB calls std.env.finish, so
# a pass terminates; the timeout is a backstop against a future one that does
# not, since a hang and a slow run look identical to a caller.
if [ "${1:-}" = "--all" ]; then
  fail=0
  for f in "${TB[@]}"; do
    [ -f "$f" ] || continue
    unit=$(basename "$f" .vhd)
    printf '  %-22s ' "$unit"

    # encoder_layer_tb is a command interpreter: it reads a script and writes a
    # result. Both tapes are gitignored, so skip rather than fail when absent.
    args=(); check_against=""
    if [ "$unit" = "encoder_layer_tb" ]; then
      if [ ! -f ops_cmd.txt ] || [ ! -f ops_res.txt ]; then
        echo "SKIP  (needs sim/ops_cmd.txt + ops_res.txt, both gitignored)"
        continue
      fi
      out_file=$(mktemp)
      args=(-gSCRIPT=ops_cmd.txt "-gRESULT=$out_file")
      check_against=ops_res.txt
    fi

    if log=$(ghdl -e --std=08 --workdir=work "$unit" 2>&1 &&
             timeout "${TB_TIMEOUT:-900}" ghdl -r --std=08 --workdir=work \
                     "$unit" "${args[@]}" 2>&1); then
      if [ -n "$check_against" ] && ! diff -q "$check_against" "$out_file" >/dev/null 2>&1; then
        echo "FAIL  (ran, but output differs from $check_against)"; fail=1
      else
        echo "PASS${check_against:+  (bit-exact vs $check_against)}"
      fi
    else
      echo "FAIL"; fail=1
      echo "$log" | grep -viE 'metavalue' | tail -6 | sed 's/^/        /'
    fi
    # plain `if`, not `[ ... ] && ...`: under `set -e` the latter's behaviour
    # when the test fails depends on a subtle exemption. Do not make a reader
    # reason about that to know whether the loop continues.
    if [ -n "${out_file:-}" ]; then
      rm -f "$out_file"
      out_file=""
    fi
  done
  echo
  [ $fail -eq 0 ] && echo "SUITE PASS" || echo "SUITE FAIL"
  exit $fail
fi

if [ $# -ge 1 ]; then
  unit="$1"; shift
  ghdl -e --std=08 --workdir=work "$unit"
  # GHDL wants simulation options AFTER the top-level unit name.
  exec ghdl -r --std=08 --workdir=work "$unit" "$@"
fi
echo "analyse OK"
