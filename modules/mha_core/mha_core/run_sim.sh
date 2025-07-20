#!/bin/bash

# Configuration
TOP_ENTITY="multi_head_tb"
SRC="../types_pkg.vhd ../types_pkg_body.vhd ../tb_io_pkg.vhd ../tb_io_pkg_body.vhd  ../matrix_mult/matrix_mult.vhd ../multi_head/multi_head.vhd"
TB="../multi_head/multi_head_tb.vhd"
WAVE="multi_head.ghw"
GTKW="multi_head.gtkw"

# Clean previous build artifacts
echo "[INFO] Cleaning..."
rm -f *.o *.cf work-obj* work-*.cf "$WAVE"

# Compile all sources
echo "[INFO] Compiling source..."
ghdl -a --std=08 $SRC || { echo "[ERROR] Failed to compile sources"; exit 1; }

echo "[INFO] Compiling testbench..."
ghdl -a --std=08 $TB || { echo "[ERROR] Failed to compile testbench"; exit 1; }

# Elaborate top-level
echo "[INFO] Elaborating $TOP_ENTITY..."
ghdl -e --std=08 $TOP_ENTITY || { echo "[ERROR] Elaboration failed"; exit 1; }

# Run simulation
echo "[INFO] Running simulation..."
ghdl -r --std=08 $TOP_ENTITY --wave="$WAVE" --stop-time=6ms || { echo "[ERROR] Simulation failed"; exit 1; }

# Launch GTKWave
pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &
