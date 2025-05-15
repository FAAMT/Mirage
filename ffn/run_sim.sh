#!/bin/bash

# Configuration
TOP_ENTITY="ffn_tb"
SRC="../types_pkg.vhd ../types_pkg_body.vhd ../matrix_mult/matrix_mult.vhd ../ffn/ffn.vhd"
TB="../ffn/ffn_tb.vhd"
WAVE="ffn.ghw"
GTKW="ffn.gtkw"

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
ghdl -r --std=08 $TOP_ENTITY --wave="$WAVE" --stop-time=200us || { echo "[ERROR] Simulation failed"; exit 1; }

# Launch GTKWave
pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &
