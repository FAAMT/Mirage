#!/bin/bash

# Configuration
TOP_ENTITY="matrix_mult_tb"
SRC="matrix_mult.vhd"
TB="matrix_mult_tb.vhd"
WAVE="matrix_mult.ghw"
GTKW="matrix_mult.gtkw"

# Clean previous build artifacts
echo "[INFO] Cleaning..."
rm -f *.o *.cf work-obj* work-*.cf "$WAVE"

# Compile source and testbench
echo "[INFO] Compiling source..."

ghdl -a --std=08 ../types_pkg.vhd  ../types_pkg_body.vhd|| { echo "[ERROR] Failed to compile types_pkg.vhd"; exit 1; }

ghdl -a --std=08 "$SRC" || { echo "[ERROR] Failed to compile $SRC"; exit 1; }

echo "[INFO] Compiling testbench..."
ghdl -a --std=08 "$TB" || { echo "[ERROR] Failed to compile $TB"; exit 1; }

# Elaborate top-level
echo "[INFO] Elaborating $TOP_ENTITY..."
ghdl -e --std=08 "$TOP_ENTITY" || { echo "[ERROR] Elaboration failed"; exit 1; }

# Run simulation
echo "[INFO] Running simulation..."
ghdl -r --std=08 "$TOP_ENTITY" --wave="$WAVE" || { echo "[ERROR] Simulation failed"; exit 1; }

pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &

