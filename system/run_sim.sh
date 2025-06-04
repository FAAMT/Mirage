#!/bin/bash

# Configuration for System Testbench
TOP_ENTITY="system_tb"
# VHDL source files: packages, file IO, cores
SRC="../types_pkg.vhd 
     ../types_pkg_body.vhd 
     ../tb_io_pkg.vhd 
     ../tb_io_pkg_body.vhd 
     ../matrix_mult/matrix_mult.vhd 
     ../multi_head/multi_head.vhd
     ../softmax/softmax.vhd
     ../ffn/ffn.vhd"

TB="system_tb.vhd"
WAVE="system.ghw"
GTKW="system.gtkw"

# Clean previous build artifacts
echo "[INFO] Cleaning..."
rm -f *.o *.cf work-obj* work-*.cf "$WAVE"

# Compile all sources
echo "[INFO] Compiling source packages and cores..."
ghdl -a --std=08 $SRC || { echo "[ERROR] Failed to compile sources"; exit 1; }

echo "[INFO] Compiling testbench..."
ghdl -a --std=08 $TB || { echo "[ERROR] Failed to compile testbench"; exit 1; }

# Elaborate top-level
echo "[INFO] Elaborating $TOP_ENTITY..."
ghdl -e --std=08 $TOP_ENTITY || { echo "[ERROR] Elaboration failed"; exit 1; }

# Run simulation
echo "[INFO] Running simulation..."
ghdl -r --std=08 $TOP_ENTITY --wave="$WAVE" --stop-time=3300us || { echo "[ERROR] Simulation failed"; exit 1; }

# Launch GTKWave
pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &