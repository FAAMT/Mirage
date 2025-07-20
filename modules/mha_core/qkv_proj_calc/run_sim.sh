#!/bin/bash

# Configuration
TOP_ENTITY="qkv_proj_calc_tb"
SRC="../../pkgs/types_pkg.vhd 
    ../../pkgs/types_pkg_body.vhd 
    ../../pkgs/tb_io_pkg.vhd 
    ../../pkgs/tb_io_pkg_body.vhd  
    ../../gemm_array/gemm_array.vhd  
    ../qkv_proj_calc/qkv_proj_calc.vhd "
TB="../qkv_proj_calc/qkv_proj_calc_tb.vhd"
WAVE="qkv_proj_calc.ghw"
GTKW="qkv_proj_calc.gtkw"

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
