#!/bin/bash

# Configuration
TOP_ENTITY="qkv_score_calc_tb"
SRC="../../pkgs/types_pkg.vhd 
     ../../pkgs/types_pkg_body.vhd  
     ../../gemm_array/gemm_array.vhd 
     ../qkv_proj_bram/qkv_proj_bram.vhd 
     ../qkv_proj_calc/qkv_proj_calc.vhd  
     ../../quantizer/quantizer.vhd 
     ../qkv_score_calc/qkv_score_calc.vhd"
TB="../qkv_score_calc/qkv_score_calc_tb.vhd"
WAVE="qkv_score_calc.ghw"
GTKW="qkv_score_calc.gtkw"

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
ghdl -r --std=08 $TOP_ENTITY --wave="$WAVE" --stop-time=16500us || { echo "[ERROR] Simulation failed"; exit 1; }

# Launch GTKWave
pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &
