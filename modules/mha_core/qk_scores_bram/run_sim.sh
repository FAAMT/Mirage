#!/bin/bash

# Configuration
TOP_ENTITY="qkv_proj_bram_tb"
SRC_FILES=(
    "../../pkgs/types_pkg.vhd"
    "../../pkgs/types_pkg_body.vhd"
    "../../pkgs/tb_io_pkg.vhd"
    "../../pkgs/tb_io_pkg_body.vhd"
    "../../gemm_array/gemm_array.vhd"
    "../qkv_proj_calc/qkv_proj_calc.vhd"
    "../qkv_proj_bram/qkv_proj_bram.vhd"  # Must be compiled before testbench
)
TB="../qkv_proj_bram/qkv_proj_bram_tb.vhd"
WAVE="qkv_proj_bram.ghw"
GTKW="qkv_proj_bram.gtkw"

# Clean previous build artifacts
echo "[INFO] Cleaning..."
rm -f *.o *.cf work-obj* work-*.cf "$WAVE"

# Verify all files exist first
for file in "${SRC_FILES[@]}" "$TB"; do
    if [ ! -f "$file" ]; then
        echo "[ERROR] File not found: $file"
        exit 1
    fi
done

# Compile in proper dependency order
echo "[INFO] Compiling sources..."
for file in "${SRC_FILES[@]}"; do
    echo "  Compiling $file..."
    ghdl -a --std=08 "$file" || { echo "[ERROR] Failed to compile $file"; exit 1; }
done

# Compile testbench
echo "[INFO] Compiling testbench..."
ghdl -a --std=08 "$TB" || { 
    echo "[ERROR] Testbench compilation failed. Common issues:"
    echo "  1. Ensure qkv_proj_bram.vhd has:"
    echo "     entity qkv_proj_bram is ... end entity;"
    echo "  2. Check bram_data_in port exists in entity declaration"
    echo "  3. Verify QKV_BRAM_DATA_W is properly defined"
    exit 1
}

# Elaborate and run
echo "[INFO] Elaborating $TOP_ENTITY..."
ghdl -e --std=08 "$TOP_ENTITY" || { echo "[ERROR] Elaboration failed"; exit 1; }

echo "[INFO] Running simulation..."
ghdl -r --std=08 "$TOP_ENTITY" --wave="$WAVE" --stop-time=24ms || { echo "[ERROR] Simulation failed"; exit 1; }

# Launch GTKWave
pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &