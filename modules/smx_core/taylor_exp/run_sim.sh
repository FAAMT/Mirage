#!/bin/bash

# Configuration
TOP_ENTITY="taylor_exp_tb"
SRC_FILES=("../../pkgs/types_pkg.vhd"
     "../../pkgs/types_pkg_body.vhd" 
     "../taylor_exp/taylor_exp.vhd"
     )
TB="../taylor_exp/taylor_exp_tb.vhd"
WAVE="taylor_exp.ghw"
GTKW="taylor_exp.gtkw"

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
    echo "[ERROR] Testbench compilation failed."
    exit 1
}

# Elaborate and run
echo "[INFO] Elaborating $TOP_ENTITY..."
ghdl -e --std=08 "$TOP_ENTITY" || { echo "[ERROR] Elaboration failed"; exit 1; }

echo "[INFO] Running simulation..."
ghdl -r --std=08 "$TOP_ENTITY" --wave="$WAVE" --stop-time=100ns || { echo "[ERROR] Simulation failed"; exit 1; }

# Launch GTKWave
pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &