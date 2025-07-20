# Configuration
TOP_ENTITY="quantizer_tb"
SRC="../pkgs/types_pkg.vhd ../pkgs/types_pkg_body.vhd ../quantizer/quantizer.vhd"
TB="quantizer_tb.vhd"
WAVE="quantizer.ghw"
GTKW="quantizer.gtkw"

# Clean previous build artifacts
echo "[INFO] Cleaning..."
rm -f *.o *.cf work-obj* work-*.cf "$WAVE"

# Compile source and testbench
echo "[INFO] Compiling source..."
for file in $SRC; do
    ghdl -a --std=08 "$file" || { echo "[ERROR] Failed to compile $file"; exit 1; }
done

echo "[INFO] Compiling testbench..."
ghdl -a --std=08 "$TB" || { echo "[ERROR] Failed to compile $TB"; exit 1; }

# Elaborate top-level
echo "[INFO] Elaborating $TOP_ENTITY..."
ghdl -e --std=08 "$TOP_ENTITY" || { echo "[ERROR] Elaboration failed"; exit 1; }

# Run simulation
echo "[INFO] Running simulation..."
ghdl -r --std=08 "$TOP_ENTITY" --wave="$WAVE"  --stop-time=50us|| { echo "[ERROR] Simulation failed"; exit 1; }

pkill gtkwave 2>/dev/null
gtkwave "$WAVE" "$GTKW" &