#!/usr/bin/env bash
# =====================================================================
# vivado_lint.sh — catch Vivado-class VHDL errors locally, without Vivado.
#
#   ./tools/vivado_lint.sh
#
# Two passes over the same source list the synthesis script uses:
#
#   --std=08  correctness. Must be clean. This is the language the design
#             is actually written in.
#
#   --std=93  Vivado proxy. Vivado's synthesiser implements only a subset
#             of VHDL-2008, and the gap that bit us is readable 'out'
#             ports -- legal in 2008, rejected by Vivado with
#             [Synth 8-10557]. VHDL-93 forbids them too, so --std=93
#             surfaces them locally in about a second.
#
#             Validated 2026-07-29 against Vivado 2022.2's own output:
#             --std=93 flagged the same 9 lines in ffn.vhd that Vivado
#             did, and nothing else. See runs/2026-07-29-vivado-2022.2/.
#
# Only the readable-out-port class is reported from the 93 pass. Other
# 93 complaints (non-ASCII in comments, universal-integer bounds) are
# 93-vs-2008 differences Vivado demonstrably accepts, so they are noise.
# =====================================================================
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

SRCS=(
    rtl/pkgs/types_pkg.vhd
    rtl/pkgs/types_pkg_body.vhd
    rtl/outputs/rc2_hardening.vhd
    rtl/mirage/gemm_array.vhd
    rtl/mirage/quantizer.vhd
    rtl/mirage/qkv_proj_bram.vhd
    rtl/mirage/qk_score_bram.vhd
    rtl/mirage/qkv_proj_calc.vhd
    rtl/mirage/qkv_score_calc.vhd
    rtl/mirage/multi_head.vhd
    rtl/mirage/taylor_exp.vhd
    rtl/mirage/ffn.vhd
    rtl/outputs/smx_cache.vhd
    rtl/outputs/smx_v_calc.vhd
    rtl/outputs/smx_core.vhd
    rtl/outputs/weight_stream_ctrl.vhd
    rtl/integration/mirage_sizing_container.vhd
    rtl/integration/mirage_sizing_subset.vhd
)

TOPS=(
    gemm_array quantizer qkv_proj_bram qk_score_bram qkv_proj_calc
    qkv_score_calc multi_head taylor_exp ffn smx_cache smx_v_calc
    smx_core weight_stream_ctrl mirage_sizing_container mirage_sizing_subset
)

command -v ghdl >/dev/null || { echo "ghdl not found"; exit 127; }
echo "ghdl: $(ghdl --version | head -1)"
fail=0

echo
echo "=== pass 1: --std=08 (correctness) ==="
cd "$WORK"
for f in "${SRCS[@]}"; do
    if ghdl -a --std=08 "$ROOT/$f" 2>/tmp/lint08.$$; then
        printf '  %-34s ok\n' "$(basename "$f")"
    else
        printf '  %-34s FAIL\n' "$(basename "$f")"
        sed 's/^/        /' /tmp/lint08.$$
        fail=1
    fi
done
rm -f /tmp/lint08.$$

if [ $fail -eq 0 ]; then
    echo
    echo "=== elaboration ==="
    for t in "${TOPS[@]}"; do
        if ghdl -e --std=08 "$t" >/dev/null 2>&1; then
            printf '  %-34s ok\n' "$t"
        else
            printf '  %-34s FAIL\n' "$t"; fail=1
        fi
    done
fi

echo
echo "=== pass 2: --std=93 (Vivado readable-out-port proxy) ==="
found=0
for f in "${SRCS[@]}"; do
    hits=$(ghdl -a --std=93 -C "$ROOT/$f" 2>&1 | grep -E 'cannot be read' || true)
    if [ -n "$hits" ]; then
        printf '  %-34s %s site(s)\n' "$(basename "$f")" "$(echo "$hits" | wc -l)"
        echo "$hits" | sed "s|$ROOT/||" | sed 's/^/        /'
        found=$((found+1)); fail=1
    fi
done
[ $found -eq 0 ] && echo "  no readable 'out' ports — Vivado will not raise [Synth 8-10557]"

echo
if [ $fail -eq 0 ]; then
    echo "LINT PASS — safe to hand to Vivado"
else
    echo "LINT FAIL — fix before spending a Vivado run"
fi
exit $fail
