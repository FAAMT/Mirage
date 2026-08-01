#!/usr/bin/env bash
# =====================================================================
# check_ops.sh -- diff the C driver's register sequence against the
# reference script that GHDL already validates
#
# Compares, between sw/test/ops_trace.txt and sim/ops_cmd.txt:
#
#   1. every CFG / JOB / MARK / DUMP / DUMPS line, in full and in order
#   2. the ADDRESS of every ACT line
#   3. the INDEX of every WROW line
#
# Not compared: the payload bytes of ACT and WROW. trace_ops.c writes
# zeros there because the values come from a seeded Python RNG and are
# test data rather than driver behaviour -- see its header.
#
# A pass means the driver drives the same registers, with the same
# values, in the same order as the sequence sim/ops_cmd.txt encodes,
# which model_ref/test_ops.py has already run against the RTL.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

REF=../../sim/ops_cmd.txt
OUR=ops_trace.txt

[ -f "$REF" ] || { echo "missing $REF"; exit 1; }
[ -f "$OUR" ] || { echo "missing $OUR -- run make trace first"; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fail=0

# --- 1. control lines -------------------------------------------------
grep -vE '^(ACT|WROW) ' "$REF" > "$work/ref.ctl"
grep -vE '^(ACT|WROW) ' "$OUR" > "$work/our.ctl"

if diff -u "$work/ref.ctl" "$work/our.ctl" > "$work/ctl.diff"; then
  echo "PASS  control sequence  ($(wc -l < "$work/ref.ctl") lines identical)"
else
  echo "FAIL  control sequence"
  head -40 "$work/ctl.diff"
  fail=1
fi

# --- 2. ACT addresses -------------------------------------------------
awk '$1=="ACT" {print $2}' "$REF" > "$work/ref.act"
awk '$1=="ACT" {print $2}' "$OUR" > "$work/our.act"

if diff -q "$work/ref.act" "$work/our.act" > /dev/null; then
  echo "PASS  activation addresses  ($(wc -l < "$work/ref.act") writes)"
else
  echo "FAIL  activation addresses"
  diff -u "$work/ref.act" "$work/our.act" | head -20
  fail=1
fi

# --- 3. WROW indices --------------------------------------------------
awk '$1=="WROW" {print $2}' "$REF" > "$work/ref.wrow"
awk '$1=="WROW" {print $2}' "$OUR" > "$work/our.wrow"

if diff -q "$work/ref.wrow" "$work/our.wrow" > /dev/null; then
  echo "PASS  weight row indices  ($(wc -l < "$work/ref.wrow") rows)"
else
  echo "FAIL  weight row indices"
  diff -u "$work/ref.wrow" "$work/our.wrow" | head -20
  fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "DRIVER CHECK PASS -- the C driver emits the verified register sequence."
else
  echo "DRIVER CHECK FAIL"
fi
exit "$fail"
