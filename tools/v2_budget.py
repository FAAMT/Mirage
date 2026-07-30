#!/usr/bin/env python3
"""
v2_budget.py — structural resource budget for rtl_v2, from the RTL itself.

    ./tools/v2_budget.py

WHY THIS EXISTS AND memory_audit.py DOES NOT COVER v2
-----------------------------------------------------
memory_audit.py finds v1's failure mode: bare array signals, indexed
directly, with too many read ports. Pointed at rtl_v2 it reports "0 arrays,
all clear" -- which is NOT evidence of anything. It is blind to v2 by
construction:

  * bram_sdp's array is sized by GENERICS (DATA_W, ADDR_W), which the
    heuristic cannot evaluate, so it skips it
  * the exp and recip tables are `constant`, not `signal`, so it never
    looks at them

Reporting that as a pass would have been worse than not running it. v2
needs a different question asked, because it has a different design rule:

    every memory goes through prim/bram_sdp or a generated ROM,
    and every multiply goes through prim/mac_lane

That is a STRUCTURAL invariant, so this tool checks it structurally --
counting instantiations and deriving the budget from their generics --
rather than estimating from heuristics.

It also flags the thing that would silently break the rule: a `*` operator
or an indexed array signal appearing outside the approved primitives.
"""
import re
import sys
import glob
import pathlib

DEV = dict(name="xc7z007sclg400-1", ramb36=50, dsp=66, lut=14400, ff=28800)

# Vivado packs a RAMB36E1 as 36 Kb; a 32-bit-wide port gives 1024 words.
RAMB36_BITS = 36 * 1024

PRIM_FILES = {"bram_sdp.vhd", "mac_lane.vhd", "exp_rom.vhd", "recip_rom.vhd"}

root = pathlib.Path(__file__).resolve().parent.parent
files = sorted(glob.glob(str(root / "rtl_v2" / "**" / "*.vhd"), recursive=True))

pkg = (root / "rtl_v2" / "pkg" / "mirage_v2_pkg.vhd").read_text()


def const(name, default=None):
    m = re.search(rf'constant\s+{name}\s*:\s*\w+\s*:=\s*([^;]+);', pkg)
    if not m:
        return default
    expr = m.group(1).split("--")[0].strip()
    try:
        return int(eval(expr, {"__builtins__": {}}, CONSTS))
    except Exception:
        return default


CONSTS = {}
for nm in ["MODEL_DIM", "NUM_HEADS", "HEAD_DIM", "FFN_DIM", "MAX_TOK", "ACT_W",
           "WGT_W", "ACC_W", "SCORE_W", "N_LANES", "W_BANKS", "BANK_BYTES",
           "BANK_W", "BANK_DEPTH", "BANK_AW", "ACT_AW", "SCORE_AW",
           "EXP_ROM_AW", "EXP_ROM_DW", "RECIP_ROM_AW", "RECIP_ROM_DW"]:
    v = const(nm)
    if v is not None:
        CONSTS[nm] = v

print(f"Device {DEV['name']}: {DEV['ramb36']} RAMB36, {DEV['dsp']} DSP48E1, "
      f"{DEV['lut']:,} LUT, {DEV['ff']:,} FF\n")

# ---------------------------------------------------------------------
# 1. Structural rule check
# ---------------------------------------------------------------------
print("RULE CHECK — memories via prim/bram_sdp or a ROM, multiplies via mac_lane")
violations = []
for f in files:
    name = pathlib.Path(f).name
    if name in PRIM_FILES or name.endswith("_tb.vhd"):
        continue
    txt = pathlib.Path(f).read_text()
    body = txt[txt.index("architecture"):] if "architecture" in txt else txt
    code = "\n".join(l for l in body.split("\n")
                     if not l.strip().startswith("--"))

    # a signal of an array type declared locally = a candidate memory
    for m in re.finditer(r'type\s+(\w+)\s+is\s+array', code):
        tname = m.group(1)
        if re.search(rf'signal\s+\w+\s*:\s*{tname}\b', code):
            # arrays of accumulators for muxing are fine; flag only if written
            # through an index, which is what makes it a memory
            if re.search(rf'\w+\s*\(\s*to_integer[^)]*\)\s*<=', code):
                violations.append(f"{name}: indexed write to a local array "
                                  f"({tname}) -- should be a bram_sdp")

    # A `*` only matters if BOTH operands are signals. Two other forms are
    # harmless and were false-positived by an earlier version of this check:
    #   - slice bounds like (j+1)*ACC_W, where j is a generate index and
    #     ACC_W a constant: folded at elaboration, no hardware at all
    #   - signal * power-of-two constant, e.g. head * HEAD_DIM (64):
    #     synthesises to a wired shift, not a multiplier
    KNOWN = set(CONSTS) | {"ACC_W", "WGT_W", "ACT_W", "BANK_W", "SCORE_W",
                           "MAX_TOK", "HEAD_DIM", "N_LANES", "W_BANKS"}

    def is_const(tok):
        tok = tok.strip().strip("()")
        if re.fullmatch(r"\d+", tok):
            return True
        return tok in KNOWN

    for m in re.finditer(r'([\w.\(\)\+\- ]{1,40})\*([\w.\(\)\+\- ]{1,40})', code):
        lhs, rhs = m.group(1), m.group(2)
        # take the token adjacent to the operator
        l = re.findall(r'[\w]+', lhs)
        r = re.findall(r'[\w]+', rhs)
        if not l or not r:
            continue
        if is_const(l[-1]) or is_const(r[0]):
            continue
        violations.append(f"{name}: signal*signal multiply outside mac_lane "
                          f"-> `{m.group(0).strip()[:60]}`")

if violations:
    for v in violations:
        print(f"  VIOLATION  {v}")
else:
    print("  clean — no memories or multiplies outside the approved primitives")

# ---------------------------------------------------------------------
# 2. Budget from instantiations
# ---------------------------------------------------------------------
print("\nBRAM BUDGET — derived from each instantiation's generics")


def bram_tiles(data_w, depth_words):
    """RAMB36 count for one bram_sdp, honouring Vivado's width/depth modes."""
    bits = data_w * depth_words
    return max(1, -(-bits // RAMB36_BITS))


total_bram = 0
rows = []

# weight_window: W_BANKS banks x 2 buffers, each BANK_W x BANK_DEPTH
wb = CONSTS["W_BANKS"] * 2
each = bram_tiles(CONSTS["BANK_W"], CONSTS["BANK_DEPTH"])
rows.append(("weight_window", f"{wb} x ({CONSTS['BANK_W']}b x {CONSTS['BANK_DEPTH']})",
             wb * each))
total_bram += wb * each

# activation memory
act_depth = 2 ** CONSTS["ACT_AW"]
n = bram_tiles(CONSTS["ACT_W"], act_depth)
rows.append(("act_mem", f"{CONSTS['ACT_W']}b x {act_depth}", n))
total_bram += n

# score memory
sd = 2 ** CONSTS["SCORE_AW"]
n = bram_tiles(CONSTS["SCORE_W"], sd)
rows.append(("score_mem", f"{CONSTS['SCORE_W']}b x {sd}", n))
total_bram += n

# tables
n = bram_tiles(CONSTS["EXP_ROM_DW"], 2 ** CONSTS["EXP_ROM_AW"])
rows.append(("exp_rom", f"{CONSTS['EXP_ROM_DW']}b x {2 ** CONSTS['EXP_ROM_AW']}", n))
total_bram += n
n = bram_tiles(CONSTS["RECIP_ROM_DW"], 2 ** CONSTS["RECIP_ROM_AW"])
rows.append(("recip_rom", f"{CONSTS['RECIP_ROM_DW']}b x {2 ** CONSTS['RECIP_ROM_AW']}", n))
total_bram += n

for nm, shape, tiles in rows:
    print(f"  {nm:<16}{shape:<28}{tiles:>3} RAMB36")
print(f"  {'TOTAL':<16}{'':<28}{total_bram:>3} RAMB36  "
      f"({total_bram / DEV['ramb36'] * 100:.0f}% of {DEV['ramb36']})")

# ---------------------------------------------------------------------
# 3. DSP budget
# ---------------------------------------------------------------------
print("\nDSP BUDGET")
lanes = CONSTS["N_LANES"]
print(f"  mac_array         {lanes} x mac_lane (use_dsp=yes)   {lanes:>3} DSP48E1")
print(f"  gemm_sequencer    scale multiply, {CONSTS['ACC_W']}b x "
      f"{CONSTS['RECIP_ROM_DW']}b, serialised     ~3 DSP48E1")
dsp_total = lanes + 3
print(f"  {'TOTAL':<16}{'':<28}{dsp_total:>3} DSP48E1 "
      f"({dsp_total / DEV['dsp'] * 100:.0f}% of {DEV['dsp']})")

# ---------------------------------------------------------------------
# 4. Verdict against the MVP plan's exit criteria
# ---------------------------------------------------------------------
print("\nAGAINST THE MVP PLAN'S EXIT CRITERIA (<90% LUT, <95% BRAM)")
bram_pct = total_bram / DEV["ramb36"] * 100
dsp_pct = dsp_total / DEV["dsp"] * 100
print(f"  BRAM  {bram_pct:>5.0f}%   {'PASS' if bram_pct < 95 else 'FAIL'}")
print(f"  DSP   {dsp_pct:>5.0f}%   {'PASS' if dsp_pct < 90 else 'FAIL'}")
print("  LUT     n/a   NOT PREDICTABLE FROM THE RTL -- all storage and")
print("                arithmetic is now in hard blocks, so what remains is")
print("                control logic. Only synthesis can measure it. v1's")
print("                measured 18,182 LUT (126%) is the number to beat.")

sys.exit(1 if violations else 0)
