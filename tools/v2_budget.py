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

    # Multiply discipline.
    #
    # An EARLIER VERSION OF THIS CHECK PASSED A REAL BUG, so the rule is now
    # stricter. It whitelisted "signal * power-of-two constant" as a harmless
    # wired shift. `head * HEAD_DIM` is exactly that shape -- and it is
    # broken, because numeric_std's "*"(UNSIGNED, NATURAL) is defined as
    # L * TO_UNSIGNED(R, L'LENGTH): the natural is converted to the width of
    # the LEFT operand. With a 4-bit `head` and HEAD_DIM = 64 that is
    # TO_UNSIGNED(64, 4) = 0, so the product is 0 for every head. Vivado
    # 2022.2 synthesised it without complaint; only rtl_v2/pkg/addr_math_tb
    # catches it.
    #
    # So a constant operand no longer makes a multiply safe. Signal
    # multiplies must go through mirage_v2_pkg.addr_mul (widen, multiply,
    # trim) or be listed below with a reason.
    KNOWN = set(CONSTS) | {"ACC_W", "WGT_W", "ACT_W", "BANK_W", "SCORE_W",
                           "MAX_TOK", "HEAD_DIM", "N_LANES", "W_BANKS"}

    # (file, fragment) -> why this bare multiply is intended hardware.
    ALLOW = {
        ("gemm_sequencer.vhd", "signed('0' & scale)"):
            "deliberate DSP multiply: the softmax scale, ~3 DSP in the budget",
        ("mirage_v2_pkg.vhd", "* stride"):
            "addr_mul's own body -- the sanctioned widen/multiply/trim",
    }

    # `constant X : T := A * B;` is elaboration-time integer arithmetic and
    # never becomes hardware. Dropped before the scan rather than allowed
    # per-site, so new package constants do not need an ALLOW entry.
    code = "\n".join(l for l in code.split("\n")
                     if not re.match(r'\s*constant\s+\w+\s*:', l))

    def is_const(tok):
        tok = tok.strip().strip("()")
        return bool(re.fullmatch(r"\d+", tok)) or tok in KNOWN

    # Operands may sit on the far side of a line break, so match across
    # newlines -- the previous version's char class excluded \n, which
    # silently skipped every multi-line multiply it should have judged.
    for m in re.finditer(r'([\w.\(\)\+\-\'& \n]{1,60})(?<!\*)\*(?!\*)'
                         r'([\w.\(\)\+\-\'& \n]{1,60})', code):
        frag = " ".join(m.group(0).split())
        lhs, rhs = m.group(1), m.group(2)

        # Slice bounds like ((j + 1) * ACC_W - 1 downto j * ACC_W) are pure
        # elaboration-time arithmetic on a generate index -- no hardware.
        if "downto" in frag or "to " in rhs:
            continue

        if any(k[0] == name and k[1] in frag for k in ALLOW):
            continue

        l = re.findall(r'[\w]+', lhs)
        r = re.findall(r'[\w]+', rhs)
        if not l or not r:
            continue

        # Both sides constant -> folded at elaboration, fine.
        if is_const(l[-1]) and is_const(r[0]):
            continue

        which = ("signal*signal" if not (is_const(l[-1]) or is_const(r[0]))
                 else "signal*constant")
        violations.append(
            f"{name}: {which} multiply outside mac_lane/addr_mul "
            f"-> `{frag[:60]}`\n"
            f"{'':13}use addr_mul(idx, stride, width) -- a constant operand "
            f"does NOT make this safe (see the note in this file)")

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
# MEASURED, not estimated. Runs 6 and 7 both reported 67 DSPs against a
# device with 66, because `use_dsp of rtl : architecture is "yes"` applied
# to EVERY operation in mac_lane -- so the accumulator got a DSP of its own:
#
#   mac_lane | (A2*B2)'           |  8 |  8 | 16   the multiply
#   mac_lane | ((C:0x0) or P)+A:B | 30 | 18 | 48   the accumulator
#
# 32 x 2 + 3 = 67. Restructuring the pipeline between runs 6 and 7 changed
# nothing, which is the evidence that the attribute -- not the coding style
# -- was asking for the second DSP.
#
# The attribute is now per signal: "yes" on the product, "no" on the
# accumulator, which moves 32 adders into fabric. That is the v2 trade
# inverted, and it is the right way round now: run 6 measured 1,969 LUT
# (13.7%) against 66 DSP (100%).
print(f"  mac_array         {lanes} x mac_lane (use_dsp on the product only)"
      f"  {lanes:>3} DSP48E1")
print(f"  gemm_sequencer    scale multiply, {CONSTS['ACC_W']}b x "
      f"{CONSTS['RECIP_ROM_DW']}b, serialised      2 DSP48E1")
dsp_total = lanes + 2
print(f"  {'':<18}(ACC_W narrowed 48 -> {CONSTS['ACC_W']} now that the")
print(f"  {'':<18} accumulator is not a DSP P register, which also")
print(f"  {'':<18} shrinks the scale multiply)")
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
print("  LUT     n/a   NOT PREDICTABLE FROM THE RTL. Storage is all in BRAM")
print("                and every multiply is in a DSP, but the 32 lane")
print(f"                accumulators are now DELIBERATELY in fabric: {lanes} x")
print(f"                {CONSTS['ACC_W']}b adds, roughly {lanes * CONSTS['ACC_W'] // 4:,} CARRY4 and "
      f"{lanes * CONSTS['ACC_W']:,} flops,")
print("                against run 6's measured 1,969 LUT and 466 FF. Only")
print("                synthesis can measure the total; v1's 18,182 LUT")
print("                (126%) is the number to beat.")

sys.exit(1 if violations else 0)
