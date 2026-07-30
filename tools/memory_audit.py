#!/usr/bin/env python3
"""
memory_audit.py — classify every memory array in the RTL against what a
7-series FPGA can actually implement, before spending a Vivado run.

    ./tools/memory_audit.py

Why this exists
---------------
The 2026-07-29 run (runs/2026-07-29-vivado-2022.2-run2/) died on:

    ERROR: [Synth 8-3391] Unable to infer a block/distributed RAM for
    'bram_qkv_reg' ... the number of bits (786440) is too large

That is not an isolated bug. Three modules declare weight arrays that are
read inside a 12x5 nested loop -- 60 concurrent reads of one array. BRAM
gives you 2 ports, LUTRAM up to 16. 60 is not implementable by any memory
primitive on any device, so Vivado degrades to registers and either errors
(above the dissolve limit) or, worse, SILENTLY SUCCEEDS with a register
file far larger than the device.

The silent case is the dangerous one: it produces a plausible-looking
utilisation number that is meaningless. This script finds both.

What blocks BRAM inference
--------------------------
  * more than 2 read ports (BRAM) / 16 (LUTRAM)
  * an array-wide reset -- `sig <= (others => (others => '0'))`.
    BRAM contents have no reset, so this alone forces registers.
  * asynchronous (unclocked) read

Heuristic, not a parser. It is calibrated against Vivado's own output:
multi_head.bram_qkv must come out at 786,440 bits, which is the figure
Vivado reported.
"""
import re, sys, glob, pathlib

DEV = dict(name="xc7z007sclg400-1", ramb36=50, bram_bits=50 * 36 * 1024, ff=28800, lut=14400)
# Vivado's default dissolveMemorySizeLimit. NOTE: the "786440" in the error
# message is the value you would have to RAISE the limit to, not the current
# limit. Above this, a non-inferrable memory is a hard error; below it, Vivado
# quietly dissolves it into flip-flops.
DISSOLVE_LIMIT = 65536

CONST = dict(NUM_HEADS=8, HEAD_DIM=64, MODEL_DIM=512, TOKEN_DIM=512, TOK_CONTEXT=4,
             TILE_H=12, TILE_W=5, MM_IN_DATA_W=8, MM_OUT_DATA_W=32, MHA_DATA_W=8,
             WEIGHT_SLICE=512, QKV_BRAM_DATA_W=16, SCORE_BRAM_DATA_W=21, DATA_W=8)
CONST['BASE_OFFSET']     = 3 * (CONST['HEAD_DIM'] * CONST['MODEL_DIM'])
CONST['QKV_BRAM_ELEM']   = 3 * CONST['TOK_CONTEXT'] * CONST['NUM_HEADS'] * CONST['HEAD_DIM']
CONST['SCORE_BRAM_ELEM'] = CONST['TOK_CONTEXT'] ** 2 * CONST['NUM_HEADS']


def ev(expr):
    try:
        return int(eval(expr, {"__builtins__": {}}, CONST))
    except Exception:
        return None


def loop_fanout(txt, sig):
    """Largest 'for a in X to Y loop' nest enclosing a read of sig.

    A read inside `for i in 0 to 11 / for j in 0 to 4` is 60 simultaneous
    read ports, not one.
    """
    best = 1
    for m in re.finditer(rf'\b{re.escape(sig)}\s*\(', txt):
        # walk backwards over enclosing loop headers still open at this point
        prefix = txt[:m.start()]
        opens = re.findall(r'for\s+\w+\s+in\s+(.+?)\s+to\s+(.+?)\s+loop', prefix, re.I)
        ends = len(re.findall(r'\bend\s+loop\b', prefix, re.I))
        active = opens[ends:] if ends < len(opens) else []
        fan = 1
        for lo, hi in active:
            a, b = ev(lo.strip()), ev(hi.strip())
            if a is not None and b is not None and b >= a:
                fan *= (b - a + 1)
        best = max(best, fan)
    return best


ARRAY_RE = (r'type\s+(\w+)\s+is\s+array\s*\(\s*([^)]*?)\s*\)\s*of\s+'
            r'(signed|unsigned|std_logic_vector)\s*\(\s*(.+?)\s+downto\s+(.+?)\s*\)')


def array_types(txt):
    """name -> (depth, width) for every constrained array-of-vector type."""
    out = {}
    for m in re.finditer(ARRAY_RE, txt, re.I | re.S):
        tname, rng, _, hi, lo = m.groups()
        rm = re.match(r'(.+?)\s+to\s+(.+)$', rng, re.I)
        if not rm:
            continue
        a, b, h, l = ev(rm.group(1)), ev(rm.group(2)), ev(hi), ev(lo)
        if None in (a, b, h, l):
            continue
        out[tname.lower()] = (b - a + 1, h - l + 1)
    return out


# Array types are often declared in types_pkg, not in the module that uses
# them (qk_score_bram's bram_scores_type is). Collect package types first so
# module-level signals of those types are not missed.
PKG = {}
for pkg in glob.glob("rtl/pkgs/*.vhd"):
    PKG.update(array_types(pathlib.Path(pkg).read_text()))

rows = []
for path in sorted(glob.glob("rtl/**/*.vhd", recursive=True)):
    if "/pkgs/" in path:
        continue
    txt = pathlib.Path(path).read_text()
    # module-local type declarations shadow/extend the package ones
    known = dict(PKG)
    known.update(array_types(txt))
    seen = set()
    # one pass over signal declarations, resolving the type from either scope.
    # `(?::=[^;]*)?` so a signal with a default initialiser is still matched.
    for sig, tname in re.findall(r'signal\s+(\w+)\s*:\s*(\w+)\s*(?::=[^;]*)?;', txt, re.I):
        if tname.lower() not in known or sig in seen:
            continue
        seen.add(sig)
        depth, width = known[tname.lower()]
        bits = depth * width
        if bits < 2048:            # register file by intent, not a memory
            continue
        rows.append(dict(
            file=path.replace("rtl/", ""), sig=sig, depth=depth, width=width, bits=bits,
            fanout=loop_fanout(txt, sig),
            wide_reset=bool(re.search(rf'\b{sig}\s*<=\s*\(\s*others', txt)),
        ))

rows.sort(key=lambda r: -r['bits'])

print(f"Device {DEV['name']}: {DEV['ramb36']} x RAMB36 = {DEV['bram_bits']:,} bits, "
      f"{DEV['ff']:,} FF, {DEV['lut']:,} LUT")
print(f"Vivado default dissolveMemorySizeLimit: {DISSOLVE_LIMIT:,} bits\n")
hdr = f"{'file':<26}{'signal':<15}{'bits':>10}{'rd ports':>9}{'reset':>7}  verdict"
print(hdr); print("-" * len(hdr) + "----------------------------")

blocked, silent, ok, total = [], [], [], 0
for r in rows:
    total += r['bits']
    reasons = []
    if r['fanout'] > 2:
        reasons.append(f"{r['fanout']} read ports (BRAM=2, LUTRAM<=16)")
    if r['wide_reset']:
        reasons.append("array-wide reset")
    # Storage is the cheap part. The expensive part is the read-port MUX tree:
    # N dynamic reads of a depth-D array cost N parallel D:1 muxes.
    #
    # CALIBRATION (run 4, 2026-07-30): qk_score_bram.bram_score holds only
    # 2,688 bits -- which an earlier version of this script called "affordable"
    # -- but 32 reads of a 128-deep array cost 11,587 LUT, 80% of the device.
    # ports*width*depth/8 predicts 10,752 there. ONE data point: treat as
    # order-of-magnitude, not a number to quote.
    mux_lut = (r['fanout'] * r['width'] * r['depth'] // 8) if r['fanout'] > 2 else 0
    r['mux_lut'] = mux_lut

    if not reasons:
        verdict, bucket = "BRAM-inferrable", ok
    elif r['bits'] > DISSOLVE_LIMIT:
        verdict, bucket = "HARD ERROR [Synth 8-3391]", blocked
    elif r['bits'] > DEV['ff'] * 0.10 or mux_lut > DEV['lut'] * 0.10:
        verdict, bucket = f"QUIET -> {r['bits']:,} FF + ~{mux_lut:,} LUT mux", silent
    else:
        verdict, bucket = f"registers, {r['bits']:,} FF + ~{mux_lut:,} LUT", ok
    bucket.append(r)
    print(f"{r['file']:<26}{r['sig']:<15}{r['bits']:>10,}{r['fanout']:>9}"
          f"{'YES' if r['wide_reset'] else '-':>7}  {verdict}")
    for x in reasons:
        print(f"{'':<26}{'':<15}{'':>10}{'':>9}{'':>7}    - {x}")

print("-" * (len(hdr) + 28))
print(f"declared memory: {total:,} bits = {total / DEV['bram_bits'] * 100:.0f}% of all device BRAM")
print(f"\n  {len(blocked)} array(s) will HARD ERROR — synthesis cannot proceed")
print(f"  {len(silent)} array(s) would SILENTLY become register files "
      f"({sum(r['bits'] for r in silent):,} FF vs {DEV['ff']:,} available)")
mux_total = sum(r.get('mux_lut', 0) for r in silent + ok)
print(f"\n  estimated read-port MUX cost of the non-blocked arrays: ~{mux_total:,} LUT "
      f"({mux_total / DEV['lut'] * 100:.0f}% of device)")
print("  order-of-magnitude only -- see the calibration note in the source.")
reg_ok = sum(r['bits'] for r in ok)
print(f"  {len(ok)} array(s) are fine (BRAM-inferrable, or small register files "
      f"totalling {reg_ok:,} FF)")
hard = sum(r['bits'] for r in blocked)
print(f"\n  blocked weight storage alone: {hard:,} bits = {hard/8/1024:.0f} KB "
      f"= {hard/DEV['bram_bits']*100:.0f}% of device BRAM")

if blocked or silent:
    print("\nDo NOT apply Vivado's suggested workaround")
    print("  set_param synth.elaboration.rodinMoreOptions "
          "{rt::set_parameter dissolveMemorySizeLimit 786440}")
    biggest = max(r['bits'] for r in blocked) if blocked else 0
    print(f"  It converts the largest such array into {biggest:,} individual flip-flops "
          f"on a device with {DEV['ff']:,}.")
    print("  It does not make the design fit; it makes Vivado grind instead of erroring.")

sys.exit(1 if (blocked or silent) else 0)
