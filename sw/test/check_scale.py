#!/usr/bin/env python3
"""
check_scale.py -- diff mirage_split_scale (C) against split_scale (Python)

The C driver has to reproduce the reference's requantise arithmetic
exactly, including its rounding mode: Python's round() is
round-half-to-even and C's obvious floor(0.5 + x) is not. That
difference is invisible to the ops trace, because every multiplier in
test_ops.py is a power of two.

Run via `make check` in sw/, or directly:
    ./split_scale_dump scale_vectors.txt | ./check_scale.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model_ref"))

import mirage_int8 as M   # noqa: E402


def reference(mult):
    try:
        scale, shift = M.split_scale(mult)
    except (ValueError, AssertionError):
        return "RANGE"
    return f"{scale} {shift}"


def main():
    vectors = os.path.join(HERE, "scale_vectors.txt")
    got = {}
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        got[parts[0]] = " ".join(parts[1:])

    fails = 0
    checked = 0
    for raw in open(vectors, encoding="utf-8"):
        raw = raw.split("#")[0].strip()
        if not raw:
            continue
        mult = float.fromhex(raw)
        want = reference(mult)
        have = got.get(raw)
        checked += 1
        if have is None:
            print(f"FAIL {raw}: C produced no answer")
            fails += 1
        elif have != want:
            print(f"FAIL {raw} ({mult!r}): python={want} c={have}")
            fails += 1

    if fails:
        print(f"\nsplit_scale: {fails} of {checked} vectors disagree")
        return 1
    print(f"PASS  split_scale  ({checked} vectors match model_ref)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
