#!/usr/bin/env python3
"""tape_split.py -- separate the weights from the command tape.

WHY

A translation tape is 64.3 MB, and 98.6% of it is WROW records: 786,432 rows
of eight 32-bit words, 32 bytes of payload written as ~80 bytes of ASCII. Those
weights are the same MarianMT matrices on every run. Shipping them down the
command channel each time is what makes the tape too big for the board -- OCM
gives roughly 64 KB per tape, and even DDR-resident it is 64 MB over JTAG for
every sentence.

Splitting them apart:

    commands   64.3 MB -> 0.91 MB      regenerated per sentence
    weights            -> 25.2 MB      loaded once, reused

WHY A RUN CAN BECOME ONE RECORD

sw/board/vectors.c already REQUIRES the WROW index to start at 0 for a new
window and increment by one, and rejects the tape otherwise. So a run of WROW
records carries no information beyond (how many rows, what the rows are) -- the
indices are implied. That makes `WBLK <n_rows> <blob_offset>` a lossless
substitution rather than a lossy summary, which is why `verify` below can
demand byte-for-byte equality rather than a tolerance.

BLOB FORMAT

Row-major, 32 bytes per row, eight uint32 little-endian words -- little-endian
so the Cortex-A9 can load a row without swapping. Offsets are byte offsets.

USAGE

    tape_split.py split  <tape> <out_cmds> <out_blob>
    tape_split.py expand <cmds> <blob> <out_tape>
    tape_split.py verify <tape> [<cmds> <blob>]      split, expand, compare
"""

from __future__ import annotations

import os
import struct
import sys

ROW_WORDS = 8
ROW_BYTES = ROW_WORDS * 4


def _die(msg: str) -> "NoReturn":            # noqa: F821
    sys.stderr.write(f"tape_split: {msg}\n")
    raise SystemExit(1)


# ---------------------------------------------------------------- split
def split(tape: str, out_cmds: str, out_blob: str) -> tuple[int, int]:
    """Write the command tape and the weight blob. Returns (runs, rows)."""
    runs = rows = 0
    pending: list[bytes] = []      # rows of the run currently being collected

    with open(tape, "r") as fin, \
         open(out_cmds, "w", newline="") as fcmd, \
         open(out_blob, "wb") as fblob:

        def flush() -> None:
            nonlocal runs, pending
            if not pending:
                return
            offset = fblob.tell()
            fblob.write(b"".join(pending))
            # hex, unpadded, matching the tape's own convention for numbers
            fcmd.write(f"WBLK {len(pending):x} {offset:x}\n")
            runs += 1
            pending = []

        for lineno, line in enumerate(fin, 1):
            if not line.startswith("WROW"):
                flush()
                fcmd.write(line)
                continue

            f = line.split()
            if len(f) != 2 + ROW_WORDS:
                _die(f"{tape}:{lineno}: WROW has {len(f) - 2} words, "
                     f"expected {ROW_WORDS}")
            idx = int(f[1], 16)
            if idx != len(pending):
                # Exactly the invariant vectors.c enforces. If a tape ever
                # violates it, splitting would silently renumber the rows, so
                # refuse rather than "fix" it.
                _die(f"{tape}:{lineno}: WROW index {idx:x} where "
                     f"{len(pending):x} was due -- run is not sequential")
            pending.append(struct.pack("<8I", *(int(w, 16) for w in f[2:])))
            rows += 1

        flush()

    return runs, rows


# ---------------------------------------------------------------- expand
def expand(cmds: str, blob: str, out_tape: str) -> int:
    """Reconstruct the original tape. Returns the number of rows emitted."""
    rows = 0
    with open(blob, "rb") as fblob:
        weights = fblob.read()

    with open(cmds, "r") as fin, open(out_tape, "w", newline="") as fout:
        for lineno, line in enumerate(fin, 1):
            if not line.startswith("WBLK"):
                fout.write(line)
                continue
            f = line.split()
            if len(f) != 3:
                _die(f"{cmds}:{lineno}: malformed WBLK")
            n, offset = int(f[1], 16), int(f[2], 16)
            end = offset + n * ROW_BYTES
            if end > len(weights):
                _die(f"{cmds}:{lineno}: WBLK runs past the blob "
                     f"({end} > {len(weights)} bytes)")
            for i in range(n):
                base = offset + i * ROW_BYTES
                words = struct.unpack_from("<8I", weights, base)
                fout.write(f"WROW {i:x} " + " ".join(f"{w:08x}" for w in words)
                           + "\n")
                rows += 1
    return rows


# ---------------------------------------------------------------- verify
def verify(tape: str, cmds: str, blob: str) -> int:
    """split -> expand -> require byte-for-byte equality with the original.

    Round-tripping is the whole proof. If the reconstruction is identical then
    no information was dropped, and feeding the split form to the RTL is by
    definition the same run -- no need to spend twelve minutes of GHDL to
    re-derive a result we already hold.
    """
    runs, rows = split(tape, cmds, blob)
    rebuilt = cmds + ".rebuilt"
    back = expand(cmds, blob, rebuilt)

    orig_sz, cmd_sz, blob_sz = (os.path.getsize(p) for p in (tape, cmds, blob))
    print(f"  runs (window loads) : {runs}")
    print(f"  rows               : {rows} split, {back} rebuilt")
    print(f"  commands           : {orig_sz/1e6:.2f} MB -> {cmd_sz/1e6:.2f} MB "
          f"({orig_sz/cmd_sz:.0f}x smaller)")
    print(f"  weight blob        : {blob_sz/1e6:.2f} MB")

    # compare streaming; these files are tens of megabytes
    same = True
    with open(tape, "rb") as a, open(rebuilt, "rb") as b:
        while True:
            ca, cb = a.read(1 << 20), b.read(1 << 20)
            if ca != cb:
                same = False
                break
            if not ca:
                break
    if not same or rows != back:
        print("  ROUND-TRIP FAIL -- reconstruction differs from the original")
        return 1
    os.remove(rebuilt)
    print("  ROUND-TRIP PASS -- reconstruction is byte-identical")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "split" and len(argv) == 5:
        runs, rows = split(*argv[2:5])
        print(f"  {runs} window loads, {rows} rows")
        return 0
    if cmd == "expand" and len(argv) == 5:
        print(f"  {expand(*argv[2:5])} rows")
        return 0
    if cmd == "verify" and len(argv) in (3, 5):
        tape = argv[2]
        cmds = argv[3] if len(argv) == 5 else tape.replace(".txt", "_wblk.txt")
        blob = argv[4] if len(argv) == 5 else tape.replace("_cmd.txt", "_wgt.bin")
        return verify(tape, cmds, blob)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
