#!/usr/bin/env python3
"""
board_translate.py -- turn the numbers the SILICON produced into Arabic.

WHAT MAKES THIS DIFFERENT FROM translate_hw.py

translate_hw.py proves the RTL is correct: it runs the job stream through
GHDL and diffs every tensor against the software model. Its `enc_hw` is
built from the SOFTWARE model's activation memory -- legitimate, because
check 1 proved the two are byte-identical, but the numbers that reach the
decoder never left a simulator.

This script closes that gap. Every tensor the fabric is responsible for is
read out of the board's own console dump instead. The elementwise work
between fabric passes -- bias, residual, LayerNorm, SiLU, requantisation --
still happens here on the host, exactly as it does on the PS in the real
driver, because the accelerator implements the matmul chain and the softmax
and nothing else. So the claim this script supports is precise:

    the 61,632 numbers on the arithmetic critical path came off the FPGA,
    and when you carry them the rest of the way they decode to the correct
    Arabic.

HOW THE SUBSTITUTION WORKS

translate_hw.run_layer() reaches into the fabric exactly seven times per
layer, always through the module-global read_act(), and every one of those
reads is paired with a check_act() that stamps a MARK into the tape. So the
board's dump contains one labelled block per read, in read order:

    L{n}.Q  L{n}.K  L{n}.V  L{n}.attn  L{n}.oproj  L{n}.ffn1  L{n}.ffn2

Seven per layer, six layers, 42 blocks -- which is exactly the `DUMP 42`
the board reports. Rebinding read_act to pop the next block off that queue
therefore substitutes silicon for simulation at every fabric boundary and
nowhere else. The queue is consumed strictly in order and every pop asserts
its own shape, so a tape that does not line up fails loudly rather than
quietly decoding something plausible.

Usage:
    board_translate.py <board_res_tape> [--text "Hello"]
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_REF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "model_ref"))
sys.path.insert(0, MODEL_REF)

import marian_fp32 as FP          # noqa: E402
import mirage_int8 as M           # noqa: E402
import sentencepiece as spm       # noqa: E402
import translate_hw as TH         # noqa: E402


# ==========================================================================
def parse_tape(path):
    """Read a MARK/value result tape into [(label, [int, ...]), ...].

    Accepts both the file translate_hw.py writes and the raw console log
    off the board, which carries the same blocks wrapped in BEGIN/END
    banners and CR line endings.
    """
    blocks, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("---", "#")):
                continue
            if line.startswith("MARK "):
                cur = (line[5:].strip(), [])
                blocks.append(cur)
            elif line.startswith(("DONE", "records", "values", "VECTOR",
                                  "ID", "cmd tape", "res tape", "stopping")):
                continue
            elif cur is not None:
                try:
                    cur[1].append(int(line))
                except ValueError:
                    pass
    return blocks


class BoardActs:
    """The fabric outputs, in the order run_layer() asks for them."""

    def __init__(self, blocks):
        # Score blocks stay inside the fabric -- the PS never reads them
        # back, so they are checked but not consumed here.
        self.queue = [(lab, vals) for lab, vals in blocks
                      if ".sc." not in lab]
        self.i = 0
        self.used = []

    def pop(self, ntok, n):
        if self.i >= len(self.queue):
            raise SystemExit(
                f"board tape ran out after {self.i} tensors -- the tape does "
                f"not match the sentence being decoded")
        lab, vals = self.queue[self.i]
        self.i += 1
        if len(vals) != ntok * n:
            raise SystemExit(
                f"{lab}: board returned {len(vals)} values, "
                f"the decode path wants {ntok} x {n} = {ntok * n}")
        self.used.append(lab)
        return np.array(vals, dtype=np.float64).reshape(ntok, n)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tape", help="board result tape or console log")
    ap.add_argument("--text", default="Hello",
                    help="the sentence the tape was built for")
    ap.add_argument("--reference", default=os.path.join(
                        MODEL_REF, "..", "sim", "xlate_res.txt"),
                    help="the tape the board was checked against")
    ap.add_argument("--scratch", default=None,
                    help="where to rewrite the job script (default: alongside "
                         "the tape)")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 72)
    print(f'MIRAGE on silicon: "{args.text}" -> Arabic')
    print("=" * 72)

    blocks = parse_tape(args.tape)
    acts = BoardActs(blocks)
    nvals = sum(len(v) for _, v in blocks)
    print(f"\n1. BOARD TAPE   {os.path.relpath(args.tape)}")
    print(f"   {len(blocks)} labelled blocks, {nvals} values")
    print(f"   {len(acts.queue)} of them are fabric outputs the PS reads back")
    if not blocks:
        raise SystemExit("   no MARK blocks in that file -- did the run "
                         "produce console output?")

    # ---- tokenise --------------------------------------------------------
    model = FP.Marian()
    w = model.w
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(FP.HF, "source.spm"))
    pieces = sp.encode(args.text, out_type=str)
    ids = [model.vocab[pc] for pc in pieces] + [FP.EOS_ID]
    ntok = len(ids)
    print(f"\n2. TOKENISE")
    print(f"   pieces      {pieces} + </s>")
    print(f"   ids         {ids}   ({ntok} tokens)")

    # ---- what the float32 model says the answer is -----------------------
    ref_ids = model.generate(np.array(ids))
    print(f"\n3. FLOAT32 REFERENCE (the answer we are trying to reproduce)")
    print(f"   ids         {ref_ids}")
    print(f"   translation {model.detokenise(ref_ids)!r}")

    # ---- six encoder layers, fabric outputs taken from the board ---------
    x = model.embed(np.array(ids), "encoder")
    scratch = args.scratch or os.path.join(
        os.path.dirname(os.path.abspath(args.tape)), "rebuild_cmd.txt")

    # run_layer() resolves read_act as a module global at call time, so
    # rebinding it here redirects every fabric read to the board without
    # touching translate_hw.py itself.
    TH.read_act = lambda fab, base, stride, nt, n: acts.pop(nt, n)

    mir = M.Mirage(scratch)
    print(f"\n4. ENCODER: 6 layers, every fabric tensor read from the board")
    for li in range(FP.N_LAYERS):
        x = TH.run_layer(mir, w, li, x, ntok)
        print(f"   layer {li}: consumed {acts.i:2d} of {len(acts.queue)} "
              f"board tensors")
    mir.close()
    enc_board = x

    if acts.i != len(acts.queue):
        raise SystemExit(f"   {len(acts.queue) - acts.i} board tensors "
                         f"left unread -- tape and decode path disagree")

    # ---- independent bit-exact check, recomputed here --------------------
    # The board already compared on-chip and printed a verdict; this repeats
    # it host-side, so a bug in the on-chip comparator cannot pass both.
    #
    # It compares against the REFERENCE TAPE, not against mir.expect. The
    # rebinding above feeds board values back into the model, so from layer 1
    # on the model's own expectation is partly derived from the board and
    # drifts once the board is wrong -- against a corrupted tape that counts
    # 327 where the board counts 112. The reference tape is the fixed thing
    # both sides were checked against, so using it makes this number and the
    # board's the same number, which is the only way disagreement between
    # them means anything.
    ref_blocks = dict(parse_tape(args.reference))
    n, fails = 0, []
    for lab, have in blocks:
        want = ref_blocks.get(lab)
        if want is None:
            fails.append((lab, -1, "not in the reference tape", None))
            continue
        if len(have) != len(want):
            fails.append((lab, -1, f"{len(want)} values", f"{len(have)}"))
            continue
        for i, (a, b) in enumerate(zip(want, have)):
            n += 1
            if a != b:
                fails.append((lab, i, a, b))
    print(f"\n5. BIT-EXACT RECHECK (host-side, independent of the board's own)")
    print(f"   {n} values compared across {len(mir.expect)} tensors")
    if fails:
        for f in fails[:10]:
            print("   FAIL", f)
        print(f"   {len(fails)} MISMATCHES -- the silicon and the model differ")
    else:
        print("   PASS: every byte the board produced matches the model")

    # ---- decode the board's encoder state --------------------------------
    print(f"\n6. DECODE (greedy, on the encoder state the board produced)")
    dec = [FP.DECODER_START]
    for _ in range(24):
        logits = model.decode_step(
            np.array(dec), enc_board.astype(np.float32))[-1]
        logits[FP.PAD_ID] = -np.inf
        nxt = int(logits.argmax())
        dec.append(nxt)
        if nxt == FP.EOS_ID:
            break
    hw_ids = dec[1:]
    print(f"   ids         {hw_ids}")
    print(f"   pieces      {[model.inv_vocab[i] for i in hw_ids]}")

    # ---- verdict ---------------------------------------------------------
    ref_s = model.detokenise(ref_ids)
    hw_s = model.detokenise(hw_ids)
    same = hw_ids == list(ref_ids)
    print("\n" + "=" * 72)
    print(f"  reference (float32, CPU) : {ref_s!r}")
    print(f"  from the FPGA            : {hw_s!r}")
    print("=" * 72)
    if fails:
        # Say this plainly, because the two lines above will usually still
        # agree: an int8 activation off by one rarely changes an argmax, so
        # the translation survives corruption that the byte comparison
        # catches immediately. The bit-exact number is the real verdict --
        # the Arabic is what makes it legible, not what makes it true.
        print(f"  BIT-EXACT FAIL -- {len(fails)} of {n} values differ from the")
        print("  reference. The translation above may well still read")
        print("  correctly; that is why the byte comparison is the verdict.")
    elif same:
        print("  MATCH. Every one of the 61,632 values on the arithmetic")
        print("  critical path came off the FPGA, and the translation is")
        print("  identical to the float32 reference.")
    else:
        print("  Bit-exact PASS but the translation differs -- decode-side")
        print("  problem, not a fabric problem.")
    print(f"  {time.time() - t0:.1f}s")
    print("=" * 72)
    return 0 if (same and not fails) else 1


if __name__ == "__main__":
    raise SystemExit(main())
