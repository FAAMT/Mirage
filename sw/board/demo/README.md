# Running the Mirage demo on hardware

You know FPGAs. You have not seen this design before. This tells you what it
is, what the demo proves, and exactly what to type.

Budget about ten minutes: four for the passing run, four for the negative
control, two for reading the output.

---

## 1. What the design is

Mirage is an **int8 systolic array that computes one transformer encoder
layer** — the matrix multiplies and the softmax, and nothing else. It sits in
the PL of a Zynq-7000 and is driven over AXI4-Lite from the ARM cores in the
PS.

Concretely, per encoder layer, the fabric does:

| it computes | it does not |
|---|---|
| Q, K, V projections | token embedding lookup |
| attention scores and softmax | bias adds |
| attention × V | residual connections |
| output projection | LayerNorm |
| FFN layer 1 and layer 2 | the SiLU activation |
| | quantisation and requantisation |
| | anything at all in the decoder |

The right column is elementwise, `O(D)` work. The left column is `O(D²)` and
is about 99.8% of the arithmetic. That split is the design: put the expensive,
regular part in fabric, leave the cheap, irregular part on the CPU.

**The chip does not translate anything by itself.** It has no vocabulary, no
embedding table, no decoder, no beam search, and not enough BRAM to hold one
weight matrix. Anyone who tells you the FPGA translates English to Arabic is
overselling it. What the FPGA does is the arithmetic that translation is made
of, and this demo shows the numbers it produces are correct to the byte.

**The board.** Digilent Cora Z7-07S, `xc7z007sclg400-1`. 512 MB of DDR3
(`MT41K256M16`), ARM cores at 650 MHz, fabric clock at 100 MHz. One micro-USB
cable carries both JTAG and the serial console — it is an FT2232H with JTAG on
interface 0 and the UART on interface 1.

**The bitstream already exists** at `build_cora/mirage_cora_wrapper.bit` and
closed timing on 2026-08-01. You do not need Vivado, a licence, or an AMD
account to run any of this.

---

## 2. What the demo proves

There is one honest claim and the demo is built to support exactly it:

> A recorded stream of 2,352 jobs, taken from a real MarianMT en→ar model
> translating the word "Hello", is replayed on the board. Every one of the
> 61,632 values the fabric produces is compared against what the verified
> software model says it should be. They match to the byte. Carrying those
> board-produced numbers the rest of the way through the decoder yields the
> correct Arabic.

Two things make that more than a demo that always prints a nice answer:

- **The comparison is byte-for-byte, zero tolerance.** Not cosine similarity,
  not "close enough". One int8 value in 61,632 being off by one is a failure.
- **There is a negative control, and you should run it.** `--negative` flips a
  single byte in a 64.3 MB tape — one activation, one bit of one int8 value —
  and the run must fail. If it passes, the check is worthless and so is the
  passing run.

What it does **not** prove: anything about speed. Weights reach the board over
JTAG at ~132 KiB/s, which makes the whole thing bandwidth-bound by a wide
margin. Any timing number you take from this run describes the JTAG cable, not
the array. Ignore the wall clock.

---

## 3. What to plug in

One cable. Micro-USB from the host to the Cora's **PROG/UART** port — the one
next to the power jack, not the USB-A host port.

Then check it:

```bash
cd Mirage/sw/board/demo
./demo.sh --check
```

Expected:

```
== 1. preflight
   board          FT2232H present on USB
   console        /dev/serial/by-id/usb-Digilent_Digilent_Adept_USB_Device_210370B9F5FA-if01-port0
   resolves to    /dev/ttyUSB0
   port opens     yes, at 115200 8N1
   openocd        /usr/bin/openocd

board is ready.
```

**Do not hardcode `/dev/ttyUSB1`.** The UART's number moves. It is `ttyUSB1`
when `ftdi_sio` has bound both FTDI interfaces, and `ttyUSB0` after openocd has
unbound interface 0 to claim JTAG over libusb. Always use the
`/dev/serial/by-id/...-if01-port0` name, which points at the UART interface
regardless of enumeration order. The scripts do this for you.

---

## 4. Run it

Three commands, in this order.

### The passing run — about four minutes

```bash
./demo.sh
```

Most of the four minutes is 25 MB of encoder weights crawling into DDR over
JTAG. Nothing is wrong; that is the cable.

### The negative control — about four minutes

```bash
./demo.sh --negative
```

This is the one that makes the first one mean something. **Run it on camera
immediately after the passing run.**

### Rehearsal, hardware untouched — three seconds

```bash
./demo.sh --offline
```

Decodes the last board log without touching the board. Useful for practising
the recording, useless as evidence.

---

## 5. Reading the output

### A passing run

The board's own verdict comes first, computed on-chip by the ARM core:

```
   board verdict: VECTOR REPLAY PASS -- silicon matches the reference bit-exactly.
```

Then the host repeats the comparison independently and decodes:

```
5. BIT-EXACT RECHECK (host-side, independent of the board's own)
   61632 values compared across 138 tensors
   PASS: every byte the board produced matches the model

6. DECODE (greedy, on the encoder state the board produced)
   ids         [1466, 1466, 0]
   pieces      ['▁مرحبا', '▁مرحبا', '</s>']

========================================================================
  reference (float32, CPU) : 'مرحبا مرحبا'
  from the FPGA            : 'مرحبا مرحبا'
========================================================================
  MATCH. Every one of the 61,632 values on the arithmetic
  critical path came off the FPGA, and the translation is
  identical to the float32 reference.
```

The comparison is done twice on purpose — once by the ARM core against a copy
of the expected results sitting in DDR, once here on the host against the
reference tape on disk. A bug in one comparator cannot pass both.

### The negative control

```
   112 MISMATCHES -- the silicon and the model differ
...
  BIT-EXACT FAIL -- 112 of 61632 values differ from the
  reference. The translation above may well still read
  correctly; that is why the byte comparison is the verdict.

NEGATIVE CONTROL PASSED -- one flipped byte was caught.
```

**112 out of 61,632 is the number to expect.** One flipped bit in the first
activation propagates into the first layer's Q projection and 111 other places.

Note what does *not* happen: the Arabic still comes out right. An int8
activation off by one almost never moves an argmax. That is precisely why the
translation is the presentation and the byte comparison is the verdict — a
demo that only checked the Arabic would have passed a corrupted run.

`--negative` exits 0 when the corruption **was** caught. That is the outcome
you want, so the mode's success and the run's failure are the same event.

---

## 6. When it goes wrong

Three failures account for essentially all of them. All three are silent,
which is why the preflight exists.

### "the board produced no console output"

Something else was reading the serial port, so the capture recorded nothing.
Run `./demo.sh --check`; it names the process and prints the `kill` command.

The usual culprit is a detached `screen`. `screen` detaches itself when it
cannot take over the terminal — under `make`, or with stdin not a tty — which
looks exactly like it exited instantly while it quietly goes on holding the
port. Check with `screen -ls` and kill what you find. Use `picocom` instead;
it fails honestly.

### The port is busy but nothing holds it

`stty` says `Device or resource busy`, and `fuser`, `lsof` and a scan of every
`/proc/*/fd` all come back empty. The FT2232H is wedged, not contended —
openocd reset the bridge while a reader had the UART open.

```bash
./demo.sh --reset
```

That is a `USBDEVFS_RESET` ioctl on the usbfs node: the software equivalent of
unplugging the cable, and it needs no root. Unplugging and replugging works
too. Note that after a reset the UART's ttyUSB number will probably have
changed, which is exactly why nothing should be hardcoded to it.

### The board is not found

`lsusb -d 0403:6010` returns nothing. Check the cable is in the PROG/UART port
and that the board has power.

### A note on trusting a PASS

If a run reports PASS suspiciously fast, check the log is actually from this
run:

```bash
ls -la build/ddr_console.log && date
```

The scripts delete both console logs before starting and refuse to proceed if
no new one appears, because a stale log from a previous passing run is
indistinguishable from a fresh one — and reading yesterday's PASS for a board
that was never asked anything is the worst failure this thing could have.

---

## 7. How it actually works

For anyone who wants to check the claim rather than take it.

```
  translate_hw.py            the reference run. Drives real MarianMT
  (already done, offline)    weights through the software fabric model
                             and RECORDS every job as a text tape.
        |
        |  xlate_cmd.txt  64.3 MB   the job stream
        |  xlate_res.txt  213 KB    what the fabric should return
        v
  tape_split.py              98.6% of the tape is weight rows, identical
                             every run. Split them out:
                               commands  0.91 MB  (per sentence)
                               weights  25.17 MB  (loaded once)
        |
        v
  openocd over JTAG          bitstream -> PL
                             ps7_init  -> DDR and clocks up
                             three load_image calls -> DDR
        |
        v
  the board                  vectors.c walks the tape, issues every job to
                             the fabric over AXI4-Lite, and compares each
                             result against the expected copy in DDR.
                             Prints all 61,632 values and a verdict.
        |
        |  the serial console, 115200 8N1
        v
  board_translate.py         parses the board's own printout, feeds those
                             numbers into the decoder, prints Arabic.
```

The substitution in the last step is the part worth understanding. The
reference script reaches into the fabric exactly seven times per layer, always
through one function, and each of those reads is paired with a `MARK` stamped
into the tape:

```
L{n}.Q   L{n}.K   L{n}.V   L{n}.attn   L{n}.oproj   L{n}.ffn1   L{n}.ffn2
```

Seven per layer × six layers = 42 blocks, which is exactly the `DUMP 42` the
board reports. `board_translate.py` rebinds that one function to pop the next
block off the board's printout instead. So every number crossing the fabric
boundary came off the FPGA, and everything else — bias, residual, LayerNorm,
SiLU, requantisation — runs on the host exactly as it would on the PS.

The queue is consumed strictly in order and every read asserts its own shape,
so a tape that does not line up fails loudly instead of quietly decoding
something plausible.

### What the board prints, in the raw

```
=== Mirage vector replay ===
ID       : 0x4d495247  OK          <- "MIRG" out of the AXI4-Lite ID register
cmd tape : 0x00400000
res tape : 0x00600000

--- BEGIN RESULTS ---
MARK L0.Q
4
9
-14
...
--- END RESULTS ---

records  : ACT 55296  WROW 786432  CFG 13680  JOB 2352  MARK 138  DUMP 42  DUMPS 96
values   : 61632 checked, 0 mismatched

VECTOR REPLAY PASS -- silicon matches the reference bit-exactly.
```

### If you want to watch it live

In one terminal:

```bash
cd ../bare && make console
```

In another, `./demo.sh`. The board's output scrolls past in the first. Note
that `demo.sh` opens the port itself, so **close the console before running
it** — two readers split the byte stream between them and both get garbage.

---

## 8. Files

| file | what it is |
|---|---|
| `demo.sh` | the driver. Preflight, board run, decode |
| `board_translate.py` | parses the board's printout and decodes it |
| `build/ddr_console.log` | the board's raw output from the last run |
| `build/xlate_cmd.corrupt.txt` | the corrupted tape, written by `--negative` |
| `../bare/ddr_replay.sh` | loads DDR over JTAG and captures the console |
| `../bare/cora.cfg` | the openocd target config |
| `../../../sim/xlate_cmd.txt` | the recorded job stream, 64.3 MB |
| `../../../sim/xlate_res.txt` | the expected results, the reference for both checks |
| `../../../build_cora/mirage_cora_wrapper.bit` | the bitstream |

Everything under `build/` is regenerated and gitignored.

### Environment overrides

| variable | default | for |
|---|---|---|
| `PORT` | the `-if01-port0` by-id name | forcing a specific serial node |
| `TEXT` | `Hello` | the sentence the tape was built for — changing this alone does **not** change what the board computes |
| `CAP_SECS` | 420 | how long the capture waits for a verdict |
| `CMD_TAPE` / `RES_TAPE` | the `xlate_*` pair | running a different recorded tape |

To demo a **different sentence** you must regenerate the tape with
`model_ref/translate_hw.py`, which needs GHDL and takes considerably longer.
The tape encodes the sentence; `TEXT` only labels the output.
