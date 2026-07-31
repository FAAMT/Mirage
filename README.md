# Mirage — Vivado resource-utilisation package

Prepared 2026-07-29 from `~/projects/agentic-chip-design`.

| source | commit |
|---|---|
| `agentic-chip-design` (parent) | `a832d74` — 2026-07-26 |
| `Mirage` (submodule, `FAAMT/Mirage` @ main) | `ed7dd77` — 2026-06-26, *"Restarting MIRAGE accelerator..."* |

The submodule is level with `origin/main`; `ed7dd77` is the latest Mirage. The
RTL under `rtl/outputs/` is the newer agent-produced work that lives in the
parent repo, not in the submodule.

## What this is for

Getting **accurate resource numbers out of Vivado without building a block
diagram**. One run, two phases:

- **Phase 1** synthesises each module standalone in out-of-context mode — no
  I/O buffers, no pruning of logic dangling off unconnected top pins. This is
  the per-block profile: who owns the DSPs, who owns the BRAM.
- **Phase 2** synthesises `mirage_sizing_subset`, which holds each
  **synthesisable root** module exactly once, with `-flatten_hierarchy none`.

Phase 2 exists because summing phase 1 double-counts: `qkv_proj_bram` and
`quantizer` are counted both standalone and inside `qkv_score_calc`, and
`smx_cache`/`smx_v_calc` both standalone and inside `smx_core`. The subset row
has each block once.

> **Read this before quoting any number.** Phase 2 covers **6 of 9** root
> modules. `ffn`, `multi_head` and `qkv_proj_calc` cannot be synthesised on any
> FPGA as written — their weight arrays are read 60-wide inside a 12×5 loop and
> BRAM offers 2 read ports. They appear as `BLOCKED` rows. The design declares
> **212% of the device's BRAM**. `KNOWN_ISSUES.md` §7 has the full picture; it is
> the dominant open issue and it is architecture work, not a packaging fix.

## Run it

Two local gates first. Together they cost about two seconds and have already
saved three Vivado round-trips:

```bash
./tools/vivado_lint.sh      # Vivado-class VHDL errors, without Vivado
./tools/memory_audit.py     # which memories can actually be mapped
```

`memory_audit.py` currently exits non-zero **by design** — see `KNOWN_ISSUES.md`
§7. That is the real state of the design, not a broken script.

Then:

```bash
cd mirage-vivado-2026-07-29
mkdir -p reports                       # -log is opened before the script runs
vivado -mode batch -source tcl/synth_utilization.tcl \
       -log reports/vivado.log -journal reports/vivado.jou
```

Part is `xc7z007sclg400-1` (Cora Z7-07S) — set `part` at the top of the tcl to
retarget. Synthesis only; no place, route, or bitstream.

**Vivado 2020 / 2022 / 2023 all work** — nothing in the script needs a recent
release (`read_vhdl -vhdl2008`, `synth_design -mode out_of_context`,
`report_utilization -return_string`, `-flatten_hierarchy none` all predate
2020). The one genuine version risk is **VHDL-2008 support in the synthesiser**,
which is partial and weaker in 2020 than 2022. The 2D array ports
(`int8_gemm_array(0 to 11, 0 to 4)`) and `signed`-element array types are what
would trip it.

**One such gap has already bitten us and is fixed** — readable `out` ports,
`[Synth 8-10557]`, in three files. See `KNOWN_ISSUES.md` §0 and the run log in
`runs/`. `tools/vivado_lint.sh` now catches that whole class locally.

**Phase 0 catches anything else in seconds.** The script starts with
`synth_design -rtl`, an elaboration-only pass. If it fails, the errors are a
Vivado version limit, not a design fault — these files analyse and elaborate
cleanly under GHDL 4.1.0. Send the phase-0 errors back rather than editing RTL
to appease them. The run continues to phase 1 regardless, since individual
modules may synthesise even when the container does not.

If you have both, use **2022** — strictly better VHDL-2008 coverage than 2020.

### If you run it from the GUI instead

The 2026-07-29 attempt used a Vivado **project** flow, which cost us two things
worth avoiding:

1. **No `-mode out_of_context`.** The container has roughly **5,900 bits of
   top-level ports** against a device with ~100 user I/O. Without OOC, Vivado
   inserts I/O buffers for all of it and the utilisation numbers are distorted.
   In a project, set the synthesis run's *More Options* to
   `-mode out_of_context`.
2. **One bad file killed the whole run.** A project synth of the container is
   all-or-nothing. The batch script's phase 1 isolates each module, so one
   failure still leaves you the rest.

Output in `reports/`:

- `summary.csv` + a printed table — LUT / FF / DSP48E1 / RAMB36 per module,
  each row tagged `root`, `child-of-*`, `BLOCKED`, or `SUBSET-TOTAL`
- `util_CONTAINER_hier.rpt` — the subset total broken back down per block
- `util_<module>.rpt` — full utilisation report per module

No timing reports: out-of-context synthesis with no XDC leaves every path
unconstrained, so the numbers would be noise. Timing is a question for after
there is a real top-level and a clock constraint.

One module failing to synthesise does not abort the batch; it is marked
`FAILED` in the table and the run continues.

## Layout

```
rtl/pkgs/        types_pkg.vhd (patched, see below) + types_pkg_body.vhd
rtl/mirage/      synthesisable RTL from the Mirage submodule
rtl/outputs/     agent-produced RTL from the parent repo's outputs/
rtl/integration/ mirage_sizing_subset.vhd — the phase-2 top (6 root modules)
                 mirage_sizing_container.vhd — all 9 roots; blocked by §7
rtl_excluded/    softmax.vhd, the one module that still does not analyse
patches/         the qk_score_bram fix (already applied; kept for reversal)
tb/              testbenches; not read by the synth script, kept for ghdl sim
tcl/             the synthesis script
tools/           vivado_lint.sh  — Vivado-class VHDL errors, without Vivado
                 memory_audit.py — which memories can actually be mapped
runs/            past Vivado runs, logs kept as evidence (identifiers scrubbed)
reference/       W6/W11/W12 reports with the pre-synth estimates, and the
                 original synth.tcl this script replaces
reports/         run output lands here
```

## Modules (10 synthesised + subset; 3 blocked)

| module | source | role |
|---|---|---|
| `gemm_array` | Mirage | 12×5 int8 systolic MAC tile — the DSP consumer |
| `quantizer` | Mirage | 60-lane combinational requantiser *(child)* |
| `qkv_proj_bram` | Mirage | Q/K/V projection buffer *(child)* — quietly becomes 49,152 FF, §7 |
| `qk_score_bram` | Mirage | QK^T score BRAM — **patched**, see KNOWN_ISSUES §1 |
| `qkv_proj_calc` | Mirage | projection datapath — **BLOCKED**, §7 (`bram_qkv` 786 kb, 60 rd ports) |
| `qkv_score_calc` | Mirage | QK^T datapath (instantiates `qkv_proj_bram` + 2× `quantizer`) |
| `multi_head` | Mirage | MHA control FSM — **BLOCKED**, §7 (`bram_qkv` 786 kb, 60 rd ports) |
| `taylor_exp` | Mirage | polynomial exp for softmax |
| `ffn` | Mirage | FFN control + datapath — **BLOCKED**, §7 (2.2 Mb of weights, 60 rd ports) |
| `smx_cache` | outputs/ | softmax score cache *(child)* |
| `smx_v_calc` | outputs/ | softmax × V *(child)* |
| `smx_core` | outputs/ | softmax top (instantiates the two above) |
| `weight_stream_ctrl` | outputs/ | DDR weight-stream sequencer |

All generics have defaults; nothing needs overriding. Defaults are
`TILE_H=12`, `TILE_W=5`, `DATA_W=8`, `NUM_HEADS=8`, `TOK_CONTEXT=4`,
`HEAD_DIM=64`.

## Three things to know before you read the numbers

**1. `rtl/pkgs/types_pkg.vhd` is patched, not upstream.** Upstream renamed
`mac_array` → `int8_gemm_array` but never updated `ffn`, `softmax`, and
`multi_head`, which still reference the old name. The shipped package adds a
three-line `subtype mac_array is int8_gemm_array;` alias so those modules
compile. Without it, `ffn` and `multi_head` do not analyse at all. The
untouched original is at `reference/types_pkg.upstream.vhd` for diffing.

**2. `qk_score_bram.vhd` is shipped patched too.** One spurious
`std_logic_vector()` cast onto a `signed` target. It is a score BRAM and feeds
the RAMB36 figure directly, so leaving it out would have understated memory.
Original at `reference/qk_score_bram.unpatched.vhd`; reversal is one `cp`.

**3. The container is a sizing harness, not a working encoder layer.** It wires
only what the source already defines unambiguously; every other module port is
brought out to a container pin rather than connected, because the design has
not decided those connections yet. Three real gaps stand between this and a
functioning top-level — `multi_head` and `qkv_proj_calc` have no result-return
port, `taylor_exp.data_out` (252b) does not match `ffn.data_in` (272b), and
four MAC clients share one `gemm_array` with no arbiter. All three are written
up in `KNOWN_ISSUES.md` §4–6. Inventing that glue would have been worse than
leaving it out: fabricated control paths that never toggle let Vivado
constant-propagate and delete whole blocks, which under-reports.

So: treat the TOTAL row as a sound upper bound with nothing double-counted and
nothing pruned. It will move once the arbiter and adapters are real design.

## Measured, run 4 (2026-07-30, partial)

Device: LUT 14,400 | FF 28,800 | DSP48E1 66 | RAMB36 50

| module | kind | LUT | FF | DSP | BRAM |
|---|---|---|---|---|---|
| `gemm_array` | root | 4,850 | 625 | 0 | 0 |
| `quantizer` | child | 0 | 0 | 0 | 0 |
| `taylor_exp` | root | 42 | 37 | 0 | 0 |
| `qk_score_bram` | root | **11,587** | 2,813 | 0 | 0 |
| `smx_cache` | child | 1,956 | 6,537 | 0 | 0 |
| `smx_v_calc` | child | 2,043 | 3,826 | **2** | 0 |
| `smx_core` | root | 1,668 | 3,173 | 2 | 0 |
| `weight_stream_ctrl` | root | 35 | 77 | 0 | 0 |
| **roots so far** | | **18,182 (126%)** | 6,725 (23%) | 2 (3%) | **0 (0%)** |

Five of nine roots. **Already 126% of device LUTs**, with `qkv_score_calc`
pending and `ffn`/`multi_head`/`qkv_proj_calc` blocked.

**Zero BRAM and zero LUT-as-Memory anywhere.** Every memory in the design became
flip-flops plus combinational mux trees. `qk_score_bram` is the clearest case:
2,688 bits of storage costing 11,587 LUT (80% of the device) in read-port muxes.

Three numbers in that table are not usable as costs:

- `taylor_exp` **42 LUT** — its `bram_rd`, `addr_off_0`, `data_out` and
  `done_taylor` are all undriven (W3/W5 fixes never applied), so it was almost
  entirely optimised away.
- `smx_core` **1,668 LUT** — *smaller* than `smx_cache` + `smx_v_calc` (3,999),
  because `score_row`, `start` and `score_row_valid` are unconnected inside it and
  Vivado deleted the paths that consume them. An under-estimate, not a saving.
- `quantizer` **0** — correct. A constant 16→8 shift is pure bit selection.

## The old DSP prediction, for contrast

`reference/w11-resource-map.md` estimates 60 of 66 DSP48E1 in `gemm_array`
alone (91%), against a <90% acceptance bar, with the 12×5 tile time-multiplexed
into 2×6×5 as the ready mitigation. `gemm_array`'s actual DSP count is the
number worth looking at first. The same report estimates RAMB36 at ~84% and
LUT/FF comfortably under 30% — but note the RAMB36 figure assumes the FFN
hidden/output weights (262 KB) are DDR-streamed rather than resident, which is
what `weight_stream_ctrl` exists to do.

Those are pre-synthesis estimates from code inspection. Nothing in
`reference/` contains a real Vivado number — this run produces the first ones.

## Run history

| date | tool | result |
|---|---|---|
| 2026-07-29 | Vivado 2022.2, project flow | **failed** — `[Synth 8-10557]` readable `out` ports, 9 sites in `ffn.vhd`. Fixed; see §0. |
| 2026-07-29 | Vivado 2022.2, batch script | **failed** — `[Synth 8-3391]`, `multi_head.bram_qkv` (786,440 b) not mappable to any memory. See §7. |
| 2026-07-30 | Vivado 2022.2, batch, subset | **phase 0 OK, 0 errors.** First real numbers: `gemm_array` = 6,818 LUT / 1,056 CARRY4 / **0 DSP** / 240 latches, vs W11's predicted 60 DSP. Latch root-caused and fixed; see §8. Run stalled in `qkv_proj_bram` (§7). |

Vivado parsed 16 of 16 sources and objected to exactly one file, so the source
set is otherwise acceptable to 2022.2. It aborts after the first failing unit,
which is why the same defect in `qkv_score_calc.vhd` and `weight_stream_ctrl.vhd`
went unreported — the lint gate found those.

Logs have had the host name, owner, PID and memory size scrubbed, and two
regenerable Vivado artifacts removed: `vivado.pb` (a binary message DB that
embedded the hostname) and `project.wdf` (webtalk usage telemetry — a project
UUID plus tool-usage counters, dropped before this went to a remote).

That scrubbing was applied to `runs/` only at first, and the root `vivado.log`
escaped it — it reached the remote in the first push carrying both a username
and the machine name from Vivado's report header. **Run
`./tools/scrub_check.sh` before every push.** It has no `head` and no match
limit, it prints a per-pattern verdict so silence is evidence rather than an
artefact of paging, and it covers the two header forms Vivado uses to name
the build machine, bare asset-tag hostnames, home paths, IPs, MACs, emails,
assigned credentials, key blocks, provider tokens and license servers. Exit
code 1 means do not push.

The script is necessarily excluded from its own scan, since it spells out the
patterns it hunts for. Nothing else is exempt — if a document needs to
discuss one of these formats, word it around the literal string rather than
adding an exemption.

## Verified before shipping

`tools/vivado_lint.sh` passes both ways: all 17 sources analyse and all 13
module tops plus `mirage_sizing_container` elaborate under GHDL 4.1.0
(`--std=08`), and the `--std=93` pass finds no readable `out` ports, so Vivado
will not raise `[Synth 8-10557]` again. Sources analyse in the exact order the tcl
reads them. The tcl parses and its table plumbing was dry-run against a sample
Vivado utilisation report. GHDL checks language correctness and elaboration,
not synthesisability — Vivado may still object to something GHDL accepts, which
is why a failing module is marked `FAILED` and the batch continues.

## Numerical verification (2026-07-31)

Run 9 met every resource and timing target and still computed wrong numbers.
`TEST_PLAN.html` is the plan that closed that; `PROVENANCE.md` records every
change from the code as received.

```
sim/build.sh                    analyse rtl_v2 + testbenches
sim/build.sh <tb> --ieee-asserts=disable --stop-time=100ms
                                run one VHDL unit testbench
model_ref/test_ops.py           op-level co-simulation, RTL vs bit-exact model
model_ref/diagnose.py           per-stage int8 accuracy vs float32 (fast, no GHDL)
model_ref/translate_hw.py       "Hello" -> Arabic, six encoder layers on the RTL
```

`model_ref/` needs a venv with numpy and sentencepiece, and the MarianMT
checkpoint:

```
python3 -m venv model_ref/.venv && model_ref/.venv/bin/pip install numpy sentencepiece
cd model_ref/hf && for f in config.json vocab.json source.spm target.spm pytorch_model.bin; do
  curl -sSLO "https://huggingface.co/Helsinki-NLP/opus-mt-en-ar/resolve/main/$f"; done
```

Neither the venv nor the checkpoint is tracked. `model_ref/torchbin.py` reads
the legacy PyTorch pickle directly into numpy, so torch is not required.

Result: 13 defects found and fixed; every job type bit-exact against the model;
"Hello" decodes to token 1466 مرحبا, identical to the float32 reference.
