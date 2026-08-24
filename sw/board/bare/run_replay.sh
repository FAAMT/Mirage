#!/usr/bin/env bash
# =====================================================================
# run_replay.sh -- replay a verified command script on the board and
# diff what silicon returns against what simulation returned
#
#   ./run_replay.sh [cmd_tape] [res_tape]
#
# defaults: ../../../sim/ops_cmd.txt and ../../../sim/ops_res.txt
#
# WHY THIS IS A SCRIPT AND NOT A MAKE RECIPE
#
# Ordering. The board starts printing the moment it is resumed, and
# openocd returns only after that, so a capture started afterwards
# catches the tail of the run at best and nothing at all at worst. The
# capture has to be running first, which means a background job, a way
# to know when the run is over, and a bound in case it never is. That is
# three things make recipes are bad at.
#
# HOW THE CAPTURE ENDS
#
# Not on a timer. sed quits at the verdict line, so a normal run takes
# as long as the run takes; the timeout is only there for a board that
# says nothing. An earlier version of this capture used `stty min 0
# time N`, which makes read() return zero bytes on timeout -- cat sees
# that as EOF and exits about a second in. Two runs then captured an
# empty file, which looks exactly like a dead board. Plain `raw` blocks
# for real data, which is what is wanted.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-/dev/ttyUSB1}
CAP_SECS=${CAP_SECS:-60}

# --- capture mode: re-entry under sg, so the quoting stays readable ---
if [ "${1:-}" = "--capture" ]; then
    # Exit rather than fall through if the port cannot be opened -- see the
    # long note on the same branch in ddr_replay.sh. A failed exec redirect
    # does not stop the shell, and the fall-through ends in a stale log being
    # read as a fresh PASS.
    stty -F "$2" 115200 raw -echo -crtscts || {
        echo "capture: cannot configure $2 -- another reader holds it" >&2
        exit 3; }
    exec timeout "$4" sed -u -e '/^VECTOR REPLAY /q' \
                             -e '/^stopping:/q' < "$2" > "$3"
    echo "capture: cannot read $2 -- another reader holds it" >&2
    exit 3
fi

CMD_TAPE=${1:-../../../sim/ops_cmd.txt}
RES_TAPE=${2:-../../../sim/ops_res.txt}
BUILD=build
LOG=$BUILD/replay_console.log
GOT=$BUILD/replay_got.txt

CMD_BASE=${CMD_BASE:-0x00010000}
RES_BASE=${RES_BASE:-0x00020000}
BIT=${BIT:-../../../build_cora/mirage_cora_wrapper.bit}
ELF=$BUILD/mirage_vectors.elf

for f in "$CMD_TAPE" "$RES_TAPE" "$BIT"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
[ -c "$PORT" ] || { echo "no serial console at $PORT -- is the board plugged in?"; exit 1; }

# --- 1. the interpreter has to be right before the board can be wrong -
echo "== hostcheck =="
make --no-print-directory hostcheck CMD_TAPE="$CMD_TAPE" RES_TAPE="$RES_TAPE" || exit 1
echo

# --- 2. build ---------------------------------------------------------
echo "== build =="
make --no-print-directory APP=vectors CMD_BASE="$CMD_BASE" RES_BASE="$RES_BASE" \
    || exit 1
echo

# The tapes and the program share OCM. The linker knows where the
# program ends; check it here rather than discovering the overlap as
# corrupted test data. (vectors.c checks this too, at runtime -- this
# one just fails before the board is touched.)
top=$(grep -oE '^ *0x[0-9a-f]+ +__stack_top' "$BUILD/mirage_vectors.map" \
      | awk '{print $1}' | tail -1)
if [ -n "$top" ] && [ "$((top))" -gt "$((CMD_BASE))" ]; then
    printf 'program ends at %s, above the tape at %s -- move CMD_BASE up\n' \
        "$top" "$CMD_BASE"
    exit 1
fi
printf 'program ends at %s; tapes at %s and %s\n\n' \
    "${top:-?}" "$CMD_BASE" "$RES_BASE"

# --- 3. capture first, then run --------------------------------------
mkdir -p "$BUILD"
: > "$LOG"
echo "== capture on $PORT =="
sg dialout -c "./run_replay.sh --capture $PORT $LOG $CAP_SECS" &
CAP=$!

echo "== load and resume =="
openocd -f cora.cfg \
  -c "init" \
  -c "targets zynq.cpu0" \
  -c "halt" \
  -c "pld load 0 $BIT" \
  -c "load_image $ELF" \
  -c "load_image $CMD_TAPE $CMD_BASE bin" \
  -c "load_image $RES_TAPE $RES_BASE bin" \
  -c "resume 0x00000000" \
  -c "shutdown" 2>&1 | sed 's/^/  /'
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "openocd failed (rc=$rc)"
    wait "$CAP" 2>/dev/null
    exit 1
fi

wait "$CAP" 2>/dev/null
echo

# --- 4. what came back ------------------------------------------------
if [ ! -s "$LOG" ]; then
    echo "the console produced nothing. The board did not run, or $PORT is"
    echo "not its console. Check with: make console"
    exit 1
fi

sed 's/\r$//' "$LOG" \
  | sed -n '/^--- BEGIN RESULTS ---$/,/^--- END RESULTS ---$/p' \
  | grep -vE '^(--- (BEGIN|END) RESULTS ---|#)' > "$GOT"

sed 's/\r$//' "$LOG" | grep -E '^(ID|cmd tape|records|values|VECTOR|#)' \
  | sed 's/^/  /'
echo

if [ ! -s "$GOT" ]; then
    echo "REPLAY FAIL -- no result block in the console output."
    echo "full capture: $LOG"
    exit 1
fi

if diff -u "$RES_TAPE" "$GOT" > "$BUILD/replay.diff"; then
    echo "REPLAY PASS -- $(wc -l < "$GOT") lines identical to $(basename "$RES_TAPE")"
    echo "the silicon and the simulation agree bit-for-bit."
    exit 0
fi

echo "REPLAY FAIL -- silicon differs from simulation:"
head -40 "$BUILD/replay.diff"
echo
echo "full diff: $BUILD/replay.diff     console: $LOG"
exit 1
