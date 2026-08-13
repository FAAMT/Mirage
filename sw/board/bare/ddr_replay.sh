#!/usr/bin/env bash
# =====================================================================
# ddr_replay.sh -- replay a SPLIT tape (WBLK + weight blob) from DDR
#
#   ./ddr_replay.sh <cmd_tape> <res_tape>
#
# A translation tape is 64.3 MB and OCM allows about 64 KB per tape, so
# tools/tape_split.py moves the weight rows into a blob: 0.91 MB of
# commands plus 25 MB of weights. Both still exceed OCM, so both go to
# DDR -- and DDR does not answer until ps7_init() has run.
#
# HOW DDR IS BROUGHT UP, AND WHY NOT ps7_init.tcl
#
# The XSA ships ps7_init.tcl, but it calls mask_write/mwr/mrd, which xsdb
# supplies and openocd does not. Shimming those is a new, untested path
# to a failure mode (a half-initialised DDR controller) that presents as
# silent corruption much later.
#
# The C ps7_init() in the same XSA is already proven on this board -- it
# is what the smoke test runs through. So: load the ELF, break at main(),
# resume. crt0.S runs ps7_init() and ps7_post_config(), DDR trains, and
# the breakpoint stops execution before main() reads anything. DDR is now
# live, the tapes are written into it, the breakpoint is removed, and
# execution resumes. No new initialisation code, no new failure mode.
#
# ORDERING
#
# The capture starts before openocd, not after: results are already on
# their way out of the UART by the time openocd returns. Nothing prints
# before the final resume, so capturing early costs nothing.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

# --- capture mode: re-entry under sg, so the quoting stays readable ---
# Before argument parsing, because in this mode $1..$4 are the capture's own
# arguments and not a tape pair.
if [ "${1:-}" = "--capture" ]; then
    stty -F "$2" 115200 raw -echo -crtscts
    exec timeout "$4" sed -u -e '/^VECTOR REPLAY /q' \
                             -e '/^stopping:/q' \
                             -e '/^#SCRIPT/q' < "$2" > "$3"
fi

CMD_TAPE=${1:?usage: ddr_replay.sh <cmd_tape> <res_tape>}
RES_TAPE=${2:?usage: ddr_replay.sh <cmd_tape> <res_tape>}
PORT=${PORT:-/dev/ttyUSB1}
CAP_SECS=${CAP_SECS:-900}
BUILD=build
BIT=${BIT:-../../../build_cora/mirage_cora_wrapper.bit}

# DDR map. Bases are passed to the compiler too, so vectors.c and this
# script cannot disagree about where anything lives.
WGT_BASE=${WGT_BASE:-0x01000000}     # 16 MB  blob (25 MB)
WGT_END=${WGT_END:-0x03000000}       # 48 MB
DDR_CMD=${DDR_CMD:-0x00400000}       #  4 MB  commands (0.91 MB)
DDR_CMD_END=${DDR_CMD_END:-0x00600000}
DDR_RES=${DDR_RES:-0x00600000}       #  6 MB  expected results
DDR_RES_END=${DDR_RES_END:-0x00800000}

WBLK=$BUILD/ddr_cmd.txt
BLOB=$BUILD/ddr_wgt.bin
LOG=$BUILD/ddr_console.log

echo "== splitting $(basename "$CMD_TAPE")"
../../../tools/tape_split.py split "$CMD_TAPE" "$WBLK" "$BLOB" || exit 1
ls -l "$WBLK" "$BLOB" | awk '{printf "   %-28s %8.2f MB\n", $NF, $5/1e6}'

echo "== building vectors for the DDR map"
make APP=vectors REGION=ocm \
     CMD_BASE=$DDR_CMD CMD_END=$DDR_CMD_END \
     RES_BASE=$DDR_RES RES_END=$DDR_RES_END \
     WGT_BASE=$WGT_BASE WGT_END=$WGT_END >/dev/null 2>&1 || {
    echo "   build failed"; exit 1; }
ELF=$BUILD/mirage_vectors.elf

# main() is where DDR is guaranteed up and nothing has been read yet.
MAIN=$(arm-none-eabi-nm "$ELF" | awk '$3=="main"{print "0x"$1}')
[ -n "$MAIN" ] || { echo "   cannot find main() in $ELF"; exit 1; }
echo "   breakpoint at main() = $MAIN"

echo "== capturing $PORT"
sg dialout -c "$0 --capture $PORT $LOG $CAP_SECS" &
CAPTURE=$!
sleep 1

echo "== openocd: bitstream, ps7_init via main() breakpoint, DDR load, resume"
openocd -f cora.cfg \
  -c "init" \
  -c "targets zynq.cpu0" \
  -c "halt" \
  -c "pld load 0 $BIT" \
  -c "load_image $ELF" \
  -c "bp $MAIN 4 hw" \
  -c "resume 0x00000000" \
  -c "wait_halt 20000" \
  -c "rbp $MAIN" \
  -c "load_image $BLOB $WGT_BASE bin" \
  -c "load_image $WBLK $DDR_CMD bin" \
  -c "load_image $RES_TAPE $DDR_RES bin" \
  -c "resume" \
  -c "shutdown" 2>&1 | sed -n 's/^/   /p' | grep -Ev '^\s*$'

echo "== waiting for the board (cap ${CAP_SECS}s)"
wait $CAPTURE 2>/dev/null
echo
cat "$LOG"
grep -q '^VECTOR REPLAY PASS' "$LOG" && exit 0 || exit 1
