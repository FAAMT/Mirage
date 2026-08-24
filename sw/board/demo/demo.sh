#!/usr/bin/env bash
# =====================================================================
# demo.sh -- English in, board runs the arithmetic, Arabic out.
#
#   ./demo.sh                 run the tape on the board, then decode
#   ./demo.sh --negative      corrupt one byte first; the check must FAIL
#   ./demo.sh --offline       decode the last board log, touch no hardware
#   ./demo.sh --check         preflight only: is the board reachable
#
# WHY THE PREFLIGHT IS NOT OPTIONAL
#
# Every failure this demo has had in practice was one of two things, and
# neither announces itself: something else was holding the serial port, so
# the capture recorded nothing and an empty log looks exactly like a dead
# board; or the board was never plugged in. Both are one grep to rule out
# and twenty minutes to debug from the symptom, so they are ruled out here
# before anything is loaded.
#
# WHY --negative EXISTS
#
# A demo that can only pass proves nothing. Corrupting a single activation
# byte in a 64 MB tape and watching the comparison catch it is the evidence
# that the passing run means something. Run it on camera immediately after
# the passing run; it is the most persuasive thirty seconds available.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

# Absolute, because ddr_replay.sh is run from its own directory and a
# relative tape path would be resolved against the wrong one.
HERE=$(cd "$(dirname "$0")" && pwd)
BARE=$(cd "$HERE/../bare" && pwd)
SIM=$(cd "$HERE/../../../sim" && pwd)
PY=$HERE/../../../model_ref/.venv/bin/python
BUILD=$HERE/build

# The console is NOT reliably /dev/ttyUSB1. The FT2232H puts JTAG on
# interface 0 and the UART on interface 1; ftdi_sio binds whichever it is
# allowed to, and openocd unbinds interface 0 to claim JTAG over libusb. So
# the UART is ttyUSB1 when both interfaces have nodes and ttyUSB0 when only
# one does -- it moves depending on what ran last. `-if01-` names the UART
# interface directly and does not renumber, so prefer it and keep the old
# guess only as a fallback for a box without /dev/serial.
if [ -z "${PORT:-}" ]; then
    PORT=$(ls /dev/serial/by-id/*-if01-port0 2>/dev/null | head -1)
    [ -n "$PORT" ] || PORT=/dev/ttyUSB1
fi
export PORT
TEXT=${TEXT:-Hello}

CMD_TAPE=${CMD_TAPE:-$SIM/xlate_cmd.txt}
RES_TAPE=${RES_TAPE:-$SIM/xlate_res.txt}

MODE=run
case "${1:-}" in
    --negative) MODE=negative ;;
    --offline)  MODE=offline ;;
    --check)    MODE=check ;;
    --reset)    MODE=reset ;;
    "")         ;;
    *) echo "usage: $0 [--negative|--offline|--check|--reset]"; exit 2 ;;
esac

# --- unwedge the bridge ------------------------------------------------
# openocd resetting the FT2232H while a reader has the UART open leaves the
# port answering EBUSY with NO process holding it -- fuser, lsof and a scan
# of every /proc/*/fd all come back empty, so it looks like a phantom. The
# cure is a USB-level reset, which is a plain ioctl on the usbfs node and
# needs no root as long as the device is group-writable. This is the
# software equivalent of unplugging the cable.
if [ "$MODE" = reset ]; then
    read -r bus dev <<<"$(lsusb -d 0403:6010 \
        | sed -n 's/Bus \([0-9]*\) Device \([0-9]*\).*/\1 \2/p' | head -1)"
    [ -n "${bus:-}" ] || { echo "no FT2232H on USB -- plug the board in"; exit 1; }
    python3 - "/dev/bus/usb/$bus/$dev" <<'PY'
import fcntl, os, sys
USBDEVFS_RESET = ord('U') << 8 | 20
fd = os.open(sys.argv[1], os.O_WRONLY)
fcntl.ioctl(fd, USBDEVFS_RESET, 0)
os.close(fd)
print(f"   reset {sys.argv[1]}")
PY
    [ $? -eq 0 ] || { echo "   reset failed -- unplug and replug the cable"; exit 1; }
    sleep 3
    echo "   console is now: $(ls /dev/serial/by-id/*-if01-port0 2>/dev/null | head -1)"
    exit 0
fi

mkdir -p "$BUILD"

bar() { printf '\033[1m%s\033[0m\n' "$*"; }

# --- preflight --------------------------------------------------------
preflight() {
    local bad=0

    if lsusb -d 0403:6010 >/dev/null 2>&1; then
        echo "   board          FT2232H present on USB"
    else
        echo "   board          NOT FOUND -- plug the Cora into USB (the"
        echo "                  micro-USB marked PROG/UART, not the host port)"
        bad=1
    fi

    if [ -c "$PORT" ]; then
        echo "   console        $PORT"
    else
        echo "   console        $PORT missing"
        bad=1
    fi

    # Print what the stable name resolves to today, because the number will
    # not be the same tomorrow and every error message quotes the number.
    echo "   resolves to    $(readlink -f "$PORT")"

    # Scan every process, not a list of likely names. The reader that caused
    # the most confusion was a leftover `sed` from a previous run's capture,
    # which no name-based check would have looked for.
    local holder
    holder=$(for p in /proc/[0-9]*; do
                 for f in "$p"/fd/*; do
                     if [ "$(readlink "$f" 2>/dev/null)" = "$PORT" ]; then
                         echo "$(basename "$p") $(tr -d '\0' < "$p/comm")"
                         break
                     fi
                 done
             done 2>/dev/null | sort -u)
    if [ -n "$holder" ]; then
        echo "   port holder    $PORT is held -- the capture would record"
        echo "                  nothing. Close it, then re-run:"
        echo "$holder" | sed 's/^/                      kill /'
        bad=1
    elif sg dialout -c "stty -F $PORT 115200 raw -echo -crtscts" 2>/dev/null; then
        echo "   port opens     yes, at 115200 8N1"
    else
        # Busy with no holder means the bridge is wedged, not contended.
        echo "   port opens     NO -- and nothing holds it, so the FT2232H is"
        echo "                  wedged rather than busy. Reset it with:"
        echo "                      $0 --reset"
        bad=1
    fi

    for f in "$CMD_TAPE" "$RES_TAPE" "$BARE/cora.cfg"; do
        [ -f "$f" ] || { echo "   missing        $f"; bad=1; }
    done
    [ -x "$PY" ] || { echo "   missing        $PY"; bad=1; }

    if command -v openocd >/dev/null; then
        echo "   openocd        $(command -v openocd)"
    else
        echo "   openocd        NOT INSTALLED"; bad=1
    fi

    return $bad
}

bar "== 1. preflight"
preflight || { echo; echo "preflight failed -- fix the above and re-run."; exit 1; }
[ "$MODE" = check ] && { echo; echo "board is ready."; exit 0; }

# --- pick the tape ----------------------------------------------------
RUN_CMD=$CMD_TAPE
if [ "$MODE" = negative ]; then
    echo
    bar "== 2. NEGATIVE CONTROL: corrupting one byte of the command tape"
    RUN_CMD=$BUILD/xlate_cmd.corrupt.txt
    # The first record of the tape is the first activation byte of the
    # first token. 05 -> 06 is the smallest change the tape can express:
    # one int8 activation, one LSB, in 64.3 MB.
    head -1 "$CMD_TAPE" | sed 's/^/   was:  /'
    sed '1s/^ACT 0000 05$/ACT 0000 06/' "$CMD_TAPE" > "$RUN_CMD" || exit 1
    head -1 "$RUN_CMD" | sed 's/^/   now:  /'
    if cmp -s <(head -1 "$CMD_TAPE") <(head -1 "$RUN_CMD"); then
        echo "   the first record was not 'ACT 0000 05' -- nothing corrupted."
        echo "   this tape is not the one this mode was written for."
        exit 1
    fi
    echo "   one byte of $(du -h "$CMD_TAPE" | cut -f1) changed."
fi

# --- run on the board -------------------------------------------------
LOG=$BUILD/ddr_console.log
if [ "$MODE" = offline ]; then
    echo
    bar "== 2. OFFLINE: reusing the last board log, hardware untouched"
    [ -s "$LOG" ] || LOG=$BARE/build/ddr_console.pass-2026-08-18.log
    [ -s "$LOG" ] || { echo "   no saved board log to decode"; exit 1; }
    echo "   $LOG"
else
    echo
    bar "== 3. running on the board  (~4 min: 25 MB over JTAG at ~134 KiB/s)"
    # Delete both logs before the run, not just check them after. A console
    # log that survives a failed run is indistinguishable from one the run
    # produced, and reading a previous PASS out of a stale file is the worst
    # failure this script could have -- it reports success for a board that
    # was never asked anything. Gone before, mandatory after.
    rm -f "$LOG" "$BARE/build/ddr_console.log"
    # 7 minutes, not the driver's 15. The capture holds the port until it
    # sees a verdict or times out, and a 15-minute squatter blocks the next
    # attempt long enough that the next run fails for a reason that has
    # nothing to do with the board.
    ( cd "$BARE" && CAP_SECS=${CAP_SECS:-420} ./ddr_replay.sh "$RUN_CMD" "$RES_TAPE" )
    rc=$?
    if [ ! -s "$BARE/build/ddr_console.log" ]; then
        echo
        echo "   the board produced no console output. Either it never ran,"
        echo "   or something took $PORT after the preflight. Re-run"
        echo "   ./demo.sh --check to see who holds the port."
        exit 1
    fi
    cp "$BARE/build/ddr_console.log" "$LOG"
    echo "   board verdict: $(grep -a '^VECTOR REPLAY' "$LOG" | tr -d '\r')"
    echo "   (replay exit $rc)"
fi

# --- decode -----------------------------------------------------------
echo
bar "== 4. decoding the numbers the board produced"
"$PY" board_translate.py "$LOG" \
      --text "$TEXT" \
      --reference "$RES_TAPE" \
      --scratch "$BUILD/rebuild_cmd.txt"
rc=$?

if [ "$MODE" = negative ]; then
    echo
    if [ $rc -ne 0 ]; then
        bar "NEGATIVE CONTROL PASSED -- one flipped byte was caught."
        echo "The comparison is capable of failing, so the passing run means"
        echo "something. Restore it by running ./demo.sh with no flags."
        exit 0
    fi
    bar "NEGATIVE CONTROL FAILED -- corruption went undetected."
    echo "The check is not doing its job. Do not trust the passing run."
    exit 1
fi
exit $rc
