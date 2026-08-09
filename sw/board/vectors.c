/* =====================================================================
 * vectors.c -- replay the verified op-level command script on silicon
 *
 * WHAT A PASS MEANS
 *
 * sim/ops_cmd.txt is the register sequence for every job type the design
 * implements. model_ref/test_ops.py generated it against the bit-exact
 * int8 reference model, and GHDL has already run it against the RTL,
 * producing sim/ops_res.txt. This program replays that same sequence
 * against the real fabric over AXI4-Lite and compares what comes back,
 * value by value, against that same expected file.
 *
 * So a pass means the synthesised, placed and routed design on this chip
 * computes bit-identical results to the simulation, which is in turn
 * bit-identical to the reference model. Not "close" -- identical, which
 * is the only useful standard for an integer datapath.
 *
 * THIS IS THE INVERSE OF io_trace.c
 *
 * That file turns register writes into this script format; this one turns
 * the script format back into register writes. Neither includes the
 * other, and `make hostcheck` runs both back to back on the host and
 * requires the result to be byte-identical to the input script. An
 * interpreter bug is therefore caught without a board, before a mismatch
 * here can be blamed on the silicon. That mattered more than it sounds:
 * the whole point of this program is to be trusted when it disagrees
 * with the hardware.
 *
 * WHAT IT DOES NOT COVER
 *
 * The driver's parameter-packing layer. This program takes CFG values
 * straight from the script, so mirage_gemm()/mirage_stage()/mirage_smax()
 * never run -- it uses mirage.c only for the activation port, the window
 * load and the job completion protocol. Checking that packing is
 * sw/test/check_ops.sh's job, against this same script. The two together
 * cover driver and fabric; neither covers both on its own.
 *
 * WHERE THE SCRIPTS LIVE
 *
 * Not compiled in. Both tapes are plain text loaded over JTAG, so
 * replaying a different script needs no rebuild:
 *
 *   load_image sim/ops_cmd.txt 0x00010000 bin
 *   load_image sim/ops_res.txt 0x00020000 bin
 *
 * IN OCM, NOT DDR, and that is forced rather than chosen. DDR is not
 * addressable until the DDR controller is configured, and the thing that
 * configures it is ps7_init() -- which runs when this program starts,
 * i.e. AFTER openocd would have had to write the tape. A tape loaded to
 * DDR at load time lands nowhere. OCM is alive out of reset, which is the
 * same reason the program itself links there.
 *
 * The cost is a ~64 KB ceiling per tape. ops_cmd.txt is 32 KB so this is
 * comfortable, but sim/xlate_cmd.txt is 64 MB and will need DDR, and
 * therefore a handshake: resume, let ps7_init() bring DDR up, halt again,
 * write the tape, resume. The XSA also ships ps7_init.tcl, which is the
 * same initialisation as a debugger script and would let openocd bring
 * DDR up itself before any load. Neither is needed at this size.
 *
 * Neither tape's length is passed in, because neither is needed: the
 * command script ends with END and the result file ends with DONE. Both
 * are self-terminating, so a tape that failed to load hits an
 * unrecognised record and says so on the first one it reads.
 * ===================================================================== */
#include <stdint.h>
#include <stddef.h>

#include "mirage.h"
#include "mirage_io.h"
#include "mirage_regs.h"

#ifdef MIRAGE_HOST_BUILD
#include <stdio.h>
#include <stdlib.h>
#include "mirage_trace.h"
#define PRINT printf
#define NL "\n"
#else
#include "xil_printf.h"
#define PRINT xil_printf
#define NL "\r\n"        /* a serial terminal wants both */
#endif

/* Where the tapes are loaded. Each tape's limit is the start of the next
 * region, so the parser is bounded by real memory rather than by a
 * round number: OCM low ends at 0x30000, and past it lies nothing this
 * board has mapped. That matters because crt0.S implements only the reset
 * vector -- a data abort from walking off the end would hang with no
 * output at all, which is the least diagnosable failure available. A
 * tape that overruns its region is reported instead. */
#ifndef MIRAGE_CMD_BASE
#define MIRAGE_CMD_BASE   0x00010000u      /*  64 KB in */
#endif
#ifndef MIRAGE_RES_BASE
#define MIRAGE_RES_BASE   0x00020000u      /* 128 KB in */
#endif
#ifndef MIRAGE_OCM_END
#define MIRAGE_OCM_END    0x00030000u      /* 192 KB, top of OCM low */
#endif

/* Readback chunk. Chunking is invisible to a trace comparison: io_trace.c
 * does not close an open DUMP on an ACT_PTR write, so contiguous chunks
 * still decode as the one DUMP line the script contains. */
#define CHUNK_BYTES   512u
#define CHUNK_WORDS   128u

/* Enough for the longest contiguous ACT run in any script so far (384);
 * longer runs simply flush more than once, which is harmless. */
#define ACT_RUN_MAX   512u

/* How many differing values to print before going quiet. A datapath that
 * is wrong is usually wrong everywhere, and 1088 mismatch lines would
 * bury the summary that says so. */
#define MAX_REPORTED  16

/* ===================================================================== */
/* a tape is a cursor and a hard limit                                   */
/* ===================================================================== */
typedef struct {
    const char *p;
    const char *end;
    const char *base;
    const char *name;
} tape_t;

static tape_t cmd;
static tape_t res;

static int at_end(const tape_t *t)
{
    return t->p >= t->end;
}

/* Whitespace-delimited tokens, newlines included. Both file formats are
 * regular enough that line structure carries no information: a record's
 * type determines exactly how many tokens follow it, so there is nothing
 * for a line-aware parser to do except break on \r\n. */
static int next_token(tape_t *t, char *buf, size_t n)
{
    size_t i = 0u;

    while (!at_end(t) && (*t->p == ' '  || *t->p == '\t' ||
                          *t->p == '\n' || *t->p == '\r'))
        t->p++;

    while (!at_end(t) && *t->p != ' '  && *t->p != '\t' &&
                         *t->p != '\n' && *t->p != '\r') {
        if (i + 1u < n)
            buf[i++] = *t->p;
        t->p++;                     /* advance even past a full buffer */
    }
    buf[i] = '\0';
    return i > 0u;
}

static int streq(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;
}

static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int parse_hex(const char *s, uint32_t *out)
{
    uint32_t v = 0u;
    int any = 0, d;

    for (; *s; s++) {
        d = hexval(*s);
        if (d < 0)
            return 0;
        v = (v << 4) | (uint32_t)d;
        any = 1;
    }
    *out = v;
    return any;
}

static int parse_dec(const char *s, int32_t *out)
{
    int32_t v = 0;
    int neg = 0, any = 0;

    if (*s == '-') { neg = 1; s++; }
    for (; *s; s++) {
        if (*s < '0' || *s > '9')
            return 0;
        v = v * 10 + (*s - '0');
        any = 1;
    }
    *out = neg ? -v : v;
    return any;
}

/* ===================================================================== */
/* deferred writes                                                       */
/* ===================================================================== */

/* ACT bytes are batched into runs of consecutive addresses so the
 * activation pointer is written once per run rather than once per byte --
 * which is also what mirage_write_act() does, and therefore what the
 * verified sequence contains. */
static int8_t   act_run[ACT_RUN_MAX];
static uint32_t act_run_n;
static uint32_t act_run_addr;
static uint32_t act_run_next;

static void act_flush(void)
{
    if (act_run_n == 0u)
        return;
    mirage_write_act(act_run_addr, act_run, act_run_n);
    act_run_n = 0u;
}

/* One weight word is always held back, because TLAST marks the end of the
 * WINDOW rather than the end of a row (see mirage_load_window), and which
 * word is last is only known once the run of WROW records ends. Holding
 * one word is what keeps this a single forward pass with no lookahead --
 * and sending TLAST per row instead would silently stack the entire
 * window into row 0. */
static uint32_t wgt_pend;
static int      wgt_pend_valid;

static void wgt_push(uint32_t word)
{
    if (wgt_pend_valid)
        mirage_wr32(MIRAGE_REG_WGT_DATA, wgt_pend);
    wgt_pend = word;
    wgt_pend_valid = 1;
}

static void wgt_flush(void)
{
    if (!wgt_pend_valid)
        return;
    mirage_wr32(MIRAGE_REG_WGT_LAST, wgt_pend);
    wgt_pend_valid = 0;
}

/* ===================================================================== */
/* comparison against the expected tape                                  */
/* ===================================================================== */
static uint32_t n_checked, n_bad, n_missing, n_reported;
static int      quiet;                     /* suppress the value stream */

static int res_value(int32_t *out)
{
    char tok[24];

    if (!next_token(&res, tok, sizeof tok))
        return 0;
    return parse_dec(tok, out);
}

static void emit(int32_t got)
{
    int32_t want;

    if (!quiet)
        PRINT("%d" NL, (int)got);

    if (!res_value(&want)) {
        n_missing++;
        return;
    }
    n_checked++;
    if (want == got) {
        return;
    }
    n_bad++;
    if (n_reported < MAX_REPORTED) {
        PRINT("#MISMATCH at value %u: want %d, got %d" NL,
              (unsigned)n_checked, (int)want, (int)got);
        n_reported++;
    } else if (n_reported == MAX_REPORTED) {
        PRINT("#MISMATCH reporting capped; see the totals below" NL);
        n_reported++;
    }
}

/* ===================================================================== */
static uint32_t n_act, n_wrow, n_cfg, n_job, n_dump, n_dumps, n_mark;

static int8_t   i8buf[CHUNK_BYTES];
static int32_t  scbuf[CHUNK_WORDS];

static void do_dump(uint32_t addr, uint32_t count, int scores)
{
    uint32_t i, n;

    while (count > 0u) {
        if (scores) {
            n = count < CHUNK_WORDS ? count : CHUNK_WORDS;
            mirage_read_scores(addr, scbuf, n);
            for (i = 0u; i < n; i++)
                emit(scbuf[i]);
        } else {
            n = count < CHUNK_BYTES ? count : CHUNK_BYTES;
            mirage_read_act(addr, i8buf, n);
            for (i = 0u; i < n; i++)
                emit((int32_t)i8buf[i]);
        }
        addr  += n;
        count -= n;
    }
    if (scores)
        n_dumps++;
    else
        n_dump++;
}

/* Check the label the result tape carries against the one the script
 * asked for. This is the desynchronisation guard: without it a dropped
 * or extra value would shift every later comparison and report a
 * datapath fault instead of a bookkeeping one. */
static int res_mark(const char *label)
{
    char tok[64];

    if (!next_token(&res, tok, sizeof tok) || !streq(tok, "MARK")) {
        PRINT("#EXPECTED TAPE: wanted MARK, found \"%s\"" NL, tok);
        return 0;
    }
    if (!next_token(&res, tok, sizeof tok) || !streq(tok, label)) {
        PRINT("#EXPECTED TAPE: block is \"%s\", script says \"%s\"" NL,
              tok, label);
        return 0;
    }
    return 1;
}

/* ===================================================================== */
static int replay(void)
{
    char tok[80], label[64];
    uint32_t v, w, idx, addr, count, expect_row = 0u;
    int32_t njobs;
    int rc, desync = 0;

    for (;;) {
        if (!next_token(&cmd, tok, sizeof tok)) {
            PRINT("#SCRIPT ended without END" NL);
            return -1;
        }

        /* ---- ACT addr byte ------------------------------------------ */
        if (streq(tok, "ACT")) {
            wgt_flush();
            if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &addr))
                goto bad_record;
            if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &v))
                goto bad_record;

            if (act_run_n > 0u &&
                (addr != act_run_next || act_run_n >= ACT_RUN_MAX))
                act_flush();
            if (act_run_n == 0u)
                act_run_addr = addr;
            act_run[act_run_n++] = (int8_t)(uint8_t)(v & 0xFFu);
            act_run_next = addr + 1u;
            n_act++;
            continue;
        }

        /* ---- WROW idx w0..w7 ---------------------------------------- */
        if (streq(tok, "WROW")) {
            act_flush();
            if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &idx))
                goto bad_record;
            if (!wgt_pend_valid)
                expect_row = 0u;               /* a new window starts */
            if (idx != expect_row) {
                PRINT("#SCRIPT: WROW index %u where %u was due"
                      "  (corrupt tape?)" NL,
                      (unsigned)idx, (unsigned)expect_row);
                return -1;
            }
            expect_row++;
            for (w = 0u; w < (uint32_t)MIRAGE_W_BANKS; w++) {
                if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &v))
                    goto bad_record;
                wgt_push(v);
            }
            n_wrow++;
            continue;
        }

        /* ---- CFG n value ------------------------------------------- */
        if (streq(tok, "CFG")) {
            act_flush(); wgt_flush();
            if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &idx))
                goto bad_record;
            if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &v))
                goto bad_record;
            if (idx >= MIRAGE_CFG_COUNT) {
                PRINT("#SCRIPT: CFG%u is outside the register file" NL,
                      (unsigned)idx);
                return -1;
            }
            mirage_wr32(MIRAGE_CFG(idx), v);
            n_cfg++;
            continue;
        }

        /* ---- JOB op ------------------------------------------------- */
        if (streq(tok, "JOB")) {
            act_flush(); wgt_flush();
            if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &v))
                goto bad_record;
            rc = mirage_run_job(v);
            if (rc != MIRAGE_OK) {
                PRINT("#JOB %u (op %u) failed rc=%d  STATUS=0x%08x" NL,
                      (unsigned)n_job, (unsigned)v, rc,
                      (unsigned)mirage_rd32(MIRAGE_REG_STATUS));
                PRINT("#stopping: every value after a failed job is noise." NL);
                return -1;
            }
            n_job++;
            continue;
        }

        /* ---- MARK label -------------------------------------------- */
        if (streq(tok, "MARK")) {
            act_flush(); wgt_flush();
            if (!next_token(&cmd, label, sizeof label))
                goto bad_record;
            mirage_mark(label);
            if (!quiet)
                PRINT("MARK %s" NL, label);
            if (!res_mark(label))
                desync = 1;
            n_mark++;
            continue;
        }

        /* ---- DUMP / DUMPS addr count -------------------------------- */
        if (streq(tok, "DUMP") || streq(tok, "DUMPS")) {
            /* The address is hex and the count is decimal, because that
             * is what io_trace.c writes ("DUMP %04x %u"). */
            int scores = streq(tok, "DUMPS");
            int32_t cnt;
            act_flush(); wgt_flush();
            if (!next_token(&cmd, tok, sizeof tok) || !parse_hex(tok, &addr))
                goto bad_record;
            if (!next_token(&cmd, tok, sizeof tok) || !parse_dec(tok, &cnt) ||
                cnt < 0)
                goto bad_record;
            count = (uint32_t)cnt;
            do_dump(addr, count, scores);
            continue;
        }

        /* ---- END ---------------------------------------------------- */
        if (streq(tok, "END")) {
            act_flush(); wgt_flush();
            break;
        }

        if (tok[0] == '#')                     /* comment record */
            continue;

    bad_record:
        PRINT("#SCRIPT: unrecognised record \"%s\" at offset %u" NL,
              tok, (unsigned)(size_t)(cmd.p - cmd.base));
        PRINT("#the command tape is not what this expects -- did it load?" NL);
        return -1;
    }

    /* The result tape closes with the number of jobs the reference ran.
     * Comparing it catches a replay that silently stopped early. */
    if (!next_token(&res, tok, sizeof tok) || !streq(tok, "DONE")) {
        PRINT("#EXPECTED TAPE: wanted DONE, found \"%s\"" NL, tok);
        desync = 1;
    } else if (!next_token(&res, tok, sizeof tok) || !parse_dec(tok, &njobs)) {
        /* decimal, not hex: the result tape counts jobs, and 16 read as
         * hex is 22 -- which is exactly the false failure this caught. */
        desync = 1;
    } else if ((uint32_t)njobs != n_job) {
        PRINT("#JOB COUNT: reference ran %d, this ran %u" NL,
              (int)njobs, (unsigned)n_job);
        desync = 1;
    }
    if (!quiet)
        PRINT("DONE %u" NL, (unsigned)n_job);

    return desync ? -1 : 0;
}

/* ===================================================================== */
static int run(void)
{
    int rc, ok;

    PRINT(NL "=== Mirage vector replay ===" NL);

    rc = mirage_probe();
    PRINT("ID       : 0x%08x  %s" NL,
          (unsigned)mirage_rd32(MIRAGE_REG_ID),
          rc == MIRAGE_OK ? "OK" : "MISMATCH (expected 0x4D495247)");
    if (rc != MIRAGE_OK) {
        PRINT("stopping: nothing below this can mean anything." NL);
        return 1;
    }
    mirage_reset_status();

#ifdef MIRAGE_HOST_BUILD
    PRINT("cmd tape : %s" NL, cmd.name);
    PRINT("res tape : %s" NL NL, res.name);
#else
    PRINT("cmd tape : 0x%08x" NL, (unsigned)MIRAGE_CMD_BASE);
    PRINT("res tape : 0x%08x" NL NL, (unsigned)MIRAGE_RES_BASE);
#endif
    PRINT("--- BEGIN RESULTS ---" NL);

    ok = (replay() == 0);

    PRINT("--- END RESULTS ---" NL NL);
    PRINT("records  : ACT %u  WROW %u  CFG %u  JOB %u"
          "  MARK %u  DUMP %u  DUMPS %u" NL,
          (unsigned)n_act, (unsigned)n_wrow, (unsigned)n_cfg,
          (unsigned)n_job, (unsigned)n_mark, (unsigned)n_dump,
          (unsigned)n_dumps);
    PRINT("values   : %u checked, %u mismatched" NL,
          (unsigned)n_checked, (unsigned)n_bad);
    if (n_missing)
        PRINT("values   : %u had no expected value on the result tape" NL,
              (unsigned)n_missing);
    PRINT(NL);

    if (ok && n_bad == 0u && n_missing == 0u && n_checked > 0u) {
        PRINT("VECTOR REPLAY PASS -- silicon matches the reference"
              " bit-exactly." NL);
        return 0;
    }
    PRINT("VECTOR REPLAY FAIL" NL);
    return 1;
}

/* ===================================================================== */
#ifdef MIRAGE_HOST_BUILD

/* Host build: tapes come from files, and register writes go to the trace
 * backend instead of a bus. Diffing that trace against the input script
 * is what validates the interpreter itself -- see `make hostcheck`.
 *
 * Value comparison is meaningless here (io_trace.c reads back zero, by
 * construction: there is no fabric), so the exit status is deliberately
 * ignored by hostcheck, which diffs the trace and nothing else. */
static char *slurp(const char *path, tape_t *t)
{
    long n;
    char *buf;
    FILE *f = fopen(path, "rb");

    if (!f) {
        fprintf(stderr, "vectors: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc((size_t)n + 1u);
    if (!buf || fread(buf, 1u, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "vectors: cannot read %s\n", path);
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    t->base = t->p = buf;
    t->end  = buf + n;
    t->name = path;
    return buf;
}

int main(int argc, char **argv)
{
    const char *cmd_path = argc > 1 ? argv[1] : "../../sim/ops_cmd.txt";
    const char *res_path = argc > 2 ? argv[2] : "../../sim/ops_res.txt";
    const char *trc_path = argc > 3 ? argv[3] : "vectors_trace.txt";

    quiet = 1;                 /* the trace is the output that matters */

    if (!slurp(cmd_path, &cmd) || !slurp(res_path, &res))
        return 2;
    if (mirage_trace_open(trc_path) != 0) {
        fprintf(stderr, "vectors: cannot write %s\n", trc_path);
        return 2;
    }
    run();
    mirage_trace_close();
    return 0;
}

#else

/* zynq.ld puts the stack immediately above .bss and grows it down from
 * here, so this symbol is the highest address the program itself uses. */
extern char __stack_top[];

int main(void)
{
    cmd.base = cmd.p = (const char *)(uintptr_t)MIRAGE_CMD_BASE;
    cmd.end  = (const char *)(uintptr_t)MIRAGE_RES_BASE;

    res.base = res.p = (const char *)(uintptr_t)MIRAGE_RES_BASE;
    res.end  = (const char *)(uintptr_t)MIRAGE_OCM_END;

    /* The tapes and this program share OCM, and openocd will happily
     * write a tape straight over the stack. That failure looks like
     * arbitrary corruption a long way from its cause, so check it here
     * where it is one comparison and an obvious message. */
    if (cmd.base < __stack_top) {
        PRINT(NL "the command tape at 0x%08x is inside this program"
              " (which ends at 0x%08x)." NL,
              (unsigned)(uintptr_t)cmd.base,
              (unsigned)(uintptr_t)__stack_top);
        PRINT("raise MIRAGE_CMD_BASE, or shrink STACK_SIZE." NL);
        return 1;
    }

    return run();
}

#endif
