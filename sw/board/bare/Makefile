# =====================================================================
# sw/board/bare/Makefile -- build sw/board/*.c for the Cora Z7, no Vitis
#
#   make                 build build/mirage_smoke.elf
#   make load            program the bitstream and run it over JTAG
#   make console         open the UART at 115200 (needs the dialout group)
#   make APP=vectors     build the vector replay rather than the smoke test
#   make replay          build it, run it on the board, diff against sim/
#   make hostcheck       validate the replay interpreter with no board
#   make hostcheck-wblk  same, for a split (WBLK + weight blob) tape
#   make clean
#
# APP names a .c file in sw/board/. Nothing below is app-specific except
# which source that is.
#
# WHY THIS EXISTS
#
# The next step after a bitstream is a bare-metal application, and the
# documented way to get one is Vitis: ~60-100 GB, and 2022.2 does not
# support this machine's Ubuntu release. None of that is actually needed.
# The XSA already carries everything the vendor flow would have
# generated for us -- ps7_init_gpl.c is the PS configuration as C, and it
# depends on nothing but a volatile pointer. So the whole platform
# reduces to: a linker script, a crt0, and a UART driver.
#
# What this does NOT replace is Vivado. Changing the hardware still means
# a synthesis run. This builds software against a bitstream that already
# exists.
# =====================================================================

# --- what to build against --------------------------------------------
SW      := ../..
REPO    := $(SW)/..
XSA     ?= $(REPO)/build_cora/mirage_cora.xsa
BIT     ?= $(REPO)/build_cora/mirage_cora_wrapper.bit
BUILD   := build

APP     ?= smoke
TARGET  := $(BUILD)/mirage_$(APP).elf
MAP     := $(BUILD)/mirage_$(APP).map
BIN     := $(BUILD)/mirage_$(APP).bin

# --- the vector replay -------------------------------------------------
# Tape addresses must agree with sw/board/vectors.c. They are OCM, not
# DDR, because DDR does not answer until ps7_init() has run and that
# happens after openocd would have written the tape. See vectors.c.
CMD_TAPE  ?= $(REPO)/sim/ops_cmd.txt
RES_TAPE  ?= $(REPO)/sim/ops_res.txt
CMD_BASE  ?= 0x00010000
RES_BASE  ?= 0x00020000
# Ends default to the OCM map. A split tape (WBLK + blob) does not fit OCM, so
# ddr_replay.sh overrides all six to place tapes and blob in DDR.
CMD_END   ?= $(RES_BASE)
RES_END   ?= 0x00030000
WGT_BASE  ?= 0x01000000
WGT_END   ?= 0x03000000

# --- where it runs -----------------------------------------------------
# ocm: on-chip RAM, alive out of reset, no dependency on DDR training.
# ddr: for anything that outgrows 192 KB.
REGION     ?= ocm
STACK_SIZE ?= 0x4000

ifeq ($(REGION),ocm)
  LOAD_BASE := 0x00000000
else ifeq ($(REGION),ddr)
  LOAD_BASE := 0x00100000
else
  $(error REGION must be ocm or ddr, got '$(REGION)')
endif

# --- toolchain ---------------------------------------------------------
CROSS   ?= arm-none-eabi-
CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
SIZE    := $(CROSS)size

# soft-float, not softfp: mirage.c calls nearbyint() and ldexp(), and a
# soft-float ABI means those resolve in libm without crt0 having to
# enable the VFP unit first. The arithmetic is a handful of operations
# per job -- there is nothing here worth an FPU enable sequence that
# could itself be the bug.
ARCH    := -mcpu=cortex-a9 -marm -mfloat-abi=soft

CFLAGS  := $(ARCH) -std=c99 -O2 -g \
           -Wall -Wextra -Wno-unused-parameter \
           -ffreestanding -fno-builtin -fno-common \
           -I. -I$(SW)

LDFLAGS := $(ARCH) -nostartfiles -specs=nosys.specs \
           -Wl,-T,zynq.ld \
           -Wl,--defsym=LOAD_BASE=$(LOAD_BASE) \
           -Wl,--defsym=STACK_SIZE=$(STACK_SIZE) \
           -Wl,-Map,$(MAP) \
           -Wl,--gc-sections
LDLIBS  := -lm

# --- sources -----------------------------------------------------------
# The three driver files are used UNMODIFIED. smoke.c compiles here
# because bare/xil_printf.h supplies the name it already expects, not
# because smoke.c was edited to suit this build -- it is verified code
# and editing it would mean the thing on the board is no longer the
# thing that was checked.
OBJS := $(BUILD)/crt0.o \
        $(BUILD)/start.o \
        $(BUILD)/uart.o \
        $(BUILD)/ps7_init_gpl.o \
        $(BUILD)/mirage.o \
        $(BUILD)/io_mmio.o \
        $(BUILD)/$(APP).o

.PHONY: all load console clean size replay hostcheck hostcheck-wblk
all: $(TARGET) size

# --- the XSA is the platform ------------------------------------------
# ps7_init_gpl.c is generated, 12,500 lines, and belongs to the build
# rather than the source tree -- same reasoning that keeps build_cora/
# gitignored. Extract it, never commit it.
$(BUILD)/ps7_init_gpl.c $(BUILD)/ps7_init_gpl.h: $(XSA)
	@mkdir -p $(BUILD)
	@echo "  UNZIP   ps7_init_gpl.[ch] <- $(notdir $(XSA))"
	@unzip -o -q $(XSA) ps7_init_gpl.c ps7_init_gpl.h -d $(BUILD)

# Generated code, compiled with warnings off: it is not ours to fix, and
# a wall of warnings from it would bury a real one from our own files.
$(BUILD)/ps7_init_gpl.o: $(BUILD)/ps7_init_gpl.c $(BUILD)/ps7_init_gpl.h
	@echo "  CC      $(notdir $<)  (generated)"
	@$(CC) $(ARCH) -std=c99 -O2 -ffreestanding -fno-builtin -w \
	       -I$(BUILD) -I. -c -o $@ $<

$(BUILD)/%.o: %.c | $(BUILD)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: %.S | $(BUILD)
	@echo "  AS      $<"
	@$(CC) $(ARCH) -c -o $@ $<

# The application, from sw/board/. Tape addresses are passed as macros so
# the Makefile and vectors.c cannot drift apart: change them in one place.
$(BUILD)/%.o: ../%.c | $(BUILD)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -DMIRAGE_CMD_BASE=$(CMD_BASE)u \
	       -DMIRAGE_RES_BASE=$(RES_BASE)u \
	       -DMIRAGE_CMD_END=$(CMD_END)u -DMIRAGE_RES_END=$(RES_END)u \
	       -DMIRAGE_WGT_BASE=$(WGT_BASE)u -DMIRAGE_WGT_END=$(WGT_END)u \
	       -c -o $@ $<

$(BUILD)/mirage.o: $(SW)/mirage.c | $(BUILD)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/io_mmio.o: $(SW)/io_mmio.c | $(BUILD)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD):
	@mkdir -p $(BUILD)

$(TARGET): $(OBJS) zynq.ld
	@echo "  LD      $@   (REGION=$(REGION) @ $(LOAD_BASE))"
	@$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)
	@$(OBJCOPY) -O binary $@ $(BIN)

size: $(TARGET)
	@echo
	@$(SIZE) $(TARGET)
	@echo
	@echo "OCM low is 192 KB (0x30000). text+data+bss must fit under it."

# --- running it --------------------------------------------------------
# resume at LOAD_BASE rather than at the ELF entry: the first word there
# is the reset vector's `b _start`, so this works for both regions and
# does not depend on openocd agreeing with us about the entry symbol.
load: $(TARGET)
	openocd -f cora.cfg \
	  -c "init" \
	  -c "targets zynq.cpu0" \
	  -c "halt" \
	  -c "pld load 0 $(BIT)" \
	  -c "load_image $(TARGET)" \
	  -c "resume $(LOAD_BASE)" \
	  -c "shutdown"

# The UART is interface 1 of the FT2232H, and its ttyUSB number moves: it is
# ttyUSB1 while ftdi_sio holds both interfaces and ttyUSB0 after openocd has
# taken interface 0 for JTAG. `-if01-` names the interface instead of the
# enumeration order, so it stays correct either way.
#
# picocom, not screen. screen detaches itself when it cannot take the
# terminal -- under `make`, or over a non-interactive stdin -- which looks
# exactly like it exited, while it goes on holding the port and starving
# every later capture. picocom just fails.
CONSOLE ?= $(firstword $(wildcard /dev/serial/by-id/*-if01-port0) /dev/ttyUSB1)

console:
	@echo "115200 8N1 on $(CONSOLE) -- ctrl-a ctrl-x to quit picocom"
	@picocom -b 115200 $(CONSOLE)

# --- the vector replay -------------------------------------------------
# The console has to be captured BEFORE the CPU is resumed, because the
# results are already on their way out of the UART by the time openocd
# returns. That ordering is more than a recipe should carry, so it lives
# in run_replay.sh; this target is here so the entry point is where
# everything else is.
replay:
	@./run_replay.sh "$(CMD_TAPE)" "$(RES_TAPE)"

# The same interpreter with no board: link vectors.c against the trace
# backend instead of the MMIO one, replay the script into a trace, and
# require that trace to be byte-identical to the script it came from.
#
# This is what makes a mismatch on hardware attributable to the hardware.
# Without it, "the board disagrees with simulation" has two suspects and
# the cheap one is never ruled out. The exit status is ignored on purpose:
# io_trace.c reads back zero because there is no fabric behind it, so the
# value comparison must fail and only the trace is evidence.
HOSTCC ?= cc
$(BUILD)/vectors_host: ../vectors.c $(SW)/mirage.c $(SW)/io_trace.c | $(BUILD)
	@echo "  CC      ../vectors.c  (host build, trace backend)"
	@$(HOSTCC) -DMIRAGE_HOST_BUILD -std=c99 -O1 -Wall -Wextra \
	   -Wno-unused-parameter -I. -I$(SW) -o $@ \
	   ../vectors.c $(SW)/mirage.c $(SW)/io_trace.c -lm

hostcheck: $(BUILD)/vectors_host
	@$(BUILD)/vectors_host $(CMD_TAPE) $(RES_TAPE) $(BUILD)/roundtrip.txt \
	   > $(BUILD)/hostcheck.log 2>&1 || true
	@if diff -q $(CMD_TAPE) $(BUILD)/roundtrip.txt >/dev/null 2>&1; then \
	   echo "HOSTCHECK PASS -- replay reproduces $(notdir $(CMD_TAPE))"\
	        "exactly ($$(wc -l < $(CMD_TAPE)) lines)"; \
	 else \
	   echo "HOSTCHECK FAIL -- the interpreter does not reproduce the script:"; \
	   diff -u $(CMD_TAPE) $(BUILD)/roundtrip.txt | head -40; \
	   exit 1; \
	 fi

# The same check for a SPLIT tape. tools/tape_split.py moves the weight rows
# out of the command stream into a blob, taking a translation tape from 64.3 MB
# to 0.91 MB; this proves the substitution is lossless through the very C
# interpreter that runs on the board -- WBLK commands plus the blob must replay
# into the ORIGINAL inline-WROW script, byte for byte. Python round-tripping in
# tape_split.py checks the file transform; this checks the thing that consumes it.
WBLK_TAPE ?= $(BUILD)/wblk_cmd.txt
WBLK_BLOB ?= $(BUILD)/wblk_wgt.bin

hostcheck-wblk: $(BUILD)/vectors_host
	@$(REPO)/tools/tape_split.py split $(CMD_TAPE) $(WBLK_TAPE) $(WBLK_BLOB) >/dev/null
	@$(BUILD)/vectors_host $(WBLK_TAPE) $(RES_TAPE) $(BUILD)/wblk_roundtrip.txt \
	   $(WBLK_BLOB) > $(BUILD)/hostcheck_wblk.log 2>&1 || true
	@if cmp -s $(CMD_TAPE) $(BUILD)/wblk_roundtrip.txt; then \
	   echo "HOSTCHECK-WBLK PASS -- $$(du -h $(WBLK_TAPE) | cut -f1) of commands"\
	        "+ $$(du -h $(WBLK_BLOB) | cut -f1) of weights replay into"\
	        "$(notdir $(CMD_TAPE)) exactly ($$(wc -l < $(CMD_TAPE)) records)"; \
	 else \
	   echo "HOSTCHECK-WBLK FAIL -- split tape does not reproduce the script:"; \
	   cmp $(CMD_TAPE) $(BUILD)/wblk_roundtrip.txt | head -5; \
	   exit 1; \
	 fi

clean:
	rm -rf $(BUILD)
