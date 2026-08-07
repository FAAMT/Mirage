# =====================================================================
# sw/board/bare/Makefile -- build smoke.c for the Cora Z7 without Vitis
#
#   make            build build/mirage_smoke.elf
#   make load       program the bitstream and run it over JTAG
#   make console    open the UART at 115200 (needs the dialout group)
#   make clean
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
XSA     ?= $(SW)/../build_cora/mirage_cora.xsa
BIT     ?= $(SW)/../build_cora/mirage_cora_wrapper.bit
BUILD   := build
TARGET  := $(BUILD)/mirage_smoke.elf

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
           -Wl,-Map,$(BUILD)/mirage_smoke.map \
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
        $(BUILD)/smoke.o

.PHONY: all load console clean size
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

$(BUILD)/smoke.o: ../smoke.c | $(BUILD)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c -o $@ $<

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
	@$(OBJCOPY) -O binary $@ $(BUILD)/mirage_smoke.bin

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

console:
	@echo "115200 8N1 on /dev/ttyUSB1 -- ctrl-a k to quit screen"
	@sg dialout -c 'screen /dev/ttyUSB1 115200'

clean:
	rm -rf $(BUILD)
