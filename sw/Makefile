# =====================================================================
# sw/Makefile -- host builds only
#
# The board build is a Vitis application project; it compiles the same
# mirage.c against io_mmio.c and needs nothing from here. This Makefile
# exists so the driver can be exercised on a machine with no Vivado,
# no Vitis and no board.
#
#   make check    build the tracing driver, run it, diff the register
#                 sequence against sim/ops_cmd.txt
#   make clean
# =====================================================================
CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -Werror -O1 -I.
LDLIBS  ?= -lm

TRACE_OBJS = mirage.o io_trace.o test/trace_ops.o
SCALE_OBJS = mirage.o io_trace.o test/split_scale_dump.o

.PHONY: all check clean
all: test/trace_ops test/split_scale_dump

test/trace_ops: $(TRACE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(TRACE_OBJS) $(LDLIBS)

test/split_scale_dump: $(SCALE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SCALE_OBJS) $(LDLIBS)

test/split_scale_dump.o: test/split_scale_dump.c mirage.h mirage_regs.h
	$(CC) $(CFLAGS) -c -o $@ $<

test/trace_ops.o: test/trace_ops.c mirage.h mirage_regs.h mirage_io.h mirage_trace.h
	$(CC) $(CFLAGS) -c -o $@ $<

mirage.o: mirage.c mirage.h mirage_regs.h mirage_io.h
	$(CC) $(CFLAGS) -c -o $@ $<

io_trace.o: io_trace.c mirage_io.h mirage_regs.h mirage_trace.h
	$(CC) $(CFLAGS) -c -o $@ $<

# io_mmio.c is not built here -- it dereferences a fixed physical
# address and belongs to the board build. It is still compile-checked so
# a typo in it cannot sit undiscovered until Vitis.
.PHONY: syntax
syntax:
	$(CC) $(CFLAGS) -fsyntax-only io_mmio.c
	$(CC) $(CFLAGS) -DMIRAGE_HOST_BUILD -fsyntax-only board/smoke.c
	@echo "io_mmio.c and board/smoke.c compile"

check: test/trace_ops test/split_scale_dump syntax
	cd test && ./trace_ops ops_trace.txt
	./test/check_ops.sh
	cd test && ./split_scale_dump scale_vectors.txt | ./check_scale.py

clean:
	rm -f mirage.o io_trace.o test/trace_ops.o test/split_scale_dump.o \
	      test/trace_ops test/split_scale_dump test/ops_trace.txt
