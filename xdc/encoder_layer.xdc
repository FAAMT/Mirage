# =====================================================================
# encoder_layer.xdc — timing constraints for Mirage v2
#
# v1 had no constraints at all, and no top level to attach them to, so
# every v1 timing number was meaningless (reports/timing_*.rpt was
# unconstrained noise -- that is why the v1 script stopped emitting them).
#
# Clock choice. The MVP plan targets 200 MHz with a 100 MHz fallback and
# notes real-time NMT is met at either. 100 MHz is the constraint here on
# purpose: it is the fallback, so meeting it is the pass/fail bar, and the
# WNS at 100 MHz tells you directly how much headroom exists toward 200.
# Retarget by editing PERIOD_NS alone.
#
#   set PERIOD_NS 5.000   -> 200 MHz stretch goal
#   set PERIOD_NS 10.000  -> 100 MHz plan fallback  (default)
# =====================================================================

set PERIOD_NS 10.000

create_clock -period $PERIOD_NS -name clk [get_ports clk]

# Reset is asserted by software and held for many cycles; it is not a
# timing-critical path and constraining it as one only creates noise.
set_false_path -from [get_ports rst]

# ---------------------------------------------------------------------
# Host and DDR-stream interfaces are quasi-static relative to the core
# clock: the host writes activations only while the core is idle, and the
# weight stream is a DMA that runs a full window ahead of compute. Both
# are declared as multicycle rather than false paths, so they are still
# checked, just not against a single-cycle budget.
# ---------------------------------------------------------------------
set_multicycle_path -setup 2 -from [get_ports host_we]
set_multicycle_path -hold  1 -from [get_ports host_we]
set_multicycle_path -setup 2 -from [get_ports {host_waddr[*]}]
set_multicycle_path -hold  1 -from [get_ports {host_waddr[*]}]
set_multicycle_path -setup 2 -from [get_ports {host_wdata[*]}]
set_multicycle_path -hold  1 -from [get_ports {host_wdata[*]}]

set_multicycle_path -setup 2 -from [get_ports wstream_valid]
set_multicycle_path -hold  1 -from [get_ports wstream_valid]
set_multicycle_path -setup 2 -from [get_ports {wstream_data[*]}]
set_multicycle_path -hold  1 -from [get_ports {wstream_data[*]}]

# ---------------------------------------------------------------------
# Out-of-context synthesis has no package pins, so no I/O constraints
# appear here. Add them when this is wrapped in a Zynq block design with
# real AXI and DDR interfaces -- that is the point at which pin
# assignment becomes meaningful, not before.
# ---------------------------------------------------------------------
