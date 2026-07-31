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

set_multicycle_path -setup 2 -from [get_ports host_re]
set_multicycle_path -hold  1 -from [get_ports host_re]
set_multicycle_path -setup 2 -from [get_ports {host_raddr[*]}]
set_multicycle_path -hold  1 -from [get_ports {host_raddr[*]}]

set_multicycle_path -setup 2 -from [get_ports wstream_valid]
set_multicycle_path -hold  1 -from [get_ports wstream_valid]
set_multicycle_path -setup 2 -from [get_ports {wstream_data[*]}]
set_multicycle_path -hold  1 -from [get_ports {wstream_data[*]}]
set_multicycle_path -setup 2 -from [get_ports {wstream_row[*]}]
set_multicycle_path -hold  1 -from [get_ports {wstream_row[*]}]
set_multicycle_path -setup 2 -from [get_ports {wstream_bank[*]}]
set_multicycle_path -hold  1 -from [get_ports {wstream_bank[*]}]

# ---------------------------------------------------------------------
# Job interface and config register file (added 2026-07-31).
#
# layer_ctrl is now a job dispatcher: the PS writes the geometry registers,
# then pulses job_start. Both happen while the core is idle, and the config
# registers may not be rewritten while `busy` is high -- that is a protocol
# rule, not a timing accident. Left unconstrained, Vivado would budget
# these at a single cycle from a chip boundary that in reality moves once
# per job, and at run 9's +58 ps of margin that is the difference between
# a real failure and an artefact.
#
# Multicycle rather than false, so they are still CHECKED. A false path
# here would hide a genuine setup problem on cfg_data, which fans out to
# k_len, the requantisation scale and every base address in the design.
# ---------------------------------------------------------------------
set_multicycle_path -setup 2 -from [get_ports job_start]
set_multicycle_path -hold  1 -from [get_ports job_start]
set_multicycle_path -setup 2 -from [get_ports {job_op[*]}]
set_multicycle_path -hold  1 -from [get_ports {job_op[*]}]
set_multicycle_path -setup 2 -from [get_ports cfg_we]
set_multicycle_path -hold  1 -from [get_ports cfg_we]
set_multicycle_path -setup 2 -from [get_ports {cfg_addr[*]}]
set_multicycle_path -hold  1 -from [get_ports {cfg_addr[*]}]
set_multicycle_path -setup 2 -from [get_ports {cfg_data[*]}]
set_multicycle_path -hold  1 -from [get_ports {cfg_data[*]}]

# Score-memory readback. Verification only, and driven only while idle.
set_multicycle_path -setup 2 -from [get_ports host_sc_re]
set_multicycle_path -hold  1 -from [get_ports host_sc_re]
set_multicycle_path -setup 2 -from [get_ports {host_sc_raddr[*]}]
set_multicycle_path -hold  1 -from [get_ports {host_sc_raddr[*]}]

# ---------------------------------------------------------------------
# Out-of-context synthesis has no package pins, so no I/O constraints
# appear here. Add them when this is wrapped in a Zynq block design with
# real AXI and DDR interfaces -- that is the point at which pin
# assignment becomes meaningful, not before.
# ---------------------------------------------------------------------
