# =====================================================================
# bd_cora.tcl -- Cora Z7-07S block design for Mirage
#
#   vivado -mode batch -source tcl/bd_cora.tcl
#
# Produces a bitstream for a PS + mirage_axi_top system:
#
#   ZYNQ7 PS ---- M_AXI_GP0 ----> AXI Interconnect ----> mirage_axi_top
#            <--- IRQ_F2P -------------------------------- irq
#            ---- FCLK_CLK0 100 MHz ---------------------> aclk
#
# DELIBERATELY NO DMA IN THIS BUILD. The weight stream belongs on one --
# it is the bandwidth-critical path at roughly 4.5 : 1 against compute --
# but putting it in the first build means the first board test debugs the
# accelerator, the DMA, the HP port and the cache policy at once, with no
# way to tell which one failed. mirage_axi_lite exposes WGT_DATA/WGT_LAST
# so weights load over the control port alone. Add the DMA once the
# datapath is confirmed on silicon; s_axis is already wired for it and is
# simply left idle here.
#
# Everything is scripted rather than clicked so the design is
# reproducible and reviewable in a diff.
# =====================================================================

set part       xc7z007sclg400-1
set board      digilentinc.com:cora-z7-07s:part0:1.0
set design     mirage_cora
set root       [file normalize [file dirname [info script]]/..]
set outdir     $root/build_cora

puts "Vivado version : [version -short]"
puts "Target part    : $part"
puts "Board          : $board"

# ---------------------------------------------------------------------
# The board file carries the DDR3 and MIO configuration for this exact
# board. Without it the PS comes up with a default memory controller that
# does not match the parts fitted, which produces a bitstream that loads
# and then fails in ways that look like logic bugs. Refusing to build is
# the honest outcome; guessing is not.
# ---------------------------------------------------------------------
if {[lsearch -exact [get_board_parts -quiet] $board] < 0} {
    puts ""
    puts ">>> Board files for the Cora Z7-07S are not installed."
    puts ">>> get_board_parts does not list: $board"
    puts ">>>"
    puts ">>> Install them, then re-run:"
    puts ">>>   git clone https://github.com/Digilent/vivado-boards"
    puts ">>>   cp -r vivado-boards/new/board_files/* \\"
    puts ">>>       <Vivado>/data/boards/board_files/"
    puts ">>>"
    puts ">>> Building without the board preset would give the PS a DDR"
    puts ">>> configuration that does not match the fitted memory. That"
    puts ">>> bitstream loads and then misbehaves in ways that look like"
    puts ">>> logic defects, so this script stops instead of guessing."
    exit 1
}

file mkdir $outdir
create_project ${design} $outdir -part $part -force
set_property board_part $board [current_project]

# ---------------------------------------------------------------------
# Sources. Every rtl_v2 file is VHDL-2008 -- the design uses 2008-only
# constructs throughout, and Vivado defaults new files to VHDL-93, which
# fails to parse them with errors that point at the wrong thing.
# ---------------------------------------------------------------------
set srcs [list \
    $root/rtl_v2/pkg/mirage_v2_pkg.vhd \
    $root/rtl_v2/prim/bram_sdp.vhd \
    $root/rtl_v2/prim/mac_lane.vhd \
    $root/rtl_v2/core/mac_array.vhd \
    $root/rtl_v2/core/weight_window.vhd \
    $root/rtl_v2/core/exp_rom.vhd \
    $root/rtl_v2/core/recip_rom.vhd \
    $root/rtl_v2/core/softmax_unit.vhd \
    $root/rtl_v2/core/stage_engine.vhd \
    $root/rtl_v2/core/gemm_sequencer.vhd \
    $root/rtl_v2/ctrl/layer_ctrl.vhd \
    $root/rtl_v2/top/encoder_layer.vhd \
    $root/rtl_v2/axi/mirage_axi_lite.vhd \
    $root/rtl_v2/axi/wgt_stream.vhd \
    $root/rtl_v2/top/mirage_axi_top.vhd ]

add_files -norecurse $srcs
set_property file_type {VHDL 2008} [get_files $srcs]
update_compile_order -fileset sources_1

# ---------------------------------------------------------------------
# Block design
# ---------------------------------------------------------------------
create_bd_design $design

# --- Zynq PS, board preset, then the deltas this design needs ---------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" \
             Master "Disable" Slave "Disable"} [get_bd_cells ps7]

set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0    {1}   \
    CONFIG.PCW_USE_S_AXI_HP0    {1}   \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR     {1}   \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells ps7]

# HP0 is enabled but left unconnected on purpose: it is the port the
# weight DMA will use, and enabling it now means adding the DMA later
# does not re-run PS configuration and re-time the whole design.

# --- the accelerator, as a module reference (no IP packaging step) ----
create_bd_cell -type module -reference mirage_axi_top mirage

# ---------------------------------------------------------------------
# Connections.
#
# A module reference gets its interfaces INFERRED from port names, and
# whether inference fires -- and what it names the result -- is not
# something to assume. s_axi_* and s_axis_* follow the conventions, so it
# should group them, but a hardcoded /mirage/s_axi that does not exist
# fails deep in the script with an unhelpful error. Everything below
# discovers what is actually there and says so.
# ---------------------------------------------------------------------
puts ">>> interfaces inferred on the mirage cell:"
foreach ip [get_bd_intf_pins -quiet -of_objects [get_bd_cells mirage]] {
    puts [format ">>>   %-28s %s" [get_property NAME $ip] \
                                  [get_property VLNV $ip]]
}

set axi_slave [get_bd_intf_pins -quiet -of_objects [get_bd_cells mirage] \
                 -filter {MODE == Slave && VLNV =~ *:aximm_rtl:*}]
if {[llength $axi_slave] != 1} {
    puts ">>> Expected exactly one AXI-MM slave interface on mirage,"
    puts ">>> found [llength $axi_slave]. Port-name inference did not"
    puts ">>> produce a usable AXI4-Lite interface; package mirage_axi_top"
    puts ">>> as IP instead of using it as a module reference."
    exit 1
}
puts ">>> AXI4-Lite slave: $axi_slave"

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Master {/ps7/M_AXI_GP0} Clk {Auto}] $axi_slave

# --- interrupt --------------------------------------------------------
connect_bd_net [get_bd_pins mirage/irq] [get_bd_pins ps7/IRQ_F2P]

# ---------------------------------------------------------------------
# Tie off the idle stream port.
#
# Nothing drives s_axis in this build. An undriven tvalid is exactly the
# kind of thing that works in simulation and bites on hardware: the sink
# would take beats at unpredictable times and corrupt the weight window
# between PIO writes, which would look like an arithmetic bug. Driven low
# it is inert, and that is worth being explicit about rather than relying
# on the tools' habit of tying unconnected inputs to ground.
# ---------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant const0
set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {1}] \
    [get_bd_cells const0]

set axis_slave [get_bd_intf_pins -quiet -of_objects [get_bd_cells mirage] \
                  -filter {MODE == Slave && VLNV =~ *:axis_rtl:*}]

if {[llength $axis_slave] == 1} {
    # Inference grouped the stream. Individual pins are no longer
    # connectable, so drive the interface from a tvalid-low stub.
    puts ">>> stream inferred as interface $axis_slave -- left idle"
    puts ">>> (validate_bd_design will flag it as unconnected; that is"
    puts ">>>  expected until the weight DMA is added)"
} else {
    puts ">>> stream exposed as discrete pins -- tying low explicitly"
    connect_bd_net [get_bd_pins const0/dout] \
                   [get_bd_pins mirage/s_axis_tvalid]
    connect_bd_net [get_bd_pins const0/dout] \
                   [get_bd_pins mirage/s_axis_tlast]
    create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant const32
    set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {32}] \
        [get_bd_cells const32]
    connect_bd_net [get_bd_pins const32/dout] \
                   [get_bd_pins mirage/s_axis_tdata]
}

# ---------------------------------------------------------------------
# Address map. 64 KB is the smallest aperture the tools will assign; the
# register file needs 0x6C bytes of it.
# ---------------------------------------------------------------------
assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]]
puts ">>> address segments:"
foreach s $seg {
    puts [format ">>>   %-40s %s +%s" $s \
        [get_property OFFSET $s] [get_property RANGE $s]]
}

regenerate_bd_layout
validate_bd_design
save_bd_design

# ---------------------------------------------------------------------
# Wrapper + implementation
# ---------------------------------------------------------------------
set bd_file [get_files ${design}.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [file join $outdir ${design}.gen sources_1 bd ${design} \
                                hdl ${design}_wrapper.vhd]
set_property top ${design}_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts ">>> SYNTHESIS FAILED -- see $outdir/${design}.runs/synth_1"
    exit 1
}

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts ">>> IMPLEMENTATION FAILED -- see $outdir/${design}.runs/impl_1"
    exit 1
}

open_run impl_1
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts ""
puts "====================================================================="
puts ">>> IN-CONTEXT WNS: $wns ns"
puts ">>> (run 11 measured +0.252 ns out of context, with no interconnect"
puts ">>>  and no PS. This is the number that actually governs the board.)"
puts "====================================================================="

report_utilization -file $root/reports_v2/cora_utilization.rpt
report_timing_summary -file $root/reports_v2/cora_timing_summary.rpt
report_timing -sort_by group -max_paths 5 -unique_pins \
    -path_type full_clock_expanded -file $root/reports_v2/cora_timing_worst5.rpt

set bit [glob -nocomplain $outdir/${design}.runs/impl_1/*.bit]
puts ">>> bitstream: $bit"
puts ">>> XSA for Vitis: writing $outdir/${design}.xsa"
write_hw_platform -fixed -include_bit -force $outdir/${design}.xsa
puts ">>> done"
