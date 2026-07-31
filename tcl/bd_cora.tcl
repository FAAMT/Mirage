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
set board_glob digilentinc.com:cora-z7-07s:*
set design     mirage_cora
set root       [file normalize [file dirname [info script]]/..]
set outdir     $root/build_cora

puts "Vivado version : [version -short]"
puts "Target part    : $part"
puts "Board (glob)   : $board_glob"

# ---------------------------------------------------------------------
# FINDING THE BOARD FILES
#
# Do not copy anything into the Vivado install. <Vivado>/data/boards/
# board_files/ was the 2019-era location and newer releases no longer
# ship that directory at all -- boards now arrive through the Vivado
# Store into data/xhub/. Writing into the toolchain also needs root on
# most installs and is silently lost when Vivado is upgraded.
#
# Instead point board.repoPaths at a checkout anywhere on disk. Set
# MIRAGE_BOARD_REPO to the directory that CONTAINS the board folders --
# for the Digilent repo that is <clone>/new/board_files, the level with
# cora-z7-07s/ inside it, not the clone root.
#
# Matching is a glob, not an exact string: the board part carries a
# revision suffix (:1.0, :1.1, ...) that changes between repo versions,
# and pinning it exactly turns a cosmetic bump into a build failure.
# ---------------------------------------------------------------------
set repo_paths [list]
if {[info exists ::env(MIRAGE_BOARD_REPO)]} {
    lappend repo_paths $::env(MIRAGE_BOARD_REPO)
}
foreach c [list $root/boards \
                ~/vivado-boards/new/board_files \
                ~/.Xilinx/Vivado/board_files] {
    lappend repo_paths $c
}

set repo_found [list]
foreach c $repo_paths {
    set c [file normalize $c]
    if {[file isdirectory $c] && [lsearch -exact $repo_found $c] < 0} {
        lappend repo_found $c
    }
}
if {[llength $repo_found]} {
    puts ">>> board repo paths: $repo_found"
    set_param board.repoPaths $repo_found
}

set matches [lsearch -all -inline [get_board_parts -quiet] $board_glob]
if {[llength $matches] == 0} {
    puts ""
    puts ">>> No board part matches $board_glob"
    puts ">>>"
    puts ">>> The board file carries the DDR3 and MIO configuration for"
    puts ">>> this exact board. Without it the PS comes up with a memory"
    puts ">>> controller that does not match the parts fitted, which"
    puts ">>> produces a bitstream that loads and then misbehaves in ways"
    puts ">>> that look like logic defects. So this stops instead of"
    puts ">>> guessing at DDR timings."
    puts ">>>"
    puts ">>> Either route works:"
    puts ">>>"
    puts ">>> (a) clone it anywhere, no root needed --"
    puts ">>>       git clone https://github.com/Digilent/vivado-boards ~/vivado-boards"
    puts ">>>       export MIRAGE_BOARD_REPO=~/vivado-boards/new/board_files"
    puts ">>>     (or drop the board folders in $root/boards, which this"
    puts ">>>      script checks with no environment variable at all)"
    puts ">>>"
    puts ">>> (b) Vivado GUI: Tools > Vivado Store > Boards, install the"
    puts ">>>     Cora Z7-07S, then re-run this script."
    puts ">>>"
    if {[llength $repo_found] == 0} {
        puts ">>> No candidate repo directory exists yet. Checked:"
        foreach c $repo_paths { puts ">>>   [file normalize $c]" }
    } else {
        puts ">>> Searched these repo paths and found no cora-z7-07s:"
        foreach c $repo_found { puts ">>>   $c" }
        puts ">>> If the clone is there, the path is probably one level"
        puts ">>> off -- it must contain cora-z7-07s/, not new/."
    }
    set any [get_board_parts -quiet]
    puts ">>> ([llength $any] board parts visible in total)"
    exit 1
}

# Highest revision wins if several are visible.
set board [lindex [lsort -dictionary $matches] end]
if {[llength $matches] > 1} {
    puts ">>> several revisions visible: $matches"
}
puts ">>> using board part: $board"

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

set ip_proj $outdir/ip_proj
set ip_repo $outdir/ip_repo
set bd_proj $outdir/bd_proj
set ip_vlnv mirage:user:mirage_axi_top:1.0

file mkdir $outdir

# =====================================================================
# STAGE 1 -- package mirage_axi_top as IP
#
# This used to be a module reference (create_bd_cell -type module).
# Vivado 2022.2 refuses:
#
#   [filemgmt 56-195] Reference 'mirage_axi_top' contains top file
#   'mirage_axi_top.vhd' of type VHDL 2008. This type is not allowed
#   as the top file in the reference.
#
# Module references require a VHDL-93 or Verilog top file. This design
# is VHDL-2008 throughout and is not going to stop being; the IP
# packager has no such restriction.
#
# Packaging is the better shape anyway. It declares the AXI interfaces
# and the memory map explicitly instead of hoping port-name inference
# fires in the block design, and it lets the address block be sized
# here rather than argued with at assign_bd_address time.
# =====================================================================
file delete -force $ip_repo
file mkdir $ip_repo

create_project mirage_ip $ip_proj -part $part -force
add_files -norecurse $srcs
set_property file_type {VHDL 2008} [get_files $srcs]
set_property top mirage_axi_top [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ip_repo -vendor mirage -library user \
    -taxonomy /UserIP -import_files -force

set core [ipx::current_core]
set_property name          mirage_axi_top $core
set_property version       1.0            $core
set_property display_name  "Mirage int8 encoder layer" $core
set_property description \
    "int8 transformer encoder layer; AXI4-Lite control, AXI4-Stream weights" \
    $core
set_property core_revision 1 $core

# --- what did the packager actually infer? ---------------------------
proc busif_matching {core pat} {
    foreach bi [ipx::get_bus_interfaces -of_objects $core] {
        if {[string match $pat [get_property BUS_TYPE_VLNV $bi]]} {
            return [get_property NAME $bi]
        }
    }
    return ""
}

puts ">>> bus interfaces inferred by the packager:"
foreach bi [ipx::get_bus_interfaces -of_objects $core] {
    puts [format ">>>   %-14s %s" [get_property NAME $bi] \
                                  [get_property BUS_TYPE_VLNV $bi]]
}

set if_axi  [busif_matching $core *:aximm_rtl:*]
set if_axis [busif_matching $core *:axis_rtl:*]

if {$if_axi eq ""} {
    puts ">>> The packager did not infer an AXI4-Lite slave interface."
    puts ">>> Port names on mirage_axi_top must follow the s_axi_* "
    puts ">>> convention for inference to fire. Nothing downstream can"
    puts ">>> work without it, so stopping here rather than failing"
    puts ">>> later inside the block design."
    exit 1
}
puts ">>> AXI4-Lite interface : $if_axi"
puts ">>> AXI-Stream interface: [expr {$if_axis eq "" ? "(none inferred)" : $if_axis}]"

# Associate the clock with each interface. BD automation can usually
# work this out by itself, so a failure here is a warning, not a stop.
foreach b [list $if_axi $if_axis] {
    if {$b eq ""} { continue }
    if {[catch {ipx::associate_bus_interfaces -busif $b -clock aclk $core} e]} {
        puts ">>> note: could not associate aclk with $b ($e)"
        puts ">>> continuing -- axi4 automation connects clocks anyway"
    }
}

# ---------------------------------------------------------------------
# Address block size. The register file decodes 8 address bits = 256
# bytes, but a Zynq GP0 segment cannot be smaller than 4K, and
# assign_bd_address will not hand out more than the memory map declares.
# Declaring 4K here settles it at package time. 0x100..0xFFF alias back
# onto the registers, which is ordinary for a small AXI4-Lite slave and
# is invisible to a driver that addresses off one base pointer.
# ---------------------------------------------------------------------
foreach m [ipx::get_memory_maps -of_objects $core] {
    foreach ab [ipx::get_address_blocks -of_objects $m] {
        set_property range 4096 $ab
        puts ">>> address block [get_property NAME $ab] range -> 4096 B"
    }
}

ipx::create_xgui_files $core
ipx::update_checksums $core
if {[catch {ipx::check_integrity $core} e]} {
    puts ">>> ipx::check_integrity reported: $e"
}
ipx::save_core $core
close_project
puts ">>> packaged $ip_vlnv into $ip_repo"

# =====================================================================
# STAGE 2 -- the block design
#
# A separate project from the packaging one. The IP carries its own
# imported copies of every source, so the raw rtl_v2 files are
# deliberately NOT added here: two copies of the same entities in one
# project is an elaboration conflict waiting to happen.
# =====================================================================
create_project ${design} $bd_proj -part $part -force

# Record the repo on the project too. set_param is a session setting, so
# without this the project reopens in the GUI unable to resolve its own
# board_part and quietly drops the preset.
if {[llength $repo_found]} {
    set_property board_part_repo_paths $repo_found [current_project]
}
set_property board_part $board [current_project]

set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog -rebuild
if {[llength [get_ipdefs -quiet $ip_vlnv]] == 0} {
    puts ">>> $ip_vlnv is not in the catalogue after update_ip_catalog."
    puts ">>> Packaging reported success, so this is a repo path problem:"
    puts ">>>   $ip_repo"
    exit 1
}

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

# --- the accelerator, from the IP catalogue ---------------------------
create_bd_cell -type ip -vlnv $ip_vlnv mirage

# ---------------------------------------------------------------------
# Connections. The interface names come from the packaged core, so they
# are known rather than guessed -- but confirm they survived onto the
# cell before using them, because a hardcoded /mirage/s_axi that does
# not exist fails deep in the script with an unhelpful error.
# ---------------------------------------------------------------------
puts ">>> interfaces on the mirage cell:"
foreach ip [get_bd_intf_pins -quiet -of_objects [get_bd_cells mirage]] {
    puts [format ">>>   %-28s %s" [get_property NAME $ip] \
                                  [get_property VLNV $ip]]
}

set axi_slave [get_bd_intf_pins -quiet -of_objects [get_bd_cells mirage] \
                 -filter {MODE == Slave && VLNV =~ *:aximm_rtl:*}]
if {[llength $axi_slave] != 1} {
    puts ">>> Expected exactly one AXI-MM slave interface on mirage,"
    puts ">>> found [llength $axi_slave]. The packaged core declared"
    puts ">>> '$if_axi' -- if that is missing here the core in $ip_repo"
    puts ">>> is stale; delete build_cora and re-run."
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
# Address map.
#
# The base is pinned rather than left to assign_bd_address. The
# bare-metal driver hardcodes it, and a base that moves when an
# unrelated peripheral is added is a bug that presents as the
# accelerator having stopped working. 0x43C00000 is the conventional
# first custom-IP slot on Zynq GP0.
# ---------------------------------------------------------------------
set MIRAGE_BASE 0x43C00000

assign_bd_address
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]] {
    if {[string match *mirage* $s]} {
        set_property offset $MIRAGE_BASE $s
        set_property range  4K           $s
    }
}

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

# make_wrapper writes under .gen in 2020.2+ and under .srcs before that.
# Take whichever exists rather than guessing at the Vivado version.
set wrap ""
foreach d [list ${design}.gen ${design}.srcs] {
    set c [file join $bd_proj $d sources_1 bd ${design} hdl ${design}_wrapper.vhd]
    if {[file exists $c]} { set wrap $c ; break }
}
if {$wrap eq ""} {
    set found [glob -nocomplain $bd_proj/*/sources_1/bd/${design}/hdl/*_wrapper.vhd]
    if {[llength $found] == 1} { set wrap [lindex $found 0] }
}
if {$wrap eq ""} {
    puts ">>> make_wrapper ran but the wrapper is not where it was expected."
    puts ">>> Looked under $bd_proj/{${design}.gen,${design}.srcs}/..."
    exit 1
}
puts ">>> wrapper: $wrap"
add_files -norecurse $wrap
set_property top ${design}_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts ">>> SYNTHESIS FAILED -- see $bd_proj/${design}.runs/synth_1"
    exit 1
}

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts ">>> IMPLEMENTATION FAILED -- see $bd_proj/${design}.runs/impl_1"
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

set bit [glob -nocomplain $bd_proj/${design}.runs/impl_1/*.bit]
puts ">>> bitstream: $bit"
puts ">>> XSA for Vitis: writing $outdir/${design}.xsa"
write_hw_platform -fixed -include_bit -force $outdir/${design}.xsa
puts ""
puts ">>> mirage register base for the driver: $MIRAGE_BASE (4K)"
puts ">>> done"
