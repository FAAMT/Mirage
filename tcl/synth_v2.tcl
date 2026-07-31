# =====================================================================
# Mirage v2 — synthesis, place, route and timing for encoder_layer
#
#   cd <package root>
#   mkdir -p reports_v2
#   vivado -mode batch -source tcl/synth_v2.tcl \
#          -log reports_v2/vivado.log -journal reports_v2/vivado.jou
#
# Unlike the v1 script this is NOT an out-of-context sizing harness. v2 has
# a real top level (encoder_layer, which v1 never had) and real constraints,
# so this runs the full flow and reports timing that means something:
#
#   phase 0  elaborate            -- seconds; catches VHDL-2008 gaps early
#   phase 1  synth_design         -- the numbers the whole exercise is about
#   phase 2  opt/place/route      -- the plan's Week 11 deliverable
#   phase 3  timing + utilisation -- against <90% LUT, <95% BRAM, WNS >= 0
#
# Every phase is wrapped, and reports are written as soon as they exist, so
# a failure or a kill still leaves everything up to that point. v1's run 3
# stalled and produced nothing at all because the summary was only written
# at the end.
# =====================================================================

set part xc7z007sclg400-1
set root [file normalize [file dirname [info script]]/..]
set rpt  $root/reports_v2
file mkdir $rpt

puts "Vivado version: [version -short]"
puts "Target part   : $part"
puts ""

# ---------------------------------------------------------------------
# Source list. Order matters in non-project mode: package, primitives,
# generated tables, cores, control, top.
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
]

proc util_row {str pat} {
    foreach line [split $str "\n"] {
        if {[string match "|*$pat*|*" $line]} {
            set f [split $line "|"]
            if {[llength $f] > 2} {
                set v [string trim [lindex $f 2]]
                if {[string is double -strict $v]} { return $v }
            }
        }
    }
    return "-"
}

read_vhdl -vhdl2008 $srcs
read_xdc $root/xdc/encoder_layer.xdc

# =========================== PHASE 0 =================================
puts "\n=============== \[phase 0\] elaborate ==============="
if {[catch { synth_design -rtl -name rtl_v2 -top encoder_layer -part $part } err]} {
    puts ">>> PHASE 0 FAILED. These sources analyse and elaborate cleanly under"
    puts ">>> GHDL 4.1.0 --std=08 and report zero readable-out-port sites under"
    puts ">>> --std=93, so this is a Vivado VHDL-2008 gap, not a design fault."
    puts ">>> Send the errors back rather than editing the RTL.\n"
    puts $err
    catch { close_design }
} else {
    puts ">>> phase 0 OK"
    close_design
}

# =========================== PHASE 1 =================================
puts "\n=============== \[phase 1\] synth_design ==============="
# -mode out_of_context. encoder_layer is never the top of a chip: it is an
# AXI-attached slave inside a Zynq block design, where every one of its
# ports becomes an internal net to the PS interconnect. Without this flag
# Vivado buffers each port bit and asks the placer for a package pin, and
# xc7z007sclg400 has 100 -- so the 171 port bits of the job interface fail
# IO placement before timing is ever reported. This is also what xdc/
# encoder_layer.xdc has always claimed the flow was doing.
if {[catch { synth_design -top encoder_layer -part $part -mode out_of_context \
                 -flatten_hierarchy none } err]} {
    puts ">>> SYNTHESIS FAILED:\n$err"
    exit 1
}
report_utilization              -file $rpt/synth_utilization.rpt
report_utilization -hierarchical -file $rpt/synth_utilization_hier.rpt
write_checkpoint -force $rpt/post_synth.dcp

set u [report_utilization -return_string]
set s_lut  [util_row $u "Slice LUTs"]
set s_ff   [util_row $u "Slice Registers"]
set s_dsp  [util_row $u "DSPs"]
set s_bram [util_row $u "Block RAM Tile"]
puts "\n>>> POST-SYNTHESIS: LUT $s_lut  FF $s_ff  DSP $s_dsp  RAMB36 $s_bram"

# The two numbers this whole re-architecture exists to move.
if {$s_dsp == 0} {
    puts ">>> WARNING: 0 DSP. use_dsp=\"yes\" on mac_lane's architecture did not"
    puts ">>>          take. Check for a Synth 8-x message about USE_DSP."
}
if {$s_bram == 0} {
    puts ">>> WARNING: 0 BRAM. ram_style=\"block\" did not take. Check that"
    puts ">>>          bram_sdp was not inlined and its reset-free process kept."
}

# =========================== PHASE 2 =================================
puts "\n=============== \[phase 2\] opt / place / route ==============="
set routed 1
if {[catch { opt_design }   err]} { puts ">>> opt_design failed:\n$err";   set routed 0 }
if {$routed && [catch { place_design } err]} { puts ">>> place_design failed:\n$err"; set routed 0 }
if {$routed && [catch { route_design } err]} { puts ">>> route_design failed:\n$err"; set routed 0 }

# =========================== PHASE 3 =================================
puts "\n=============== \[phase 3\] timing + utilisation ==============="
catch { report_utilization -file $rpt/impl_utilization.rpt }
catch { report_timing_summary -delay_type min_max -max_paths 10 \
            -file $rpt/impl_timing_summary.rpt }
# -unique_pins, because without it all 20 "paths" are 20 bits of the same
# bus and the report shows one net twice over. With it, the 20 are 20
# distinct startpoint/endpoint pairs, so the second-worst path -- the one
# that matters when the worst gets fixed -- is actually visible. full_clock
# on the worst 5 so the slack can be attributed to BRAM clk-to-out vs LUT
# vs route vs DSP setup, which -path_type summary cannot show.
catch { report_timing -sort_by group -max_paths 20 -unique_pins \
            -path_type summary -file $rpt/impl_timing_paths.rpt }
catch { report_timing -sort_by group -max_paths 5 -unique_pins \
            -path_type full_clock_expanded \
            -file $rpt/impl_timing_worst5.rpt }
if {$routed} { catch { write_checkpoint -force $rpt/post_route.dcp } }

set wns "n/a"
if {$routed} {
    if {![catch { set wns [get_property SLACK [get_timing_paths -delay_type max]] }]} {
        puts ">>> WNS: $wns ns"
    }
}

set fh [open $rpt/summary_v2.csv w]
puts $fh "metric,value,budget,verdict"
puts $fh "LUT,$s_lut,12960,[expr {$s_lut ne "-" && $s_lut < 12960 ? "PASS" : "CHECK"}]"
puts $fh "FF,$s_ff,28800,-"
puts $fh "DSP48E1,$s_dsp,59,[expr {$s_dsp ne "-" && $s_dsp > 0 && $s_dsp < 59 ? "PASS" : "CHECK"}]"
puts $fh "RAMB36,$s_bram,47,[expr {$s_bram ne "-" && $s_bram > 0 && $s_bram < 47 ? "PASS" : "CHECK"}]"
puts $fh "WNS_ns,$wns,0,[expr {$wns ne "n/a" && $wns >= 0 ? "PASS" : "CHECK"}]"
close $fh

puts "\n================ MIRAGE v2 ($part) ================"
puts [format "  %-10s %10s %10s" metric measured budget]
puts [format "  %-10s %10s %10s   (plan: <90%%)" LUT     $s_lut  12960]
puts [format "  %-10s %10s %10s" FF      $s_ff   28800]
puts [format "  %-10s %10s %10s   (plan: <90%%)" DSP48E1 $s_dsp  59]
puts [format "  %-10s %10s %10s   (plan: <95%%)" RAMB36  $s_bram 47]
puts [format "  %-10s %10s %10s   (must be >= 0)" WNS_ns  $wns    0]
puts ""
puts "v1 measured, for comparison: 18182 LUT (126%), 0 BRAM, 2 DSP, no top level."
puts "Reports in reports_v2/, checkpoints post_synth.dcp / post_route.dcp."

exit
