# =====================================================================
# Mirage — resource utilisation without a block diagram
#
#   cd <package root>
#   vivado -mode batch -source tcl/synth_utilization.tcl \
#          -log reports/vivado.log -journal reports/vivado.jou
#
# Two phases, one run:
#
#   Phase 1 — per-module out-of-context synthesis. Gives the per-block
#             profile: which module owns the DSPs, which owns the BRAM.
#             Do NOT sum this table: qkv_proj_bram and quantizer are
#             counted both standalone and inside qkv_score_calc, and
#             smx_cache/smx_v_calc both standalone and inside smx_core.
#
#   Phase 2 — one synthesis of mirage_sizing_subset, which holds each
#             synthesisable ROOT module exactly once. -flatten_hierarchy none
#             keeps module boundaries so the hierarchical report breaks it
#             back down.
#
# IMPORTANT: this is a SUBSET, 6 of 9 root modules. ffn, multi_head and
# qkv_proj_calc are excluded because their weight arrays are read 60-wide
# inside a 12x5 loop and no FPGA memory primitive supports that -- Vivado
# errors with [Synth 8-3391] after a long grind. They appear as BLOCKED rows
# in the summary. Run tools/memory_audit.py, and read KNOWN_ISSUES.md
# section 7 before quoting any total as "the design".
#
# Phase 2 failing does not cost you phase 1: each synth is wrapped and
# the batch continues.
# =====================================================================

set part xc7z007sclg400-1  ;# Cora Z7-07S. Change for a different board.
set root [file normalize [file dirname [info script]]/..]
file mkdir $root/reports

puts "Vivado version: [version -short]"
puts "Target part   : $part"
puts ""
puts "If phase 0 below reports VHDL-2008 parse errors, that is a Vivado"
puts "version limit, not a design problem -- these files analyse cleanly under"
puts "GHDL 4.1.0. Vivado's VHDL-2008 synthesis support is partial and weaker in"
puts "2020 than 2022. Report the phase-0 errors rather than editing the RTL."
puts ""

# ---------------------------------------------------------------------
# Source list. VHDL is order-sensitive in non-project mode: packages
# first, then leaves, then the modules that instantiate them.
# ---------------------------------------------------------------------
set srcs [list \
    $root/rtl/pkgs/types_pkg.vhd \
    $root/rtl/pkgs/types_pkg_body.vhd \
    $root/rtl/outputs/rc2_hardening.vhd \
    $root/rtl/mirage/gemm_array.vhd \
    $root/rtl/mirage/quantizer.vhd \
    $root/rtl/mirage/qkv_proj_bram.vhd \
    $root/rtl/mirage/qk_score_bram.vhd \
    $root/rtl/mirage/qkv_proj_calc.vhd \
    $root/rtl/mirage/qkv_score_calc.vhd \
    $root/rtl/mirage/multi_head.vhd \
    $root/rtl/mirage/taylor_exp.vhd \
    $root/rtl/mirage/ffn.vhd \
    $root/rtl/outputs/smx_cache.vhd \
    $root/rtl/outputs/smx_v_calc.vhd \
    $root/rtl/outputs/smx_core.vhd \
    $root/rtl/outputs/weight_stream_ctrl.vhd \
    $root/rtl/integration/mirage_sizing_container.vhd \
    $root/rtl/integration/mirage_sizing_subset.vhd \
]

# Phase 1: modules that can be synthesised. Children marked so the table
# can flag them as already counted inside a root.
# ORDER MATTERS. Run 3 (2026-07-30) stalled inside qkv_proj_bram, whose
# bram_qkv degrades to a 49,152-bit register file, and because the summary was
# only written at the end that run produced NO table at all -- despite having
# already synthesised gemm_array and quantizer successfully.
#
# So: cheap and informative modules first, memory-degrading ones last, and the
# CSV is flushed after every module (see PHASE 1). A stall now costs you only
# the modules after it.
set tops [list \
    {gemm_array         root} \
    {quantizer          child-of-qkv_score_calc} \
    {taylor_exp         root} \
    {qk_score_bram      root} \
    {smx_cache          child-of-smx_core} \
    {smx_v_calc         child-of-smx_core} \
    {smx_core           root} \
    {weight_stream_ctrl root} \
    {qkv_proj_bram      child-of-qkv_score_calc} \
    {qkv_score_calc     root} \
]

# Excluded, with the reason reported in the summary table rather than hidden.
# These hold weight arrays read 60-wide inside a 12x5 loop; BRAM offers 2 read
# ports, LUTRAM 16. Vivado degrades them to registers and errors with
# [Synth 8-3391] -- after grinding for a long time first, which is why they are
# excluded rather than merely caught. See KNOWN_ISSUES.md section 7 and
# tools/memory_audit.py.
set blocked [list \
    {ffn           {bram_hlw/olw/proj 2.2 Mb, 60 read ports}} \
    {multi_head    {bram_qkv 786 kb, 60 read ports}} \
    {qkv_proj_calc {bram_qkv 786 kb, 60 read ports}} \
]

# ---------------------------------------------------------------------
# Pull a count out of a report_utilization string by row label. Tolerant
# of the trailing "*" Vivado puts on some rows and of column padding.
# ---------------------------------------------------------------------
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

proc grab {top} {
    set u [report_utilization -return_string]
    return [list [util_row $u "Slice LUTs"] \
                 [util_row $u "Slice Registers"] \
                 [util_row $u "DSPs"] \
                 [util_row $u "Block RAM Tile"]]
}

# Read once for the session. In non-project mode the file set persists
# across synth_design/close_design.
read_vhdl -vhdl2008 $srcs

set summary {}

# Open the CSV now and flush after every module, so a stall or a kill still
# leaves you everything completed up to that point.
set fh [open $root/reports/summary.csv w]
puts $fh "module,kind,status,LUT,FF,DSP48E1,RAMB36"
flush $fh

# =========================== PHASE 0 =================================
# RTL elaboration only (-rtl): seconds, not minutes, and it surfaces
# VHDL-2008 parse problems before you sit through a full synthesis run.
# If this fails, stop and send the errors -- it is a Vivado version
# limit, not something to fix by editing the RTL.
puts "\n=============== \[phase 0\] RTL elaboration check ==============="
if {[catch {
        synth_design -rtl -name rtl_check -top mirage_sizing_subset -part $part
    } err]} {
    puts "\n>>> PHASE 0 FAILED -- Vivado could not elaborate the sources."
    puts ">>> These files analyse and elaborate cleanly under GHDL 4.1.0 --std=08,"
    puts ">>> so this is almost certainly a VHDL-2008 support gap in [version -short]."
    puts ">>> Send these errors back rather than editing the RTL:\n"
    puts $err
    puts "\n>>> Continuing to phase 1 anyway: individual modules may still"
    puts ">>> synthesise even if the container does not."
    catch { close_design }
} else {
    puts ">>> Phase 0 OK -- Vivado elaborates the full source set."
    close_design
}

# =========================== PHASE 1 =================================
foreach entry $tops {
    lassign $entry top kind
    puts "\n=============== \[phase 1\] $top ==============="
    if {[catch { synth_design -mode out_of_context -top $top -part $part } err]} {
        puts "SYNTH FAILED for $top:\n$err"
        set row [list $top $kind FAILED - - - -]
        lappend summary $row
        puts $fh [join $row ","]; flush $fh
        catch { close_design }
        continue
    }
    report_utilization -file $root/reports/util_$top.rpt
    set row [concat [list $top $kind OK] [grab $top]]
    lappend summary $row
    puts $fh [join $row ","]; flush $fh
    puts [format "  -> %s" [join $row " "]]
    close_design
}

# =========================== PHASE 2 =================================
puts "\n=============== \[phase 2\] mirage_sizing_subset ==============="
set container_row {}
if {[catch {
        synth_design -mode out_of_context -top mirage_sizing_subset \
                     -part $part -flatten_hierarchy none
    } err]} {
    puts "CONTAINER SYNTH FAILED:\n$err"
    set container_row [list mirage_sizing_subset SUBSET-TOTAL FAILED - - - -]
    catch { close_design }
} else {
    report_utilization              -file $root/reports/util_CONTAINER.rpt
    report_utilization -hierarchical -file $root/reports/util_CONTAINER_hier.rpt
    set container_row [concat [list mirage_sizing_subset SUBSET-TOTAL OK] \
                              [grab mirage_sizing_subset]]
    close_design
}

# ---------------------------------------------------------------------
# Combined table -> reports/summary.csv and stdout
# ---------------------------------------------------------------------
set fmt "%-24s %-24s %-7s %8s %8s %6s %8s"

puts "\n\n================ MIRAGE UTILISATION ($part) ================"
puts [format $fmt module kind status LUT FF DSP RAMB36]
puts [string repeat - 90]
foreach r $summary {
    puts [format $fmt {*}$r]
}
foreach b $blocked {
    lassign $b m why
    set r [list $m BLOCKED "cannot-synth" - - - -]
    puts $fh [join $r ","]
    puts [format $fmt {*}$r]
    puts [format "%-24s %s" "" "  reason: $why"]
}
puts [string repeat - 90]
puts $fh [join $container_row ","]
puts [format $fmt {*}$container_row]
close $fh

puts "\nDevice budget (xc7z007s): LUT 14400 | FF 28800 | DSP48E1 66 | RAMB36 50"
puts ""
puts "READ THE TABLE THIS WAY:"
puts "  * SUBSET-TOTAL is 6 of 9 root modules, each once. NOT the design."
puts "  * rows marked child-of-* are already inside a root row. They are"
puts "    there for the per-block profile only. Summing the phase-1 rows"
puts "    double-counts them."
puts "  * BLOCKED rows are excluded from the run, not zero-cost. Their"
puts "    weight memories are not implementable as written -- see"
puts "    KNOWN_ISSUES.md section 7 and tools/memory_audit.py."
puts "  * SUBSET-TOTAL therefore covers 6 of 9 root modules, NOT the design."
puts "  * qkv_proj_bram (inside qkv_score_calc) quietly becomes 49,152 FF on a"
puts "    28,800 FF device. Expect FF over 100%: that is a memory that must"
puts "    become real BRAM, not a logic cost."
puts "  * the container is a sizing harness, not the real architecture."
puts "    Unconnected module ports are exposed rather than wired, because"
puts "    the source does not yet define those connections (KNOWN_ISSUES)."
puts ""
puts "Per-module reports in reports/, hierarchical breakdown in"
puts "reports/util_CONTAINER_hier.rpt, combined table in reports/summary.csv"

exit
