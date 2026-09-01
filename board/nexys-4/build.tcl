# Builds the bitstream from the generated Verilog.
#
# First: make verilog-<era>, or dune exec bin/gen_verilog.exe -- <era>
# Then, from the repository root:
#   vivado -mode batch -journal board/_build/vivado.jou \
#          -log board/_build/vivado.log -source board/nexys-4/build.tcl

set script_dir [file dirname [file normalize [info script]]]
set root [file normalize $script_dir/../..]
set build_dir $root/board/_build
file mkdir $build_dir

read_verilog $root/board/_generated/top.v
read_xdc $script_dir/nexys4.xdc
synth_design -top top -part xc7a100tcsg324-1
opt_design
# THE TWO DIRECTIVES ARE THE FLOW OF THIS PROJECT SINCE 2026-08-29, and they are here
# because the default flow was MEASURED and lost. Every era was built both ways that day,
# after the lifts into lib/nn; the default flow refused all three under the lottery rule of
# build-log.md, and Explore met in every one of them with NO clock-skew adjustment at all:
#
#   era  |  default WNS / WHS  | 32-703 |  Explore WNS / WHS  | 32-703
#   four |  +0.013 / +0.041    |   1    |  +0.143 / +0.041    |   0
#   five |  +0.030 / +0.005    |   2    |  +0.236 / +0.107    |   0
#   six  |  +0.008 / +0.017    |   3    |  +0.070 / +0.021    |   0
#
# Era five's default build failed on the HOLD side at 5 ps; era six's climbed from -0.202
# only through phys_opt, three skew adjustments among them. The bitstream in the flash was
# made by this flow, thus a build that drops these directives cannot remake it. The older
# entries of build-log.md that read "MET at default directives" are history and stay so.
place_design -directive Explore
# Physical synthesis runs after the placement and again after the route. The design stands
# at 108.5 block RAM tiles of 135 and all 240 DSPs, thus routing is most of every long path and the route can
# give back the whole slack that the placement won; the post-route pass replicates the
# drivers that the congestion stretched. Each pass costs about one second and is silent
# while the design has slack. Keep them.
phys_opt_design
route_design -directive Explore
phys_opt_design
report_timing_summary -file $build_dir/timing.rpt
report_utilization -file $build_dir/utilization.rpt
# The routed netlist, as it goes into the bitstream. A timing question then costs seconds
# in open_checkpoint and not a build: report_timing -max_paths, the congestion of
# report_design_analysis, and the paths that the summary does not name.
write_checkpoint -force $build_dir/top_routed.dcp
write_bitstream -force $build_dir/top.bit
