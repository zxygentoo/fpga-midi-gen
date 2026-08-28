# Builds the bitstream from the generated Verilog.
#
# First: dune exec board/nexys-4/gen_verilog.exe
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
place_design
# Physical synthesis runs after the placement and again after the route. The design stands
# at 108.5 block RAM tiles of 135 and all 240 DSPs, thus routing is most of every long path and the route can
# give back the whole slack that the placement won; the post-route pass replicates the
# drivers that the congestion stretched. Each pass costs about one second and is silent
# while the design has slack. Keep them.
phys_opt_design
route_design
phys_opt_design
report_timing_summary -file $build_dir/timing.rpt
report_utilization -file $build_dir/utilization.rpt
# The routed netlist, as it goes into the bitstream. A timing question then costs seconds
# in open_checkpoint and not a build: report_timing -max_paths, the congestion of
# report_design_analysis, and the paths that the summary does not name.
write_checkpoint -force $build_dir/top_routed.dcp
write_bitstream -force $build_dir/top.bit
