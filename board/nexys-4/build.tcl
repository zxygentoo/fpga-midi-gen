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
phys_opt_design
route_design
# The design needs this pass and does not merely gain from it. At six layers it stands at
# 126 block RAM tiles of 135, thus routing is three quarters of every long path, and the
# route gives back the whole slack that placement won: measured at -0.252 ns here and
# +0.031 ns after this pass. Post-route physical synthesis replicates the drivers that the
# congestion stretched. Remove it and the bitstream misses the period.
phys_opt_design
report_timing_summary -file $build_dir/timing.rpt
report_utilization -file $build_dir/utilization.rpt
write_bitstream -force $build_dir/top.bit
