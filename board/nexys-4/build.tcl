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
route_design
report_timing_summary -file $build_dir/timing.rpt
report_utilization -file $build_dir/utilization.rpt
write_bitstream -force $build_dir/top.bit
