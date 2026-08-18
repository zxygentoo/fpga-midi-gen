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
# The pass was load-bearing, and it is now insurance. At six layers the design stands at
# 126 block RAM tiles of 135, thus routing is three quarters of every long path, and the
# route gave back the whole slack that placement won: measured at -0.233 ns here and
# +0.059 ns after this pass (2026-08-19, with the registered ROM address; -0.252 and
# +0.031 before it). Post-route physical synthesis replicates the drivers that the
# congestion stretched.
#
# The board simplification took the design out of that regime. Both passes now find no
# setup violation and change no netlist: +0.046 ns after the placement and +0.113 ns after
# the route, with 262 endpoints fewer (2026-08-19). Keep them. Each one costs about one
# second, each one is silent while the design has slack, and the margin here is one
# Vivado run wide.
phys_opt_design
report_timing_summary -file $build_dir/timing.rpt
report_utilization -file $build_dir/utilization.rpt
write_bitstream -force $build_dir/top.bit
