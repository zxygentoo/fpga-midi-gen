# Reads one unit out of context: its utilization, and its timing at 100 MHz.
#
# First: make probe-UNIT, or dune exec bin/gen_probe.exe -- UNIT [lanes]
# Then, from the repository root:
#   vivado -mode batch -journal board/_build/probe.jou \
#          -log board/_build/probe.log -source board/nexys-4/probe.tcl
#
# OUT OF CONTEXT, because no device carries a 768-bit port. The unit's own logic and
# routing are what this reads. The ports hide the store and the window on the far side,
# thus the script charges 2 ns at each boundary rather than letting those paths run free;
# the number to watch is the register-to-register one, which the report names separately.

set script_dir [file dirname [file normalize [info script]]]
set root [file normalize $script_dir/../..]
set build_dir $root/board/_build
file mkdir $build_dir

read_verilog $root/board/_generated/probe.v
synth_design -top probe -part xc7a100tcsg324-1 -mode out_of_context
create_clock -name clock -period 10.000 [get_ports clock]
set_input_delay -clock clock 2.000 \
    [get_ports -filter {DIRECTION == IN && NAME != "clock"}]
set_output_delay -clock clock 2.000 [all_outputs]
opt_design
place_design
phys_opt_design
route_design
phys_opt_design
report_utilization -file $build_dir/probe_utilization.rpt
report_timing_summary -file $build_dir/probe_timing.rpt
# the path the risk is about: the array's own registers to its own registers, with no
# boundary in it
report_timing -from [all_registers] -to [all_registers] -max_paths 10 \
    -file $build_dir/probe_internal.rpt
write_checkpoint -force $build_dir/probe_routed.dcp
