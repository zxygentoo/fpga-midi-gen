# Programs the bitstream into the board through the USB JTAG.
#
# From the repository root, with the board connected and powered:
#   vivado -mode batch -journal board/_build/vivado.jou \
#          -log board/_build/vivado.log -source board/nexys-4/program.tcl

set script_dir [file dirname [file normalize [info script]]]
set root [file normalize $script_dir/../..]

open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE $root/board/_build/top.bit [current_hw_device]
program_hw_devices [current_hw_device]
close_hw_manager
