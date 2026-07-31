# Writes the bitstream into the QSPI flash. The board boots it at each power-on, with
# the JP1 jumper on QSPI.
#
# First: build the bitstream (build.tcl).
# Then, from the repository root, with the board connected and powered:
#   vivado -mode batch -journal board/_build/vivado.jou \
#          -log board/_build/vivado.log -source board/nexys-4/flash.tcl
#
# The last step boots the FPGA from the flash, thus the ritual ends in the same state
# as a power cycle.

set script_dir [file dirname [file normalize [info script]]]
set root [file normalize $script_dir/../..]
set build_dir $root/board/_build

# the Nexys 4 QSPI flash: Spansion S25FL128S, 16 MB
set flash_part s25fl128sxxxxxx0-spi-x1_x2_x4

write_cfgmem -force -format mcs -size 16 -interface SPIx4 \
    -loadbit "up 0x0 $build_dir/top.bit" $build_dir/top.mcs

open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] 0]
create_hw_cfgmem -hw_device [current_hw_device] \
    [lindex [get_cfgmem_parts $flash_part] 0]
set cfgmem [current_hw_cfgmem]
set_property PROGRAM.FILES [list $build_dir/top.mcs] $cfgmem
set_property PROGRAM.ADDRESS_RANGE use_file $cfgmem
set_property PROGRAM.ERASE 1 $cfgmem
set_property PROGRAM.CFG_PROGRAM 1 $cfgmem
set_property PROGRAM.VERIFY 1 $cfgmem
set_property PROGRAM.CHECKSUM 0 $cfgmem
create_hw_bitstream -hw_device [current_hw_device] \
    [get_property PROGRAM.HW_CFGMEM_BITFILE [current_hw_device]]
program_hw_devices [current_hw_device]
program_hw_cfgmem $cfgmem
boot_hw_device [current_hw_device]
close_hw_manager
