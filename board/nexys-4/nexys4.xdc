# Nexys 4 (not the DDR version), XC7A100T-1CSG324.
# The pins agree with AGENT.md and with the Digilent Nexys-4-Master.xdc.

# Configuration
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# QSPI boot: the flash is quad-wide, 33 MHz is a safe config clock for the S25FL128S,
# and the compression makes both the flash image and the boot much smaller.
set_property CONFIG_MODE SPIx4 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

# Clock: 100 MHz
set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -period 10.000 -name clk [get_ports clk]

# Reset button, active low
set_property -dict { PACKAGE_PIN C12 IOSTANDARD LVCMOS33 } [get_ports btnCpuReset]

# Center button: toggles the run state
set_property -dict { PACKAGE_PIN E16 IOSTANDARD LVCMOS33 } [get_ports btnC]

# USB-UART
set_property -dict { PACKAGE_PIN C4 IOSTANDARD LVCMOS33 } [get_ports RsRx]
set_property -dict { PACKAGE_PIN D4 IOSTANDARD LVCMOS33 } [get_ports RsTx]

# LEDs
set_property -dict { PACKAGE_PIN T8 IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN V9 IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN R8 IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN T6 IOSTANDARD LVCMOS33 } [get_ports {led[3]}]
set_property -dict { PACKAGE_PIN T5 IOSTANDARD LVCMOS33 } [get_ports {led[4]}]
set_property -dict { PACKAGE_PIN T4 IOSTANDARD LVCMOS33 } [get_ports {led[5]}]
set_property -dict { PACKAGE_PIN U7 IOSTANDARD LVCMOS33 } [get_ports {led[6]}]
set_property -dict { PACKAGE_PIN U6 IOSTANDARD LVCMOS33 } [get_ports {led[7]}]
set_property -dict { PACKAGE_PIN V4 IOSTANDARD LVCMOS33 } [get_ports {led[8]}]
set_property -dict { PACKAGE_PIN U3 IOSTANDARD LVCMOS33 } [get_ports {led[9]}]
set_property -dict { PACKAGE_PIN V1 IOSTANDARD LVCMOS33 } [get_ports {led[10]}]
set_property -dict { PACKAGE_PIN R1 IOSTANDARD LVCMOS33 } [get_ports {led[11]}]
set_property -dict { PACKAGE_PIN P5 IOSTANDARD LVCMOS33 } [get_ports {led[12]}]
set_property -dict { PACKAGE_PIN U1 IOSTANDARD LVCMOS33 } [get_ports {led[13]}]
set_property -dict { PACKAGE_PIN R2 IOSTANDARD LVCMOS33 } [get_ports {led[14]}]
set_property -dict { PACKAGE_PIN P2 IOSTANDARD LVCMOS33 } [get_ports {led[15]}]

# Slide switches: they write SEED. They are inputs, thus they need no drive
# and no slew property.
set_property -dict { PACKAGE_PIN U9 IOSTANDARD LVCMOS33 } [get_ports {sw[0]}]
set_property -dict { PACKAGE_PIN U8 IOSTANDARD LVCMOS33 } [get_ports {sw[1]}]
set_property -dict { PACKAGE_PIN R7 IOSTANDARD LVCMOS33 } [get_ports {sw[2]}]
set_property -dict { PACKAGE_PIN R6 IOSTANDARD LVCMOS33 } [get_ports {sw[3]}]
set_property -dict { PACKAGE_PIN R5 IOSTANDARD LVCMOS33 } [get_ports {sw[4]}]
set_property -dict { PACKAGE_PIN V7 IOSTANDARD LVCMOS33 } [get_ports {sw[5]}]
set_property -dict { PACKAGE_PIN V6 IOSTANDARD LVCMOS33 } [get_ports {sw[6]}]
set_property -dict { PACKAGE_PIN V5 IOSTANDARD LVCMOS33 } [get_ports {sw[7]}]
set_property -dict { PACKAGE_PIN U4 IOSTANDARD LVCMOS33 } [get_ports {sw[8]}]
set_property -dict { PACKAGE_PIN V2 IOSTANDARD LVCMOS33 } [get_ports {sw[9]}]
set_property -dict { PACKAGE_PIN U2 IOSTANDARD LVCMOS33 } [get_ports {sw[10]}]
set_property -dict { PACKAGE_PIN T3 IOSTANDARD LVCMOS33 } [get_ports {sw[11]}]
set_property -dict { PACKAGE_PIN T1 IOSTANDARD LVCMOS33 } [get_ports {sw[12]}]
set_property -dict { PACKAGE_PIN R3 IOSTANDARD LVCMOS33 } [get_ports {sw[13]}]
set_property -dict { PACKAGE_PIN P3 IOSTANDARD LVCMOS33 } [get_ports {sw[14]}]
set_property -dict { PACKAGE_PIN P4 IOSTANDARD LVCMOS33 } [get_ports {sw[15]}]

# Seven-segment display: it shows SEED in hexadecimal. The segments a to g are
# shared by the eight digits and each digit has one anode; both sides are
# active low. These pins drive LEDs through the transistors of the board, thus
# they take the board default and none of the care that JD[0] takes.
set_property -dict { PACKAGE_PIN L3 IOSTANDARD LVCMOS33 } [get_ports {seg[0]}]
set_property -dict { PACKAGE_PIN N1 IOSTANDARD LVCMOS33 } [get_ports {seg[1]}]
set_property -dict { PACKAGE_PIN L5 IOSTANDARD LVCMOS33 } [get_ports {seg[2]}]
set_property -dict { PACKAGE_PIN L4 IOSTANDARD LVCMOS33 } [get_ports {seg[3]}]
set_property -dict { PACKAGE_PIN K3 IOSTANDARD LVCMOS33 } [get_ports {seg[4]}]
set_property -dict { PACKAGE_PIN M2 IOSTANDARD LVCMOS33 } [get_ports {seg[5]}]
set_property -dict { PACKAGE_PIN L6 IOSTANDARD LVCMOS33 } [get_ports {seg[6]}]
set_property -dict { PACKAGE_PIN M4 IOSTANDARD LVCMOS33 } [get_ports dp]
set_property -dict { PACKAGE_PIN N6 IOSTANDARD LVCMOS33 } [get_ports {an[0]}]
set_property -dict { PACKAGE_PIN M6 IOSTANDARD LVCMOS33 } [get_ports {an[1]}]
set_property -dict { PACKAGE_PIN M3 IOSTANDARD LVCMOS33 } [get_ports {an[2]}]
set_property -dict { PACKAGE_PIN N5 IOSTANDARD LVCMOS33 } [get_ports {an[3]}]
set_property -dict { PACKAGE_PIN N2 IOSTANDARD LVCMOS33 } [get_ports {an[4]}]
set_property -dict { PACKAGE_PIN N4 IOSTANDARD LVCMOS33 } [get_ports {an[5]}]
set_property -dict { PACKAGE_PIN L1 IOSTANDARD LVCMOS33 } [get_ports {an[6]}]
set_property -dict { PACKAGE_PIN M1 IOSTANDARD LVCMOS33 } [get_ports {an[7]}]

# Pmod JD. JD[0] is the MIDI output: DRIVE 8, because the pin must sink
# 5.4 mA with a low output voltage. The other pins have no connection:
# DRIVE 4 limits the current if the wires are not in the correct positions.
set_property -dict { PACKAGE_PIN H4 IOSTANDARD LVCMOS33 DRIVE 8 SLEW SLOW PULLTYPE NONE } [get_ports {JD[0]}]
set_property -dict { PACKAGE_PIN H1 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW PULLTYPE NONE } [get_ports {JD[1]}]
set_property -dict { PACKAGE_PIN G1 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW PULLTYPE NONE } [get_ports {JD[2]}]
set_property -dict { PACKAGE_PIN G3 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW PULLTYPE NONE } [get_ports {JD[3]}]
set_property -dict { PACKAGE_PIN H2 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW PULLTYPE NONE } [get_ports {JD[4]}]
set_property -dict { PACKAGE_PIN G4 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW PULLTYPE NONE } [get_ports {JD[5]}]
set_property -dict { PACKAGE_PIN G2 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW PULLTYPE NONE } [get_ports {JD[6]}]
set_property -dict { PACKAGE_PIN F3 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW PULLTYPE NONE } [get_ports {JD[7]}]
