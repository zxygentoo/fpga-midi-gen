# RULES

1. ALWAYS show the diff for review before you make a commit. Make a commit
   automatically ONLY if the user gives permission.
2. Write ALL technical documents in ASD-STE100 English.

# Basic

This project makes MIDI on an FPGA. An external synthesizer plays it.

## Toolchain

OCaml for all code. Hardcaml for the RTL.

- opam switch: `5.2.0+ox` (OxCaml)
- Hardcaml version: `v0.18~preview`
- Vivado: hardware synthesis
- dune

## Hardware

- FPGA board: Digilent Nexys 4, XC7A100T-1CSG324. This is **not** the Nexys 4
  DDR.
- Synthesizer: Roland S-1.

## Connections

- Pmod JD header pin 1 -> 33 Ω -> Jack L (tip)
- Pmod JD header pin 5 -> Jack G (sleeve)
- Pmod JD header pin 6 -> 33 Ω -> Jack R (ring)
- Jack -> Cable -> S-1 MIDI IN
- Host -> Nexys 4 USB UART (`/dev/ttyUSB1`)
- Host -> S-1 USB

The Jack is a 3.5 mm TRS adapter with screw terminals. The Cable is a 3.5 mm
male-male audio cable.

Header pin 1 is the only FPGA pin here. Header pins 5 and 6 are the ground and
the 3.3 V supply of the Pmod connector, and the FPGA cannot control them. The
ring stays at 3.3 V, and it needs no RTL. This is MIDI TRS Type A.

The MIDI input of the synth is an isolated current loop with a TLP2368
photocoupler. Therefore the two USB connections cannot make a ground loop. The
loop is about 286 Ω and gives about 5.4 mA, against a worst-case threshold of
5 mA. Do not add resistance in series, because the margin is only about 24 Ω.
The 33 Ω value is correct for a 3.3 V driver, and the older 220 Ω value is for
5 V.

## Safety

**WARNING: Remove the power from the board before you connect or disconnect the
Cable.** A TRS plug makes a short circuit between tip, ring and sleeve as it
moves into the Jack. If the FPGA output is low at that moment, the current is
about 50 mA. This is more than the limit of an Artix-7 pin.

**WARNING: Examine Pmod header pins 5 and 6 before the first power-on.** Pin 5
is the ground and pin 6 is the 3.3 V supply. If the two wires are not in the
correct positions, you make a short circuit on the supply. The Nexys 4 has no
fuse for each Pmod connector.

**WARNING: Do not make a short circuit between the tip terminal and the VCC
terminal.** The current is about 100 mA.

# Board reference

The MIDI data pin is header pin 1. In the XDC this pin is `JD[0]`, and the
package pin is H4. The header pin number and the XDC index are not the same.

| Signal | Package pin |
|---|---|
| `clk` | E3 (100 MHz) |
| `btnCpuReset` | C12 |
| `RsRx` (host to FPGA) | C4 |
| `RsTx` (FPGA to host) | D4 |
| `JD[0]` to `JD[7]` | H4, H1, G1, G3, H2, G4, G2, F3 |

These properties are necessary:

- `DRIVE 8` on `JD[0]`. The pin must sink 5.4 mA with a low output voltage,
  because that voltage subtracts from the loop current.
- `DRIVE 4` on the other JD pins. They have no connection. A low drive limits
  the current if the wires are not in the correct positions.
- `PULLTYPE NONE` and `SLEW SLOW` on `JD[*]`.

MIDI is 31250 baud, 8N1. The divisor from 100 MHz is exactly 3200, and the
clock error of this hardware is −279 ppm. The console UART to the host is
115200 baud, 8N1. The maximum MIDI byte rate is 3125 bytes each second.

# The synthesizer

- The default receive channel is **3**, not 1. Roland uses different channels
  so that you can connect more than one AIRA Compact unit.
- If there is no sound but all counters are correct, examine the channel first.
- The synth receives Program Change on channel **16**.
- Clock, Start and Stop have no channel. Therefore the synth can play its own
  sequencer from your clock and ignore all your notes. Sound does not show that
  the channel is correct. To test the channel, send a note.
- To change the channel: SHIFT + PAD 15 (MENU), dial to `CH`, PAD 2 (ENTER),
  dial to the value, PAD 1 (EXIT).
- CC 74 is the filter cutoff. The synth does not keep the value after a power
  cycle, and the CUTOFF knob has priority over MIDI. Use CC 102 for integrity
  tests, because it has no audible effect.
- A control change is only audible on the next note.
- The S-1 has four voices.

## USB audio

The S-1 is a class-compliant USB audio device. Therefore you can record its
output with no loss of quality. The format is S32_LE, 44100 Hz, 2 channels. It is not 48000 Hz.

- The firmware must be version 1.02 or later. Push STEP during power-on to see
  the version.
- AIRA LINK must be OFF: SHIFT + pad 15 -> `A.Lnk` -> `OFF`, then power-cycle.
- Do not use a USB hub.

# Layout

- `bin/` — executables: drivers, model trainer and other tools.
- `board/` — RTL, configuration and scripts. Each board has a directory, for
  example `board/nexys-4`.
- `board/_generated/` — Verilog from the Hardcaml top level. Git ignores it.
- `board/_build/` — the Vivado work directory. Git ignores it.
- `lib/` — the core library and the RTL.

# Design

```
Model (RTL/Hardcaml) -- ABI -- Drivers (OCaml)
```

- Model: the FPGA does the inference. Train the model on the host computer if
  it is necessary. Possible models are a Markov chain, an RNN and a UNet.
- ABI: one interface for all drivers. Every model gives MIDI as its output, so
  the interface is easy to keep stable. A control message is a SysEx message
  with manufacturer ID 7D. MIDI reserves this ID for non-commercial use. All
  other data goes to the MIDI transmitter with no change.
- Drivers: self-check, inference control and other functions.

Keep the model `config` (for example the transition matrix, the tempo and the
channel) in a memory that the host writes at run time. Then a change needs only
a serial write and not a rebuild. The map of this memory is part of the ABI.
Define the map one time in `lib/`, and use it in both the RTL and the drivers.

# Traps

- A harmonic product spectrum can find a sub-harmonic and report a note one
  octave too low. Limit the search to the range of the scale.

# Gitflow

This repository uses git-flow, with the branches `main` and `develop`, and the
prefix `feat/` for features.
