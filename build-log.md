# Build log

This log keeps the build milestones. One entry for each milestone, in time
order. New entries go at the end.

## 2026-07-31 — first sound (feat/first-sound)

The board makes a note on the Roland S-1. The full chain is proven: the
board powers on into the design from the QSPI flash, the mgt tool speaks
the host control over the USB UART, the doorbell sends the test message,
and the MIDI current loop carries it to the synth. The branch merged into
develop.

## 2026-07-31 — the control redesign (feat/regfile-redesign)

The control section was a byte memory with one address port. A memory of
that shape permits one reader at a time, thus the doorbell kept its own
copy of the message cells, the write decode was written two times, and the
MIDI sender had to live inside the wire-protocol engine. The model needs to
read the same cells and to send its own MIDI, therefore the shape had to
change before the model era starts.

The design is in `docs/host_control_rtl.md`. There are four blocks:
`Control_port` for the wire protocol, `Control_regs` for the cells and the
doorbell, `Midi_merge` for the priority, and `Midi_out` for the line. The
cells give named views with the natural width of each value, thus a
consumer never sees a byte and needs no address. `Regfile` is deleted.

What changed for the host:

- A write applies at one time. The port fills a shadow copy and one commit
  strobe moves the whole burst, therefore a value of more than one byte
  does not tear.
- The cells carry their power-on value in the bitstream, thus there is no
  init walk and the first request after configuration gets a correct
  answer.
- ADDR is one byte, and the registers are at `00` to `0F`. The two-byte
  address was the last part of a global memory map that this project never
  had.
- MIDI_GO reads 1 until the transmitter takes the message, and no longer
  until the last byte is on the line. `Control_regs` copies the message at
  the ring, thus a write after MIDI_GO reads 0 cannot damage it.

`test/test_txn.ml` passed with no change at each step. It drives the board
top level at the true UART rates and examines the MIDI line, thus it shows
that the behavior on the wire and on the MIDI line stays the same while
each block below it moves.

On the board: timing is met, 418 LUTs and 522 flip-flops, no block RAM.
`mgt dump` gives the same 16 bytes as the simulation, and the doorbell
sends a Note On, a Note Off and a one-byte real-time message. The bitstream
is in the QSPI flash.
