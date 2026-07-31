# Build log

This log keeps the build milestones. One entry for each milestone, in time
order. New entries go at the end.

## 2026-07-31 — first sound (feat/first-sound)

The board makes a note on the Roland S-1. The full chain is proven: the
board powers on into the design from the QSPI flash, the mgt tool speaks
the host control over the USB UART, the doorbell sends the test message,
and the MIDI current loop carries it to the synth. The branch merged into
develop.
