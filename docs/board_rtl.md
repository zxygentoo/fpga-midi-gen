# The board in RTL

## Scope

This document states the whole system at one level: the parts, what each part
owns, and where the design of each part is. The detail is in the documents
that the last section names.

The picture holds for every era, because the model is a seat and not a part.
A model core drops into the socket and states one frame for each step, and
nothing else in the picture knows which core it is.

![The blocks of the board: the pins, the control path, the seed panel, the
source socket and the MIDI path](board_rtl.svg)

## The system

```
Model (RTL/Hardcaml) ── host control ── Drivers (OCaml)
```

The FPGA makes the music. The host does not play and it cannot put a byte on
the MIDI line: it sets the parameters, it starts a run, and the run states the
music. A driver in `bin/` speaks the host control over the USB UART, and
`AGENT.md` states the wiring of the board, the current loop and the safety
rules.

The board needs no host at all. One push of the center button plays, and the
slide switches state the seed.

## The parts

| Part | It owns |
|---|---|
| `Control_port` | the wire protocol: the COBS decode, the frame buffer, the header checks and the response |
| `Control_regs` | the control cells: the storage, the write decode and the named views |
| `Button` | the center button: two synchroniser flip-flops, a debounce of 10 ms, and a toggle of RUN |
| `Seed_switches` | the slide switches, which write SEED, and the eight digits, which show the cell |
| `Socket` | the model seat: a source, the sequencer that drives it, and the line that carries its messages |

One clock of 100 MHz, and one clear which is `¬btnCpuReset`. Both go to every
block that holds state, thus the board has no clock domain crossing except the
inputs that a person moves, and each of those has two synchroniser flip-flops.

## The socket

**A source states frames, not notes.** One step of music is one frame: the
four voice codes in one 32-bit word, seat 0 in the low byte. A frame states
which voice holds which pitch and nothing else, thus the socket carries no
note, no seat and no release.

**The handshake is `rewind`, `step` and `idle`.** `rewind` goes to the origin
of the sequence, which is the run start. `step` asks for one step, `valid`
answers it one time, and `frame` holds that step. `idle` is 1 when the source
is at rest, and the sequencer strobes only then. A late `valid` only makes the
sequencer wait, thus the socket is latency-insensitive and that is a rule of
correctness and not of timing.

**No configuration crosses the socket.** A source takes what it needs by
closure at elaboration — the seed view, a weights ROM, whatever it is —
therefore the seed goes into the slot and not across the socket.

**The sequencer decodes; it composes nothing.** It holds the set of pitches
that sound, and the frame states the set that must sound: it sends the
releases first, which is the first set less the second, and then the strikes.
Each held pitch keeps the channel of its Note On, thus a CHANNEL write inside
a run cannot hang a note. The run stop is a silent frame, thus the stop needs
no rule of its own.

## What does not change when the model changes

- **The pins and the rates.** The host UART takes 868 clocks for a bit at
  115200 baud, and the MIDI line takes 3200 at 31250, which is exact. The MIDI
  line idles at 1, the no-current level of the loop.
- **The cells are correct at cycle 0.** Each cell carries its power-on value in
  the bitstream and a clear gives the same value, thus the section needs no
  init walk. SEED is the one cell whose stored value no driver can read,
  because the panel writes the switches into it three cycles after the
  power-on.
- **A write does not tear.** The port fills a shadow copy one byte in each
  cycle and one commit strobe moves the whole burst into the live cells.
- **Two cells have a writer on the board**: the button toggles RUN and the
  switches write SEED. Each one takes the rule that the last writer wins.
- **The model is the only source of MIDI**, thus the transmitter stays inside
  the seat and the top level wires no handshake for the MIDI path.
- **The diagnostics are on the board**: `led0` states the run state, `led1`
  states MIDI activity, and the digits state the seed. The two lamps are not
  one lamp, because the model is silent through the lead-in of one bar, thus
  `led0` answers the push while `led1` is still dark.

## Where the design of each part is

| Document | What it states |
|---|---|
| `docs/host_control.md` | the host control as the host sees it: the cells, the wire protocol and the status codes |
| `docs/host_control_rtl.md` | how the FPGA supplies it: `Control_port`, `Control_regs` and the message interface |
| `docs/seed_switches_rtl.md` | the seed panel: the switches, the change detector, the scan and the digits |
| `docs/transformer.md` | the model of era four: the step frame, the decode, the corpus, the model itself and what the era measured |
| `docs/transformer_rtl.md` | that model as a circuit: the engine, the block RAM budget and the timing |
| `docs/pink.md` | the model of era one: the 1/f walk, the register decomposition and the mapping |
| `docs/pink_rtl.md` | that model as a circuit, and it shares the socket |
| `build-log.md` | the milestones in time order, and what each one measured |

The cost, the timing and the block RAM of a source belong to that source, thus
this document carries none of those numbers: they move with the model and this
picture does not.
