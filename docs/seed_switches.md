# Physical seed control

## Scope

The seed decides the piece. Today only the host can change it, over the
UART with `board_tool`, thus the board alone plays one piece for ever. This
document defines how the 16 slide switches set the seed and how the
seven-segment display shows it.

This design changes one rule of the host control. The section "Changes to
the host control" states it. Correct `docs/host_control.md` with the code.

## The problem

SEED has one writer and the board has none. RUN already has two — the host
and the centre button — thus the board knows this shape and `Control_regs`
already resolves it: both writers apply, and the last one wins. SEED takes
the same rule and no new one.

A physical input needs a physical readout. A person who sets a value on 16
switches and cannot see what the board holds is setting it blind, because
the switches state their own position and not the position the board took.
Therefore the display shows the SEED cell and never the switches.

## The rule

```
sw[15:0] ──▶ 2FF sync ──▶ previous ──▶ change? ──▶ write {16'b0, sw} into SEED
```

- The switches go through two flip-flops. They are asynchronous to the
  clock and a slide switch bounces.
- A register holds the value that the change detector read last time. The
  detector fires when the synchronised value and that register disagree.
- At a fire, the low two SEED cells take the switch bytes, the high two
  take zero, and the register takes the new value.

Therefore:

| Event | What happens |
|---|---|
| A switch moves | The panel wins at once, running or stopped |
| The host writes SEED | It stands until a switch moves |
| The run starts | The source reads SEED one time, as it does today |
| Switches move inside a run | SEED changes, and the run keeps the value it read |

The rule is "the last writer wins", and it is the rule RUN keeps.

### Power-on

The change detector must not fire when nothing moved, thus it needs the
value it read last time. Give that register, the two synchroniser registers
and the SEED cell **no power-on value at all**: each one is 0, which is what
a register does when nobody states otherwise. The design then takes the
switches at the power-on and spends nothing to do it:

- The switches sit at `0x1234`. The synchroniser settles on it, which
  disagrees with 0, thus the detector fires and the cell takes the switches.
  There is no first-time flag, no counter and no init walk.
- The switches sit at `0x0000`. Nothing fires and the cell keeps 0, which
  **is** the switch value. The two cases give one result.
- `btnCpuReset` clears each of them to 0 again, thus a clear and a power-on
  behave the same.

The cost of the whole rule is 32 synchroniser flip-flops, 16 registers for
the value read last, and one 16-bit comparator.

Two consequences follow. SEED loses its power-on value of 42 and its range
in `Control_intf.Reg`: the panel can set 0 and this design accepts it, thus
a range that starts at 1 would only make the table disagree with the panel.
`Control_intf.Default.seed` keeps 42, because a host tool still needs a seed
when a person does not name one — that is a default of the tools and no
longer a value of the board.

## A seed of zero

All the switches down is `0x0000`, and the RTL PRNG takes the seed with no
folding: `xorshift32(0)` is 0 for ever. Each row stays 0, each voice of the
pink model clamps to the bottom of its window and lands on its root, and the
frame never changes. Therefore a seed of zero plays **one chord and holds
it**. It is not silence.

**This design accepts it**, and the reason is that the failure is loud: a
held chord states plainly that the seed is wrong, where silence would look
like a dead board. All the switches down is the rest position of the panel,
thus the case is common and not a corner, and a person who hears one chord
knows what to do about it. The range of SEED goes from `Control_intf.Reg`
for the same reason — a table that refuses 0 while the panel sets it would
only make the two disagree.

## The width of the seed

The switches give 16 bits and the cell holds 32, thus the high half is
always zero and the panel reaches 65 535 of the 4 294 967 295 seeds.

**Measured, and 16 bits costs nothing.** The board loads the seed raw, thus
a narrow seed is a small xorshift32 state, and the transformer draws nothing
through its lead-in of one bar — its first real draw sees a state that has
barely moved. Over 64 seeds of each width, with the KING checkpoint, counted
by how long after the lead-in the first note comes:

| | seeds | sound within 2 steps | first sound after 8 | silent for 48 |
|---|---|---|---|---|
| 16-bit | 64 | 57 | 7 | 0 |
| 32-bit | 64 | 60 | 3 | 0 |

No seed of either width stays silent. The late openings are the lottery that
era four already records — about one piece in ten opens thin — and 7 against
3 is well inside the noise of 64 samples. The pink model has no question
here at all: its rewind draws eight rows before the first step, thus the
state is warm.

## The display

The display shows the whole SEED cell in hexadecimal, on all eight digits.

**Eight and not four.** The panel writes zero into the high half, thus four
digits look sufficient. They are not: the host writes the same cell and the
cell is 32 bits, thus a host can put a value in the high half and a display
of four digits would show the low half alone and call it the seed. A readout
that can misreport the value it exists to report is worse than no readout.
Eight digits cost one bit of the scan index.

### Why the display is a scan

Eight digits of seven segments, with a decimal point for each one, is 64
lamps. To light each lamp on its own needs 64 pins, and the board does not
spend them. **The digits share their segment lines.** Each segment letter is
one wire to all eight digits, and each digit has one anode of its own:

```
          seg[0]=a  seg[1]=b  seg[2]=c   ...   seg[6]=g     7 shared wires
             │         │         │               │
  an[0] ─────┼─────────┼─────────┼───────────────┤   digit 0, at the right
  an[1] ─────┼─────────┼─────────┼───────────────┤   digit 1
  an[2] ─────┼─────────┼─────────┼───────────────┤   digit 2
    ⋮        ⋮         ⋮         ⋮               ⋮
  an[7] ─────┴─────────┴─────────┴───────────────┘   digit 7, at the left
```

That is 7 + 8 = 15 pins and not 64. Both sides are active low on this board:
a 0 on `an[k]` selects digit k, and a 0 on `seg[j]` lights segment j.

**Therefore the segments show one pattern at a time**, and every digit whose
anode is on shows that same pattern. Four different digits cannot stand on
the display together, thus a static display of a number is not possible.

The block lights one digit at a time and moves along them. The eye holds an
image longer than a digit is dark, thus it joins them into one number. Two
bounds hold the rate:

- **Too slow makes a flicker.** The eye stops seeing it above about 60 Hz.
- **Too fast makes a ghost.** The anode transistors and the lamps need time
  to go off. A change that comes before they do leaves the light of the
  digit before on the digit after, with the pattern of the digit before, and
  the number looks smeared.

The Digilent reference asks for 1 ms to 16 ms over the whole display. This
design takes 5.24 ms, which is in the middle of that range:

- The digit index is counter bits 18 to 16. One digit is lit for 2^16
  cycles, which is 655 µs at 100 MHz.
- Eight digits give a full scan of 5.24 ms, which is 191 Hz.

```
an[0] ──┐     ┌───────────────────────────────────────────┐     ┌──
        └─────┘ 655 µs                                    └─────┘
an[1] ────────┐     ┌─────────────────────────────────────────────
              └─────┘
an[2] ──────────────┐     ┌───────────────────────────────────────
                    └─────┘
seg   ══╳══F══╳══2══╳══A══╳══1══╳══0══╳══0══╳══0══╳══0══╳══F══╳══
        digit0 digit1 digit2 digit3   the high half, which is zero
        └──────────────── 5.24 ms, and again ─────────────┘
```

**The price is the brightness.** One digit of eight is lit one eighth of the
time, thus the display is dimmer than a lamp that stands on. Four digits
would be two times as bright, and the paragraph above states why this design
takes eight anyway.

**The RTL is smaller than its explanation**, and it holds no state machine.
The slice of the counter does the sequence, thus there is no divisor and no
state to get wrong:

```
index   = counter[18:16]                  the digit, and it moves each 655 µs
nibble  = seed[4 * index + 3 : 4 * index] an 8 to 1 mux over the four-bit slices
segment = decode nibble                   a table of the sixteen digits, 0 to F
anode   = ~(1 << index)                   one hot, and inverted for the active low
```

`decode` is a named function and not a module: it is one table.

The block reads the SEED cell and nothing else, thus it takes no part in the
seed rule and a fault in it cannot change a note.

## Changes to the host control

One rule of `docs/host_control.md` changes, and it is the rule that RUN
already states:

> SEED holds the seed of the next run. A host write sets it, and a move of
> any slide switch sets it to the switch value in the low 16 bits with the
> high 16 bits at zero. A read returns the current value. Therefore SEED can
> change without a host write, and a host write stands only until a switch
> moves.

The Default column of the SEED row takes "the switches" and its range goes,
because a driver can never read 42 again: the panel writes three cycles
after the power-on and the fastest transaction is thousands of cycles.

The seed rule itself does not change: the model reads SEED at the run start
and one run plays one sequence.

## The pins

The XDC has no switch pin and no display pin today — it declares `clk`,
`btnCpuReset`, `btnC`, `RsRx`, `RsTx`, `led[15:0]` and `JD[7:0]` and nothing
else. This design adds `sw[15:0]`, `seg[6:0]`, `dp` and `an[7:0]`, which is
16 inputs and 16 outputs. `dp` takes a constant 1 from the top level and
not from the block; the interface section states why.

The pins come from `Nexys-4-Master.xdc` of `Digilent/digilent-xdc`, where
each line is commented out and a design takes the lines it uses. **It is not
the file of the Nexys 4 DDR**, which has a different pinout: the wrong file
gives a design that builds, programs and does nothing. The LED pins already
in our XDC — T8, V9, R8, T6 and the rest — are the pins of this file and not
of the DDR one, thus the board and the file agree.

Every pin below is `IOSTANDARD LVCMOS33`.

| Signal | Pins, from bit 0 upward |
|---|---|
| `sw[15:0]` | U9 U8 R7 R6 R5 V7 V6 V5 U4 V2 U2 T3 T1 R3 P3 P4 |
| `seg[6:0]` | L3 N1 L5 L4 K3 M2 L6 (segments a to g) |
| `dp` | M4 |
| `an[7:0]` | N6 M6 M3 N5 N2 N4 L1 M1 (`an[0]` is the digit at the right) |

The switch pins need no drive and no slew property. They are inputs. The
display pins drive LEDs through the transistors of the board, thus they take
the board default and none of the care that `JD[0]` takes for the current
loop.

## The blocks

```
                 ┌──────────────────────────┐
  sw[15:0] ─────▶│      Seed_switches       │──▶ seed_write ─┐
                 │  sync · Δ · scan · decode │──▶ seed_value ─┤
              ┌─▶│                          │──▶ an, seg, dp │
              │  └──────────────────────────┘                │
              │                                              ▼
              │                                   ┌──────────────────┐
              └────────── params.seed ────────────│   Control_regs   │
                                                  └──────────────────┘
```

| Block | It owns |
|---|---|
| `lib/board/seed_switches.ml` | the synchroniser, the value read last, the change strobe, the scan counter, the digit mux and the nibble decode |
| `Control_regs` | one more writer of SEED, beside the host and beside the button of RUN |

**One block and not two.** The panel and the display are one feature: the
switches state a seed and the digits state the seed the board holds. The
display has one producer and one sink and it will never have another — the
section "What this design does not do" states why nothing else goes on it —
thus a second module would be a name and not a purpose. This is the rule
that put `Midi_out` inside `Socket`. The nibble decode is a named function
inside the block and not a module of its own.

The block gives a strobe and a value; it holds no seed of its own, thus the
cell stays the one home of the value. `Control_regs` takes them as it takes
`run_toggle` today. The path is a loop on the page and not in the circuit:
`seed_value` comes from the synchroniser and `params.seed` comes from the
cells, thus each end is a register.

### The interface

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; switches : 'a [@bits 16]
    (** the slide switches, straight from the pins and asynchronous to the
        clock *)
    ; seed : 'a [@bits 32]
    (** the live view of the SEED cell: what the display shows, and not what
        the switches say *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { seed_write : 'a
    (** a strobe: the synchronised switches moved, thus [seed_value] must go
        into the cell *)
    ; seed_value : 'a [@bits 32]
    (** the seed the panel states: the switches in the low 16 bits and zero
        above them *)
    ; anode : 'a [@bits 8] (** the digit that is lit; active low *)
    ; segment : 'a [@bits 7] (** the segments a to g of that digit; active low *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
```

**The decimal point is not here.** It is a shared cathode like the seven
segments, thus it is a pin that this design must drive, and the value is 1
for ever: the convention is active low, thus a 0 lights it and an undriven
pin is not a defined level. But a constant is not an output of a block — it
is a pin of a group that this design does not use, and the top level ties it
off. `JD[7:1]` set the precedent: `Socket` gives the MIDI line alone, and
`top.ml` writes `concat_msb [ ones 7; serial ]`. The top level writes `dp`
at 1 beside `an` and `seg` in the same way.

The block would own `dp` if it ever lit it. The section "What this design
does not do" closes that door: nothing but the seed goes on the display.

`seed_value` is 32 bits and not 16, because `Control_regs` must not know how
wide the panel is. The panel states a seed; that its high half is zero today
is a fact of this board and not of the cell.

`seed` is the whole cell and not its low half, for the same reason in the
other direction: the block decides what the display shows, thus the top
level slices nothing and a host write of a wide seed reaches the digits.

`create` takes no parameter. The scan period is a slice of a counter and the
clock is the board clock, thus there is no divisor to pass and no
elaboration constant to get wrong.

## What this design does not do

- It does not give the panel the high 16 bits of the seed. The board has 16
  switches.
- It does not refuse a seed of zero. The section above states why.
- It does not show anything but the seed. A display that shows a step count
  or a state is a diagnostic, and the host answers a diagnostic better and
  answers it with a reason.

## The steps

1. `Seed_switches`, the seed half, with tests: the synchroniser, a move of
   one switch, a move while a run stands, and the power-on case in which the
   switches read zero and nothing fires.
2. `Seed_switches`, the display half, with tests: the nibble decode against
   a table of the sixteen digits, and a waveform of one full scan that shows
   each anode low in its turn.
3. `Control_regs`: the second writer of SEED, and the power-on value of the
   SEED cells goes to 0 with the range of the row. Two tests move with it —
   a host burst and a switch move in one cycle must both apply, as the RUN
   test states it, and the expect test of the defaults reads
   `00 00 00 00 64 c8 00 02 00` in the place of `2a 00 00 00 ...`.
4. The XDC: the switch and display pins, from the master file of the correct
   board.
5. `top.ml`: seat the block, and tie `dp` at 1 beside `an` and `seg`.
6. Correct `docs/host_control.md`, the SEED row and its Default column.

   **`test/test_txn.ml` changes with it, and the change is the proof.** It
   reads the whole section through the board top level and expects
   `2a 00 00 00` in the seed bytes today. With the panel seated it must read
   the switches that the test drives, thus the test states the new rule at
   the top level: hold the switches at a value, and the cell answers with
   it.
7. Build, program, and check on the board: the display follows the switches
   at power-on, a host write shows on the display, and a switch move takes
   it back.
