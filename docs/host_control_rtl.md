# The host control in RTL

## Scope

`docs/host_control.md` defines what the host sees. This document defines how
the FPGA supplies it: the blocks, what each block owns, and the interface of
each block.

This design changes three rules of the host control. The section "Changes to
the host control" lists them. Correct that document with the code.

**Era four removed the doorbell.** The host cannot put a byte on the MIDI
line: the model is the only source, and one button push plays it. Therefore
`Midi_merge` went away, `Midi_out` moved into the model seat, and the cells
MIDI_MSG, MIDI_LEN and MIDI_GO went away with them. The sections below state
the board as it is. The sections "The problem", "The cost", "What this
design does not do", "Changes to the host control" and "The steps" are the
record of the era-one redesign: they name the doorbell, because it was
there, and their numbers are the numbers of that build.

## The problem

The wire protocol gives an address to each control cell, thus the first
design put the cells in `Regfile`, a byte memory with one address port.
This is the wrong shape.

A control cell is not a memory location. Many blocks look at a control cell
continuously, and one block writes it. A memory with one address port
permits one reader at a time. The response path holds that address port for
the full length of a read. Therefore no other block can look at a cell.

The consequences are in `Control_port` today:

- The doorbell keeps its own copy of the MIDI_MSG and MIDI_LEN cells,
  because the sender must look at them while the response path uses the
  address port.
- The write decode is written two times: one time for the register file,
  and one time for the copy.
- The read path has a special case that sends the doorbell cells to the
  copy and the other cells to the register file.
- Cells 0 to 4 of the register file take flip-flops, and no block reads
  them.
- The MIDI sender is inside `Control_port`, because only that block can see
  the cells. `Control_port` is the wire-protocol engine, and a MIDI sender
  does not belong in it.

The next two requirements make this worse. The board button is a second
writer of RUN. The model is a second reader of the parameter cells and a
second source of MIDI messages.

## The blocks

```
             ┌──────────────┐   write, commit    ┌──────────────┐
  RsRx ─────▶│              │───────────────────▶│              │
             │ Control_port │   read_address     │ Control_regs │◀── run_toggle
  RsTx ◀─────│  (protocol)  │───────────────────▶│   (state)    │
             │              │◀── read_data ──────│              │
             └──────────────┘                    └──────┬───────┘
                                                        │
                                                 params │
                                                        ▼
                                             ┌────────────────────┐
                                             │       Socket       │
                                             │ ┌────────┐         │
                                             │ │ Source │◀──▶ Seq │
                                             │ └────────┘      │  │
                                             │            Midi_out│──▶ JD[0]
                                             └────────────────────┘
```

| Block | It owns |
|---|---|
| `Control_port` | the wire protocol: the COBS decode, the frame buffer, the header checks, and the response |
| `Control_regs` | the control cells: the storage, the write decode, the named views, and the value that each cell reads as |
| `Socket` | the model seat: a source, the sequencer that drives it, and the line that carries its messages |
| `Midi_out` | one message to the MIDI line, and the transmitter |

`Regfile` goes away. No other block needs it, because a control cell is not
a memory location.

`cell_bits` is `address_bits_for Control_intf.Reg.size`, which is 4. A cell
address is an index into the control section, and not a full address.
`Control_port` owns the address map, because it must give STATUS `02` for
an address outside the range.

## The message interface

Three blocks send or take a MIDI message. They use one interface, in
`lib/core/midi.ml`:

```ocaml
module Message = struct
  type 'a t =
    { data : 'a [@bits 24]
    (** the message bytes; the first byte is in the low 8 bits *)
    ; len : 'a [@bits 8] (** the number of bytes, 1 to 3 *)
    ; valid : 'a (** the sink takes the message when its [ready] is also 1 *)
    }
  [@@deriving hardcaml]
end
```

A source holds `data` and `len` while `valid` is 1. The transfer is the one
cycle in which `valid` and the `ready` of the sink are both 1. After that
cycle the source is free. When `valid` is 0, `data` and `len` have no
meaning, and a waveform can show any value in them.

The simulation names the ports of a nested message `message$data`,
`midi$valid`, and so on.

A message and not a byte stream: therefore a source cannot put its bytes
between the bytes of another source. One source states one whole message,
and that is the shape of the interface.

The limit of 3 bytes covers each channel voice message and each real-time
message. A System Exclusive message does not fit. A later design that needs
one adds a byte-stream source beside this interface.

`Midi` is also the home of the MIDI status bytes that the model needs, and
it makes the bytes of a channel voice message. Therefore no other block
holds the byte layout.

## Control_regs

### The interface

```ocaml
module Params = struct
  type 'a t =
    { run : 'a [@bits 1] (** the run state *)
    ; channel : 'a [@bits 4] (** the MIDI channel of the model, 0 to 15 *)
    ; step_ms : 'a [@bits 16] (** the step period in ms *)
    ; velocity : 'a [@bits 8] (** the note velocity *)
    ; seed : 'a [@bits 32] (** the PRNG seed *)
    }
  [@@deriving hardcaml]
end

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; write_enable : 'a (** writes [write_data] into the shadow cell *)
    ; write_address : 'a [@bits cell_bits] (** the cell index of the write *)
    ; write_data : 'a [@bits 8] (** the byte to write *)
    ; commit : 'a (** a strobe: copy the shadow into the live cells *)
    ; read_address : 'a [@bits cell_bits]
    (** the cell index that [read_data] answers *)
    ; run_toggle : 'a
    (** a strobe from the board button: invert bit 0 of RUN *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { params : 'a Params.t (** the named views; each one is stable *)
    ; read_data : 'a [@bits 8] (** the byte at [read_address] *)
    }
  [@@deriving hardcaml]
end
```

The storage is 9 8-bit cells, and the write decode is uniform. The output
is a set of named views with the natural width of each value.

A consumer reads a view. A consumer never sees a byte, and no consumer
needs an address. The uniform byte array keeps the write decode simple, and
the views keep the consumers simple.

`Params` holds the parameters of the model, and each cell is in it: the
section carries the model state and nothing else.

`Params` is a nested interface. Thus the simulation names its ports
`params$run`, `params$step_ms`, and so on. Only the top level becomes a
Verilog module, therefore the name separator has no effect on the hardware.
Give an internal name with `--` to each signal that a waveform test shows.

`run_toggle` has no counterpart in this design, because the board button
comes later. The input stays, because its behavior is settled: it costs one
bit and no decision, and the top level ties it to 0.

The strobe that tells a PRNG to load the seed is not here. Its behavior is
not settled — a load at each write and a load at the start of a run are both
reasonable — and it comes with the model that consumes it.

### The defaults

Each live cell register takes `~initialize_to` and `~clear_to` with its
default value. Therefore the cells are correct at power-on and after a
clear, and the design has no `Init` state, no init walk, and no defaults
mux. The first request needs no wait.

The Verilog keeps the power-on value, as `reg [7:0] cell = 8'b...;`. Vivado
makes this the INIT attribute of the flip-flop.

### The atomic commit

`Control_port` writes one byte in each cycle into a shadow copy. At the end
of the burst it strobes `commit`, and the block copies all 9 shadow bytes
into the live cells in one cycle.

The shadow follows the live cells when `write_enable` and `commit` are both
0. The shadow holds its value in each other cycle. A rule that looks only
at `write_enable` leaves the shadow stale for one cycle after the commit.

Therefore a view never shows a part of one write and a part of the next. A
16-bit or 32-bit value has no torn state, and a consumer does not have to
know when the port is idle.

Each cell holds a value, thus no cell is a strobe and each write is
idempotent.

### The cost

The 128 live cells replace the 128 cells of the register file, thus they are
not a cost. The change is:

| Item | Flip-flops |
|---|---|
| the shadow copy | +128 |
| the doorbell latch and the pending flag | +33 |
| the message latch of `Midi_out` | +32 |
| the copies in `Control_port` that go away | −32 |
| the init index that goes away | −4 |
| net | about +157 |

The measurement agrees: the `reg` bits of the generated Verilog go from 562
to 722, which is +160. This count also holds the temporary registers that
the `Always` DSL makes, thus it is a little more than the flip-flops. It is
0.13% of the XC7A100T.

The combinational logic goes up, and it does not go down. The shadow needs
its own write decode and its own follow mux, and these are larger than the
defaults mux, the second write decode and the read override that go away.
The AND operators of the Verilog go from 11 to 31.

## Control_port

### The interface

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a [@bits 8] (** the byte stream from the host UART *)
    ; in_valid : 'a (** a strobe: [in_data] holds one stream byte *)
    ; hold : 'a
    (** from the host transmitter: 1 stalls the response stream *)
    ; read_data : 'a [@bits 8]
    (** from [Control_regs]: the byte at [read_address] *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { out_data : 'a [@bits 8]
    (** the response byte stream: COBS frames with their delimiters *)
    ; out_valid : 'a (** the transmitter takes [out_data] when [hold] is 0 *)
    ; write_enable : 'a (** writes one shadow byte *)
    ; write_address : 'a [@bits cell_bits] (** the cell index to write *)
    ; write_data : 'a [@bits 8] (** the byte to write *)
    ; commit : 'a (** a strobe at the end of the burst *)
    ; read_address : 'a [@bits cell_bits] (** the cell index to read *)
    }
  [@@deriving hardcaml]
end
```

`Control_port` keeps the wire protocol and nothing more. It loses:

- the `msg_store` and `len_store` copies
- the second write decode
- the `cell_read` special case
- `Send_fsm` and its three `Byte_` states
- the `midi_data` and `midi_valid` outputs
- the `Init` state, `init_index` and the defaults mux
- the assertion `max_midi_len = 3`
- the `state` output, and the `busy` bit that replaced it: the atomic
  commit makes each view stable, thus no consumer waits for the port, and
  era four took the LED that showed it

The response path is the only user of `read_address` and `read_data`,
therefore no other block waits for it.

The names have no prefix, because the cells are the one thing that this
block reads and writes. The stream side is `in_data` and `out_data`.
Therefore `Control_port.O` and `Control_regs.I` agree field for field, and
a wrong connection is easy to see.

## Midi_out

### The interface

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; message : 'a Midi.Rtl.Message.t
    (** the block takes the message when [ready] is 1 *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { serial : 'a (** the MIDI line; it idles at 1 *)
    ; ready : 'a (** 1 when the block can take a message *)
    }
  [@@deriving hardcaml]
end

val create : clocks_per_bit:int -> Signal.t I.t -> Signal.t O.t
```

The block latches the message at the transfer, walks its bytes, and gives
each byte to the transmitter. `ready` is 0 from the transfer until the last
stop bit is on the line.

The block holds the transmitter, at 31250 baud, and it drives the MIDI line
directly. `clocks_per_bit` is an elaboration parameter, as it is for
`Uart_tx` today.

`Socket` holds this block, because the sequencer is the one source of MIDI
and the line is where its messages go. Therefore the message interface
stays inside the seat, and the top level wires no handshake for it.

## The top level

The pins do not change: `clk`, `btnCpuReset` and `RsRx` are the inputs, and
`led`, `RsTx` and `JD` are the outputs.

The LEDs:

| LED | Content |
|---|---|
| 0 | the run state |
| 1 | MIDI activity |
| 2 to 15 | 0 |

Era one to era three showed six things: a heartbeat, the two UART lines,
the MIDI line, the `busy` bit of the port and the run state. Era four keeps
two. The heartbeat showed only that the clock ran, and the three activity
lamps answered "is the wire alive", which the host answers better and
answers with a reason. The run state stays beside the MIDI lamp, because
the two are not one fact: the model is silent through the lead-in of one
bar, thus `led 0` answers "did the push take" while `led 1` is dark.

## The connections

One path goes in both directions between the blocks: `Control_port` gives
`read_address` and takes `read_data`. It is a chain and not a loop, because
`read_address` comes from a register in `Control_port`. Make the connection
with a `wire` and an `assign`, as `top.ml` does for `tx_busy` today.

The MIDI path is a chain to the pin, thus the top level makes no wire for
it: `Socket` gives the line and `JD[0]` takes it. The two handshakes inside
the seat — the source strobes and the `ready` of `Midi_out` — are wires of
`Socket`, and each far end is a register.

An interface field name must not be in `I` and in `O` of the same block. A
collision fails only when the block becomes a circuit, in a
`With_interface` simulation. In `Control_port` the two pairs that come near
are `read_data` in `I` against `read_address` in `O`, and `write_data` in
`O` against `in_data` in `I`. Each pair is distinct.

## What this design does not do

- It does not add the model block. `Midi_merge` gets a second source, and
  the source stays unconnected.
- It does not connect the board button.

## Changes to the host control

These three rules of `docs/host_control.md` change.

1. The SEED rule. The PRNG loads the seed at the end of a write that covers
   a SEED cell. The old rule was a load on the write of the last seed byte,
   and the atomic commit deletes that event, because the burst commits one
   time. The design has no PRNG, thus no block obeys this rule yet, as no
   block obeys the old one today.
2. The tearing rule. A write no longer tears a value of more than one byte.
   `Control_port` loses the note that tells a consumer to sample the cells
   only in the `Ready` state.
3. MIDI_GO and the poll rule. MIDI_GO reads 1 from the ring until
   `Midi_out` takes the message, and no longer until the last byte is on
   the line. `Control_regs` latches the message at the ring, thus a write
   to MIDI_MSG or MIDI_LEN after MIDI_GO reads 0 cannot damage the message.
   The driver reads MIDI_GO as 0 to know that the FPGA accepts the
   subsequent ring, and no longer to keep the message complete. A ring
   while a message is on the line waits, and it is not lost.

## The steps

1. `Midi.Rtl.Message`, the message interface.
2. `Midi_out`, with tests: the byte walk, the `ready` envelope, and the
   line at 31250 baud.
3. `Midi_merge`, with tests: the priority, and the `ready` of each source.
4. `Control_regs`, with tests: the defaults at power-on and after a clear,
   the views, the atomic commit, the RUN toggle, and the doorbell source
   with its ignore rules.
5. `Control_port`: remove the doorbell, the register file and the `Init`
   state, and use the two new inputs. Read each waveform difference before
   you accept it.
6. `top.ml`: connect the blocks, and tie `run_toggle` and `model.valid` to
   0.
7. Delete `Regfile`.
8. Correct `docs/host_control.md`.

`test/test_txn.ml` drives the board top level at the true UART rates and
examines the MIDI line. It must pass with no change. This is the test that
shows that the behavior on the wire and on the MIDI line stays the same.
