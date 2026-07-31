# The host control in RTL

## Scope

`docs/host_control.md` defines what the host sees. This document defines how
the FPGA supplies it: the blocks, what each block owns, and the interface of
each block.

This design changes three rules of the host control. The section "Changes to
the host control" lists them. Correct that document with the code.

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
             └──────────────┘                    └──┬────────┬──┘
                                                    │        │
                                             params │        │ doorbell
                                                    ▼        │
                                             ┌──────────┐    │
                                             │  Model   │    │
                                             │ (later)  │    │
                                             └────┬─────┘    │
                                                  │ model    │
                                                  ▼          ▼
                                               ┌────────────────┐
                                               │   Midi_merge   │
                                               │   (priority)   │
                                               └───────┬────────┘
                                                       │ out
                                                       ▼
                                               ┌────────────────┐
                                               │    Midi_out    │──▶ JD[0]
                                               │ message to line│
                                               └────────────────┘
```

| Block | It owns |
|---|---|
| `Control_port` | the wire protocol: the COBS decode, the frame buffer, the header checks, and the response |
| `Control_regs` | the control cells: the storage, the write decode, the named views, the value that each cell reads as, and the doorbell source |
| `Midi_merge` | which source gives the next message |
| `Midi_out` | one message to the MIDI line, and the transmitter |

`Regfile` goes away. No other block needs it, because a control cell is not
a memory location.

`cell_bits` is `address_bits_for Control.Reg.size`, which is 4. A cell
address is an index into the control section, and not a full address.
`Control_port` owns the address map, because it must give STATUS `02` for
an address outside the range.

## The message interface

Three blocks send or take a MIDI message. They use one interface, in
`lib/midi.ml`:

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

The simulation names the ports of a nested message `doorbell$data`,
`model$valid`, `out$len`, and so on.

A message and not a byte stream: therefore a source cannot put its bytes
between the bytes of another source. The merge at message boundaries is the
shape of the interface, and not a rule that a source must obey.

The limit of 3 bytes covers each channel voice message and each real-time
message. A System Exclusive message does not fit. A later design can add a
byte-stream source for it, and the merge rule then applies to that source
alone.

`Midi` is also the home of the MIDI status bytes that the model needs.

## Control_regs

### The interface

```ocaml
module Params = struct
  type 'a t =
    { run : 'a [@bits 8] (** the run state, in bit 0 *)
    ; channel : 'a [@bits 8] (** the MIDI channel of the model, 0 to 15 *)
    ; step_ms : 'a [@bits 16] (** the step period in ms *)
    ; gate_ms : 'a [@bits 16] (** the gate time in ms *)
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
    ; doorbell_ready : 'a
    (** from [Midi_merge]: 1 when the MIDI path takes the doorbell message *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { params : 'a Params.t (** the named views; each one is stable *)
    ; read_data : 'a [@bits 8] (** the byte at [read_address] *)
    ; doorbell : 'a Midi.Message.t (** the test message, as a message source *)
    }
  [@@deriving hardcaml]
end
```

The storage is 16 8-bit cells, and the write decode is uniform. The output
is a set of named views with the natural width of each value.

A consumer reads a view. A consumer never sees a byte, and no consumer
needs an address. The uniform byte array keeps the write decode simple, and
the views keep the consumers simple.

`Params` holds the parameters of the model, and nothing else. MIDI_MSG,
MIDI_LEN and MIDI_GO have no view: no block outside `Control_regs` reads
them, because the block gives the doorbell as a message source.

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

### The doorbell

The doorbell is a cell behavior, thus it stays with the cells. The block
holds the message latch and the pending flag, and it gives an ordinary
message source. Nothing in the MIDI path knows that the doorbell is
special.

- At a commit that covers MIDI_GO with bit 0 set, and with MIDI_LEN in 1 to
  3, and with no message pending: the block copies MIDI_MSG and MIDI_LEN
  into the latch and sets pending.
- `doorbell.valid` is the pending flag, and `doorbell.data` and
  `doorbell.len` are the latch.
- At the transfer, which is `doorbell.valid` and `doorbell_ready` together,
  the block clears pending.
- A read of MIDI_GO gives the pending flag.

The latch is the reason a write to MIDI_MSG cannot damage a message that
waits. The block reads the cells one time, at the ring.

### The defaults

Each live cell register takes `~initialize_to` and `~clear_to` with its
default value. Therefore the cells are correct at power-on and after a
clear, and the design has no `Init` state, no init walk, and no defaults
mux. The first request needs no wait.

The Verilog keeps the power-on value, as `reg [7:0] cell = 8'b...;`. Vivado
makes this the INIT attribute of the flip-flop.

### The atomic commit

`Control_port` writes one byte in each cycle into a shadow copy. At the end
of the burst it strobes `commit`, and the block copies all 16 shadow bytes
into the live cells in one cycle.

The shadow follows the live cells when `write_enable` and `commit` are both
0. The shadow holds its value in each other cycle. A rule that looks only
at `write_enable` leaves the shadow stale for one cycle after the commit.

Therefore a view never shows a part of one write and a part of the next. A
16-bit or 32-bit value has no torn state, and a consumer does not have to
know when the port is idle.

The block clears the stored MIDI_GO after each commit. Thus the stored
value is 1 only in the commit cycle of a burst that rang the doorbell.

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
    ; busy : 'a (** 1 while a transaction is in progress *)
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

`busy` replaces the `state` output. The `Init` state goes away with the
defaults, thus only `Ready` and `Busy` are left, and one bit gives them.
`busy` no longer tells a consumer when to sample a cell, because the atomic
commit makes each view stable. It is a diagnostic.

The response path is the only user of `read_address` and `read_data`,
therefore no other block waits for it.

The names have no prefix, because the cells are the one thing that this
block reads and writes. The stream side is `in_data` and `out_data`.
Therefore `Control_port.O` and `Control_regs.I` agree field for field, and
a wrong connection is easy to see.

## Midi_merge

### The interface

```ocaml
module I = struct
  type 'a t =
    { doorbell : 'a Midi.Message.t (** the test-message source *)
    ; model : 'a Midi.Message.t (** the model source *)
    ; out_ready : 'a (** from [Midi_out]: 1 when it can take a message *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { out : 'a Midi.Message.t (** the message of the source that has the grant *)
    ; doorbell_ready : 'a (** 1 when [Midi_out] takes the doorbell message *)
    ; model_ready : 'a (** 1 when [Midi_out] takes the model message *)
    }
  [@@deriving hardcaml]
end
```

The block has no state, therefore it has no clock and no clear. It is a
priority mux, and the `ready` of `Midi_out` goes back to the source that
has the grant:

```
granted_doorbell = doorbell.valid
granted_model    = model.valid &: ~:(doorbell.valid)

out.data  = mux2 granted_doorbell doorbell.data model.data
out.len   = mux2 granted_doorbell doorbell.len  model.len
out.valid = doorbell.valid |: model.valid

doorbell_ready = out_ready &: granted_doorbell
model_ready    = out_ready &: granted_model
```

The doorbell has the priority. The host waits in a poll loop, and the model
accepts a delay of one millisecond. A manual debug action is too rare to
stop the model.

`Midi_out` holds `ready` at 0 for the full length of a send. Therefore no
other source can put a message between the bytes of the message that goes
out, and the merge needs no state to make this true.

A later fairness rule, or a third source, gives this block state, and then
it takes a clock.

## Midi_out

### The interface

```ocaml
module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; message : 'a Midi.Message.t
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

## The top level

The pins do not change: `clk`, `btnCpuReset` and `RsRx` are the inputs, and
`led`, `RsTx` and `JD` are the outputs.

The LEDs:

| LED | Content |
|---|---|
| 0 | the heartbeat |
| 1 | RsRx activity |
| 2 | RsTx activity |
| 3 | MIDI activity |
| 4 | `busy` of `Control_port` |
| 5 to 15 | 0 |

The top level ties `run_toggle` to 0, and it ties `model.valid` to 0.

## The connections

Three paths go in both directions between the blocks:

- `Control_port` gives `read_address` and takes `read_data`.
- `Control_regs` gives `doorbell` and takes `doorbell_ready`.
- `Midi_merge` gives `out` and takes `out_ready`.

Each path is a chain and not a loop. `read_address` comes from a register
in `Control_port`. `doorbell.valid` is the pending flag, a register in
`Control_regs`. `out_ready` comes from a register in `Midi_out`.
`Midi_merge` is combinational between two registered ends. Make each
connection with a `wire` and an `assign`, as `top.ml` does for `tx_busy`
today.

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

1. `Midi.Message`, the message interface.
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
