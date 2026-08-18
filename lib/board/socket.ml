open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { serial : 'a } [@@deriving hardcaml]
end

let create ~clocks_per_ms ~clocks_per_bit ~source (i : _ I.t) : _ O.t =
  (* the strobes of the sequencer arrive after the source exists, and the [ready] of the
     line arrives after the sequencer; the wires break the order *)
  let source_rewind = wire 1 in
  let source_step = wire 1 in
  let midi_ready = wire 1 in
  let source_out =
    source
      { Source_intf.I.clock = i.clock
      ; clear = i.clear
      ; rewind = source_rewind
      ; step = source_step
      }
  in
  let sequencer =
    Sequencer.create
      ~clocks_per_ms
      { Sequencer.I.clock = i.clock
      ; clear = i.clear
      ; params = i.params
      ; source = source_out
      ; midi_ready
      }
  in
  assign source_rewind sequencer.source_rewind;
  assign source_step sequencer.source_step;
  let midi_out =
    Midi_out.create
      ~clocks_per_bit
      { Midi_out.I.clock = i.clock; clear = i.clear; message = sequencer.midi }
  in
  assign midi_ready midi_out.ready;
  { O.serial = midi_out.serial }
;;
