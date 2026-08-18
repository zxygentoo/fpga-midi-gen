open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t
    ; midi_ready : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { midi : 'a Midi.Rtl.Message.t } [@@deriving hardcaml]
end

let create ~clocks_per_ms ~source (i : _ I.t) : _ O.t =
  (* the strobes of the sequencer arrive after the source exists; the wires break the
     order *)
  let source_rewind = wire 1 in
  let source_step = wire 1 in
  let source_ready = wire 1 in
  let source_out =
    source
      { Source_intf.I.clock = i.clock
      ; clear = i.clear
      ; rewind = source_rewind
      ; step = source_step
      ; ready = source_ready
      }
  in
  let sequencer =
    Sequencer.create
      ~clocks_per_ms
      { Sequencer.I.clock = i.clock
      ; clear = i.clear
      ; params = i.params
      ; source = source_out
      ; midi_ready = i.midi_ready
      }
  in
  assign source_rewind sequencer.source_rewind;
  assign source_step sequencer.source_step;
  assign source_ready sequencer.source_ready;
  { O.midi = sequencer.midi }
;;
