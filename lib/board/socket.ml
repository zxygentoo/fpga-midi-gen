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

module For_test = struct
  type t =
    { inputs : Bits.t ref I.t
    ; clear_line : unit -> unit
    ; run_for : cycles:int -> unit
    ; messages : unit -> int list list
    }

  let harness ~clocks_per_ms ~clocks_per_bit ~source () =
    let module Sim = Cyclesim.With_interface (I) (O) in
    let sim =
      Sim.create (fun (i : _ I.t) ->
        create ~clocks_per_ms ~clocks_per_bit ~source:(source i.params) i)
    in
    let inputs = Cyclesim.inputs sim in
    let out = Cyclesim.outputs ~clock_edge:Before sim in
    let wave = Buffer.create (1024 * 1024) in
    let run_for ~cycles =
      for _ = 1 to cycles do
        Cyclesim.cycle sim;
        Buffer.add_char wave (if Bits.to_bool !(out.serial) then '1' else '0')
      done
    in
    let messages () =
      Uart_rx.For_test.decode_line (Buffer.contents wave) ~clocks_per_bit
      |> Bytes.to_list
      |> List.map ~f:Char.to_int
      |> List.chunks_of ~length:Midi.max_message_bytes
    in
    { inputs; clear_line = (fun () -> Buffer.clear wave); run_for; messages }
  ;;
end
