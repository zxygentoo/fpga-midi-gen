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
  type 'a t = { midi : 'a Midi.Message.t } [@@deriving hardcaml]
end

let create ~clocks_per_ms ~source (i : _ I.t) : _ O.t =
  (* the strobes of the sequencer arrive after the source exists; the wires break the
     order *)
  let source_rewind = wire 1 in
  let source_step = wire 1 in
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
      ; midi_ready = i.midi_ready
      }
  in
  assign source_rewind sequencer.source_rewind;
  assign source_step sequencer.source_step;
  { O.midi = sequencer.midi }
;;

(* The integration harness drives the parameter views directly and takes every message.
   [play] runs one whole run: run to 1, [steps] boundaries, run to 0, and the drain. *)
let clocks_per_ms = 4

let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim =
    Sim.create (fun (i : _ I.t) ->
      create
        ~clocks_per_ms
        ~source:(Voss.create ~params:Pink.Params.default ~seed:i.params.seed)
        i)
  in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  inp.midi_ready := Bits.vdd;
  let set field value = field := Bits.of_unsigned_int ~width:(Bits.width !field) value in
  let messages = ref [] in
  let cycle () =
    Cyclesim.cycle sim;
    if Bits.to_bool !(out.midi.valid)
    then (
      let data = Bits.to_int_trunc !(out.midi.data) in
      messages
      := [ data land 0xff; (data lsr 8) land 0xff; (data lsr 16) land 0xff ] :: !messages)
  in
  let play ~steps =
    messages := [];
    set inp.params.run 1;
    for _ = 1 to steps * clocks_per_ms * Bits.to_int_trunc !(inp.params.step_ms) do
      cycle ()
    done;
    set inp.params.run 0;
    for _ = 1 to 4 * clocks_per_ms * Bits.to_int_trunc !(inp.params.step_ms) do
      cycle ()
    done;
    List.rev !messages
  in
  inp, set, play
;;

(* the messages that the reference composes under the rules of the design: one on and one
   off for each note, in the step order, with the off of the legato case in front of the
   next on — the flat byte stream is the same either way *)
let reference_messages ~seed ~channel ~velocity ~count =
  let notes =
    Pink.notes Pink.Params.default ~seed
    |> (fun sequence -> Sequence.take sequence count)
    |> Sequence.to_list
  in
  List.concat_map notes ~f:(fun note ->
    [ [ Midi.note_on lor channel; note; velocity ]
    ; [ Midi.note_off lor channel; note; Midi.release_velocity ]
    ])
;;

let%expect_test "the message stream is the reference stream" =
  let inp, set, play = harness () in
  set inp.params.seed Control.Default.seed;
  set inp.params.channel Control.Default.channel;
  set inp.params.velocity Control.Default.velocity;
  set inp.params.step_ms 3;
  set inp.params.gate_ms 1;
  let circuit = play ~steps:32 in
  let reference =
    reference_messages
      ~seed:Control.Default.seed
      ~channel:Control.Default.channel
      ~velocity:Control.Default.velocity
      ~count:(List.length circuit / 2)
  in
  Stdio.printf
    "%d messages, the stream agrees: %b\n"
    (List.length circuit)
    ([%compare.equal: int list list] circuit reference);
  [%expect {| 62 messages, the stream agrees: true |}]
;;

let%expect_test "the legato stream carries the same bytes" =
  let inp, set, play = harness () in
  set inp.params.seed Control.Default.seed;
  set inp.params.channel Control.Default.channel;
  set inp.params.velocity Control.Default.velocity;
  set inp.params.step_ms 3;
  (* the gate is not less than the step, thus each off goes immediately before the next
     on, and the last off comes from the stop *)
  set inp.params.gate_ms 9;
  let circuit = play ~steps:16 in
  let reference =
    reference_messages
      ~seed:Control.Default.seed
      ~channel:Control.Default.channel
      ~velocity:Control.Default.velocity
      ~count:(List.length circuit / 2)
  in
  (* the same pairs in a different interleave: on, then off-with-next-on; the comparison
     sorts each adjacent pair back to on-off order *)
  let paired = function
    | on :: rest ->
      let rec go acc = function
        | off :: on :: rest -> go (on :: off :: acc) rest
        | [ off ] -> List.rev (off :: acc)
        | [] -> List.rev acc
      in
      on :: go [] rest
    | [] -> []
  in
  Stdio.printf
    "%d messages, the stream agrees: %b\n"
    (List.length circuit)
    ([%compare.equal: int list list] (paired circuit) reference);
  [%expect {| 30 messages, the stream agrees: true |}]
;;

let%expect_test "a new run repeats the sequence from the seed" =
  let inp, set, play = harness () in
  set inp.params.seed 99;
  set inp.params.channel 2;
  set inp.params.velocity 100;
  set inp.params.step_ms 2;
  set inp.params.gate_ms 1;
  let first = play ~steps:12 in
  let again = play ~steps:12 in
  Stdio.printf
    "%d messages, the second run repeats: %b\n"
    (List.length first)
    ([%compare.equal: int list list] first again);
  [%expect {| 18 messages, the second run repeats: true |}]
;;
