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

(* The integration harness drives the parameter views directly and takes every message.
   [play] runs one whole run of [steps] steps: run to 1, then run to 0 in the middle of
   the last step, then the drain. The run start costs the rewind walk, thus the drop is at
   the middle of a step and not at a boundary; the step period must be longer than the
   rewind walk, which the tests give it. *)
let clocks_per_ms = 4

let harness ~model () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim =
    Sim.create (fun (i : _ I.t) ->
      create ~clocks_per_ms ~source:(Voss.create ~model ~seed:i.params.seed) i)
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
    let period = clocks_per_ms * Bits.to_int_trunc !(inp.params.step_ms) in
    set inp.params.run 1;
    for _ = 1 to ((steps - 1) * period) + (period / 2) do
      cycle ()
    done;
    set inp.params.run 0;
    for _ = 1 to 2 * period do
      cycle ()
    done;
    List.rev !messages
  in
  inp, set, play
;;

(* the messages of the reference: the player gives the events of each step and of each
   gate, and the run ends with the stop. One definition of the rule serves the audition
   tool and this test. *)
let reference_messages ~model ~seed ~channel ~velocity ~steps ~gated =
  let encode = function
    | Player.Event.On note -> [ Midi.note_on lor channel; note; velocity ]
    | Player.Event.Off note -> [ Midi.note_off lor channel; note; Midi.release_velocity ]
  in
  let player, messages =
    List.fold
      (List.range 0 steps)
      ~init:(Player.create ~model ~seed, [])
      ~f:(fun (player, acc) _ ->
        let player, struck = Player.step player in
        let player, closed = if gated then Player.gate player else player, [] in
        player, acc @ List.map (struck @ closed) ~f:encode)
  in
  let _, stopped = Player.stop player in
  messages @ List.map stopped ~f:encode
;;

let compare_run ~model ~seed ~step_ms ~gate_ms ~steps =
  let inp, set, play = harness ~model () in
  set inp.params.seed seed;
  set inp.params.channel Control.Default.channel;
  set inp.params.velocity Control.Default.velocity;
  set inp.params.step_ms step_ms;
  set inp.params.gate_ms gate_ms;
  let circuit = play ~steps in
  let reference =
    reference_messages
      ~model
      ~seed
      ~channel:Control.Default.channel
      ~velocity:Control.Default.velocity
      ~steps
      ~gated:(gate_ms < step_ms)
  in
  Stdio.printf
    "%d messages, the stream agrees: %b\n"
    (List.length circuit)
    ([%compare.equal: int list list] circuit reference);
  if not ([%compare.equal: int list list] circuit reference)
  then (
    Stdio.print_s ([%sexp_of: int list list] (List.take circuit 12));
    Stdio.print_s ([%sexp_of: int list list] (List.take reference 12)))
;;

let%expect_test "the four voices agree with the player, message for message" =
  compare_run
    ~model:Pink.default
    ~seed:Control.Default.seed
    ~step_ms:20
    ~gate_ms:8
    ~steps:32;
  [%expect {| 88 messages, the stream agrees: true |}]
;;

let%expect_test "the four voices agree with no gate" =
  (* the gate is not less than the step, thus it never comes: the highest voice sends its
     Note Off immediately before its next Note On, and the stop closes every voice *)
  compare_run
    ~model:Pink.default
    ~seed:Control.Default.seed
    ~step_ms:20
    ~gate_ms:40
    ~steps:16;
  [%expect {| 46 messages, the stream agrees: true |}]
;;

let%expect_test "a new run repeats the sequence from the seed" =
  let inp, set, play = harness ~model:Pink.default () in
  set inp.params.seed 99;
  set inp.params.channel 2;
  set inp.params.velocity 100;
  set inp.params.step_ms 20;
  set inp.params.gate_ms 8;
  let first = play ~steps:12 in
  let again = play ~steps:12 in
  Stdio.printf
    "%d messages, the second run repeats: %b\n"
    (List.length first)
    ([%compare.equal: int list list] first again);
  [%expect {| 36 messages, the second run repeats: true |}]
;;
