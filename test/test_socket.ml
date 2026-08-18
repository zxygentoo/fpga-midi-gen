(* Integration test: the note-source socket against the software player.

   [Socket] wires a source to the [Sequencer] and takes no model of its own — the source
   is a parameter, typed by [Source_intf]. Therefore its behaviour is only observable with
   a source in the seat, and the test that observes it is an integration test and not a
   unit test of the module.

   It lives here and not beside [Socket] for one more reason. A test in [Socket] would put
   a concrete model in the library that exists so that the board knows none of them: with
   [Pink] in [lib/board], the transformer would ask for the same and the board would
   depend on every model it carries. The socket is the abstraction that keeps it from
   having to.

   The gate: the circuit sends the messages the software player composes, byte for byte
   and in order, and a second run from the same seed repeats the first. *)

open Base
module Socket = Mgen_board.Socket
module Control_intf = Mgen_core.Control_intf
module Midi = Mgen_core.Midi
module Pink = Mgen_pink.Pink
module Player = Mgen_pink.Player
module Source = Mgen_pink.Source

(* The harness drives the parameter views directly and takes every message. [play] runs
   one whole run of [steps] steps: run to 1, then run to 0 in the middle of the last step,
   then the drain. The run start costs the rewind walk, thus the drop is at the middle of
   a step and not at a boundary; the step period must be longer than the rewind walk,
   which the tests give it. *)
let clocks_per_ms = 4

let harness ~model () =
  let open Hardcaml in
  let module Sim = Cyclesim.With_interface (Socket.I) (Socket.O) in
  let sim =
    Sim.create (fun (i : _ Socket.I.t) ->
      Socket.create ~clocks_per_ms ~source:(Source.create ~model ~seed:i.params.seed) i)
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

(* the messages of the reference: the player gives the events of each step, and the run
   ends with the stop. The board sequencer has no gate, thus the highest voice sends its
   Note Off immediately before its next Note On, and the stop closes every voice. *)
let reference_messages ~model ~seed ~channel ~velocity ~steps =
  let encode = function
    | Player.Event.On note -> Midi.note_on_bytes ~channel ~note ~velocity
    | Player.Event.Off note -> Midi.note_off_bytes ~channel ~note
  in
  (* the fold pushes each step in front and one [List.rev_append] puts the run in order:
     an append inside the fold is quadratic *)
  let player, reversed =
    List.fold
      (List.range 0 steps)
      ~init:(Player.create ~model ~seed, [])
      ~f:(fun (player, acc) _ ->
        let player, struck = Player.step player in
        player, List.rev_append struck acc)
  in
  let _, stopped = Player.stop player in
  List.map (List.rev_append reversed stopped) ~f:encode
;;

let same_messages = List.equal (List.equal Int.equal)

(* the first messages of each side, in hex, when the two disagree *)
let show label messages =
  Stdio.printf
    "%s %s\n"
    label
    (String.concat
       ~sep:" "
       (List.map messages ~f:(fun bytes ->
          String.concat (List.map bytes ~f:(Printf.sprintf "%02x")))))
;;

let compare_run ~model ~seed ~step_ms ~steps =
  let inp, set, play = harness ~model () in
  set inp.params.seed seed;
  set inp.params.channel Control_intf.Default.channel;
  set inp.params.velocity Control_intf.Default.velocity;
  set inp.params.step_ms step_ms;
  let circuit = play ~steps in
  let reference =
    reference_messages
      ~model
      ~seed
      ~channel:Control_intf.Default.channel
      ~velocity:Control_intf.Default.velocity
      ~steps
  in
  Stdio.printf
    "%d messages, the stream agrees: %b\n"
    (List.length circuit)
    (same_messages circuit reference);
  if not (same_messages circuit reference)
  then (
    show "the circuit " (List.take circuit 12);
    show "the reference" (List.take reference 12))
;;

let () =
  compare_run ~model:Pink.default ~seed:Control_intf.Default.seed ~step_ms:20 ~steps:32;
  let inp, set, play = harness ~model:Pink.default () in
  set inp.params.seed 99;
  set inp.params.channel 2;
  set inp.params.velocity 100;
  set inp.params.step_ms 20;
  let first = play ~steps:12 in
  let again = play ~steps:12 in
  Stdio.printf
    "%d messages, the second run repeats: %b\n"
    (List.length first)
    (same_messages first again)
;;
