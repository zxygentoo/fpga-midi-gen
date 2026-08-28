(* Integration test: the note-source socket against the reference of the model.

   [Socket] wires a source to the [Sequencer] and takes no model of its own — the source
   is a parameter, typed by [Source_intf]. Therefore its behaviour is only observable with
   a source in the seat, and the test that observes it is an integration test and not a
   unit test of the module.

   It lives here and not beside [Socket] for one more reason. A test in [Socket] would put
   a concrete model in the library that exists so that the board knows none of them: the
   board would then depend on every model it carries. The socket is the abstraction that
   keeps it from having to.

   The gate: the bytes on the MIDI line are the messages that [Frame.events_of_frames]
   states over the frames of [Quantized.Engine], byte for byte and in order, and a second
   run from the same seed repeats the first. The two halves of the model path are proved
   separately — [Source] gives the same frames as the engine, and this gives the same
   messages as the frames — thus a failure here names the sequencer and not the network.

   The seat holds the transmitter, thus the test reads the line and not a message wire: it
   decodes the waveform with the software receiver, at a baud divisor short enough that a
   step carries its messages.

   **The run stop is a silent frame.** The reference states it by playing one more frame,
   which is the whole rule the sequencer keeps. *)

open Base
module Socket = Mgen_board.Socket
module Control_intf = Mgen_core.Control_intf
module Frame = Mgen_core.Frame
module Midi = Mgen_core.Midi
module Model = Mgen_transformer.Model
module Quantized = Mgen_transformer.Quantized
module Source = Mgen_transformer.Source

(* The model of the test: drawn weights in the test shape, thus the test reads no
   checkpoint that git ignores. A step of it costs about 32 000 cycles, thus the step
   period below must be longer than that — the sequencer holds the boundary until the
   source is idle, and a period that is too short would only stretch the step and prove
   nothing about the decode. *)
let model = Model.For_test.drawn Model.For_test.shape ~seed:11

(* THE ORACLE OF THIS SIMULATION IS STILL OCAML'S. [Quantized.Engine] draws the same walk
   from the same drawn weights, thus the two models here are one model in two types; step
   three of the all-era cut replaces the engine with the circuit's own bench harness, as
   era six's socket simulation already reads it, and this bridge goes with it. *)
let engine_model = Quantized.Model.For_test.(init config ~seed:11)
let clocks_per_ms = 4
let step_ms = 9000

(* One step sends at most eight messages, which is 240 bit times; the step period below is
   36 000 cycles, thus the line is never the thing that holds a step. *)
let clocks_per_bit = 4

(* The harness drives the parameter views directly and samples the line in each cycle.
   [play] runs one whole run of [steps] steps: run to 1, then run to 0 in the middle of
   the last step, then the drain; it gives the messages that the line carried. *)
let harness () =
  let open Hardcaml in
  let module Sim = Cyclesim.With_interface (Socket.I) (Socket.O) in
  let sim =
    Sim.create (fun (i : _ Socket.I.t) ->
      Socket.create
        ~clocks_per_ms
        ~clocks_per_bit
        ~source:(Source.create ~model ~seed:i.params.seed)
        i)
  in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let set field value = field := Bits.of_unsigned_int ~width:(Bits.width !field) value in
  let wave = Buffer.create (1024 * 1024) in
  let cycle () =
    Cyclesim.cycle sim;
    Buffer.add_char wave (if Bits.to_bool !(out.serial) then '1' else '0')
  in
  (* every message of the sequencer is three bytes, thus the byte stream of the line
     divides into messages with no ambiguity *)
  let messages () =
    Mgen_board.Uart_rx.For_test.decode_line (Buffer.contents wave) ~clocks_per_bit
    |> Bytes.to_list
    |> List.map ~f:Char.to_int
    |> List.chunks_of ~length:Midi.max_message_bytes
  in
  let play ~steps =
    Buffer.clear wave;
    let period = clocks_per_ms * Bits.to_int_trunc !(inp.params.step_ms) in
    set inp.params.run 1;
    for _ = 1 to ((steps - 1) * period) + (period / 2) do
      cycle ()
    done;
    set inp.params.run 0;
    for _ = 1 to 2 * period do
      cycle ()
    done;
    messages ()
  in
  inp, set, play
;;

(* the messages of the reference: the frames the engine draws, the silent frame of the
   stop behind them, and [Frame.events_of_frames] over the whole run *)
let reference_messages ~seed ~channel ~velocity ~steps =
  let (_ : Quantized.Engine.t), frames =
    List.fold_map
      (List.range 0 steps)
      ~init:(Quantized.Engine.init engine_model ~seed)
      ~f:(fun engine (_ : int) ->
        let engine, step = Quantized.Engine.next_step engine in
        engine, step.Quantized.Engine.frame)
  in
  Frame.events_of_frames (Array.of_list (frames @ [ Frame.silent ]))
  |> List.concat
  |> List.map ~f:(function
    | Frame.Event.On note -> Midi.note_on_bytes ~channel ~note ~velocity
    | Frame.Event.Off note -> Midi.note_off_bytes ~channel ~note)
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

let compare_run ~seed ~steps =
  let inp, set, play = harness () in
  set inp.params.seed seed;
  set inp.params.channel Control_intf.Default.channel;
  set inp.params.velocity Control_intf.Default.velocity;
  set inp.params.step_ms step_ms;
  let circuit = play ~steps in
  let reference =
    reference_messages
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
  (* the walk must cross the lead-in of one bar, which draws nothing: the steps after it
     are the ones that state notes *)
  compare_run ~seed:Control_intf.Default.seed ~steps:20;
  let inp, set, play = harness () in
  set inp.params.seed 99;
  set inp.params.channel 2;
  set inp.params.velocity 100;
  set inp.params.step_ms step_ms;
  let first = play ~steps:19 in
  let again = play ~steps:19 in
  Stdio.printf
    "%d messages, the second run repeats: %b\n"
    (List.length first)
    (same_messages first again)
;;
