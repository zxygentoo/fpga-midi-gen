(* Integration test: the note-source socket with era six in the seat.

   [Socket] wires a source to the [Sequencer] and takes no model of its own — the source
   is a parameter, typed by [Source_intf]. Therefore its behaviour is only observable with
   a source in the seat, and the test that observes it is an integration test and not a
   unit test of the module. It lives here and not beside [Socket] because a test in
   [Socket] would put a concrete model into the library that exists so that the board
   knows none of them.

   The gate: the bytes on the MIDI line are the messages that [Frame.events_of_frames]
   states over THE FRAMES THE SOURCE IN THE SEAT ANSWERS, byte for byte and in order; the
   run stop plays the silent frame; and a second run from the same seed repeats the first.
   The two halves of the model path are proved separately — the walk gate of
   [jax/tests/test_rtl.py] holds the source's walk to the integer twin phase for phase,
   and this holds the sequencer to the frames that source answers — thus a failure here
   names the sequencer's side and never the network.

   THE REFERENCE IS THE SOURCE, PLAYED ON ITS OWN. [Source]'s bench drives a second
   instance of the same block from the same seed and the same elaboration, thus it answers
   the same frames; what this gate then measures is everything BETWEEN those frames and
   the wire — the socket, the sequencer, the note tracking and the UART.

   ERA SIX IS THE EASY TENANT OF THIS SOCKET. The source answers [step] in one cycle from
   a canvas that already stands, thus the step period constrains nothing; what it does
   instead is hold [idle] low for a whole walk at the rewind, which is exactly the wait
   that [WaitRewind] exists for.

   **THE GATE DOES NOT DEPEND ON THE LENGTH OF THAT WALK.** The run below carries the
   whole canvas and some steps past it, and PAST STEP T - 1 EVERY FRAME IS SILENT: the
   first of them releases what the last cell held and every one after it states no event
   at all. The stop's own silent frame is the same frame again. Thus the message stream
   saturates, the draw budget only has to be an over-estimate, and no number of this test
   has to track the machine's cycles. *)

open Base
module Socket = Mgen_board.Socket
module Control_intf = Mgen_core.Control_intf
module Frame = Mgen_core.Frame
module Midi = Mgen_core.Midi
module Diffusion = Mgen_diffusion.Diffusion
module Elaboration = Mgen_diffusion.Elaboration
module Quantized = Mgen_diffusion.Quantized
module Source = Mgen_diffusion.Source

(* The model of the test: drawn weights in the era's test shape, thus the test reads no
   checkpoint that git ignores. The canvas is short and the walk is two passes; P stays at
   the era's 48, because the seat registers of the opening are the corpus's. *)
let steps = 8
let walk = 2

(* One channel wider than the twin's own test shape: at H 6 a layer dwells 54 cycles and
   the fused floor asks 61, thus the shape the twin tests with is one this elaboration
   refuses. Nothing here reads the width. *)
let model =
  Quantized.Model.For_test.drawn { Diffusion.Config.layers = 4; width = 8 } ~seed:11
;;

let e = Elaboration.create model ~steps ~lanes:2 ~walk
let clocks_per_ms = 4
let step_ms = 400

(* One step sends at most eight messages, which is 240 bit times; the step period is 1 600
   cycles, thus the line is never the thing that holds a step. *)
let clocks_per_bit = 4

(* The walk, over-estimated: every pass at its model price and EVERY cell drawn, at 200
   cycles for a draw that measures 154 at P 48. The machine is shorter than this, and the
   gate does not read the number — it only has to cover the rewind. *)
let draw_budget =
  (walk * (Elaboration.pass_cycles e + (steps * Frame.voices * 200)))
  + Elaboration.cell_walk_cycles e
;;

(* The harness drives the parameter views directly and samples the line in each cycle.
   [play] runs one whole run: run to 1, the draw, the canvas and some silence past it,
   then run to 0 and the drain; it gives the messages that the line carried. *)
let harness () =
  let open Hardcaml in
  let module Sim = Cyclesim.With_interface (Socket.I) (Socket.O) in
  let sim =
    Sim.create (fun (i : _ Socket.I.t) ->
      Socket.create
        ~clocks_per_ms
        ~clocks_per_bit
        ~source:(Source.create ~e ~seed:i.params.seed)
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
  let play () =
    Buffer.clear wave;
    let period = clocks_per_ms * Bits.to_int_trunc !(inp.params.step_ms) in
    set inp.params.run 1;
    for _ = 1 to draw_budget + ((steps + 2) * period) do
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

(* the messages of the reference: the frames the source answers over the whole canvas, the
   silent frame of the stop behind them, and [Frame.events_of_frames] over the run.

   THE FRAMES PAST T - 1 ARE NOT IN THE LIST AND DO NOT HAVE TO BE. The first of them
   releases what the last cell held and every one after it states no event at all, thus
   the stop's silent frame states that release and the message stream saturates — which is
   why no number of this test tracks the machine's cycles. *)
let reference_messages ~seed ~channel ~velocity =
  let h = Source.For_test.Bench.harness ~e ~seed () in
  h.rewind ();
  (* [List.init] applies its function in the reverse index order, thus it cannot collect
     from a simulation; the fold steps in the true order *)
  let played =
    List.rev
      (List.fold (List.range 0 steps) ~init:[] ~f:(fun got (_ : int) -> h.play () :: got))
  in
  Array.of_list (played @ [ Frame.silent ])
  |> Frame.events_of_frames
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

let compare_run ~seed ~channel ~velocity =
  let inp, set, play = harness () in
  set inp.params.seed seed;
  set inp.params.channel channel;
  set inp.params.velocity velocity;
  set inp.params.step_ms step_ms;
  let circuit = play () in
  let reference = reference_messages ~seed ~channel ~velocity in
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
  compare_run
    ~seed:Control_intf.Default.seed
    ~channel:Control_intf.Default.channel
    ~velocity:Control_intf.Default.velocity;
  let inp, set, play = harness () in
  set inp.params.seed 99;
  set inp.params.channel 2;
  set inp.params.velocity 100;
  set inp.params.step_ms step_ms;
  let first = play () in
  let again = play () in
  Stdio.printf
    "%d messages, the second run repeats: %b\n"
    (List.length first)
    (same_messages first again)
;;
