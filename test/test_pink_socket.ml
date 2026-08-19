(* Integration test: the pink model in the note-source socket of the board.

   [Socket] takes any source of [Source_intf] and gives the MIDI line. This seats
   [Pink.Source] in it and reads the line, thus it proves the one claim the socket exists
   to make: a model of another era drops into the seat and nothing else in the board knows
   which core it is. The transformer has the same gate in [test_socket.ml], and the two
   together are what "one interface serves every model" means.

   The gate: the bytes on the line are the messages that [Frame.events_of_frames] states
   over the frames of [Pink.next_frame], byte for byte and in order.

   **A frame cannot state a re-strike.** Era one gave the soprano and the alto a
   [restrike] policy, and the frame socket cannot express it: a voice that articulates a
   pitch it already holds states nothing new. The reference here has no such policy
   either, thus the two sides agree by construction and the smoothing is a property of the
   model and not a disagreement. [Pink] holds what it costs. *)

open Base
module Control_intf = Mgen_core.Control_intf
module Frame = Mgen_core.Frame
module Midi = Mgen_core.Midi
module Pink = Mgen_pink.Pink
module Socket = Mgen_board.Socket
module Source = Mgen_pink.Source

let clocks_per_ms = 4

(* One step sends at most eight messages, which is 240 bit times; the step period below is
   1 600 cycles, thus the line never holds a step. The draw is 18 cycles. *)
let clocks_per_bit = 4
let step_ms = 400

let harness ~seed =
  let open Hardcaml in
  let module Sim = Cyclesim.With_interface (Socket.I) (Socket.O) in
  let sim =
    Sim.create (fun (i : _ Socket.I.t) ->
      Socket.create
        ~clocks_per_ms
        ~clocks_per_bit
        ~source:(Source.create ~model:Pink.default ~seed:i.params.seed)
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
  set inp.params.seed seed;
  set inp.params.channel Control_intf.Default.channel;
  set inp.params.velocity Control_intf.Default.velocity;
  set inp.params.step_ms step_ms;
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
    let period = clocks_per_ms * step_ms in
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
  play
;;

(* the messages of the reference: the frames the model states, the silent frame of the
   stop behind them, and [Frame.events_of_frames] over the whole run *)
let reference_messages ~seed ~steps =
  (* [List.init] applies [f] in the reverse index order, thus it cannot walk a model; the
     fold steps in the true order *)
  let (_ : Pink.walk), frames =
    List.fold_map
      (List.range 0 steps)
      ~init:(Pink.create ~model:Pink.default ~seed)
      ~f:(fun walk (_ : int) -> Pink.next_frame walk)
  in
  Frame.events_of_frames (Array.of_list (frames @ [ Frame.silent ]))
  |> List.concat
  |> List.map ~f:(function
    | Frame.Event.On note ->
      Midi.note_on_bytes
        ~channel:Control_intf.Default.channel
        ~note
        ~velocity:Control_intf.Default.velocity
    | Frame.Event.Off note ->
      Midi.note_off_bytes ~channel:Control_intf.Default.channel ~note)
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

let () =
  let run ~seed ~steps =
    let play = harness ~seed in
    let circuit = play ~steps in
    let reference = reference_messages ~seed ~steps in
    Stdio.printf
      "seed %d: %d messages, the stream agrees: %b\n"
      seed
      (List.length circuit)
      (same_messages circuit reference);
    if not (same_messages circuit reference)
    then (
      show "the circuit " (List.take circuit 12);
      show "the reference" (List.take reference 12))
  in
  (* the walk crosses the alto and the tenor periods, thus the run holds a step in which
     each of the three fast voices moves and a step in which it holds *)
  run ~seed:Control_intf.Default.seed ~steps:24;
  run ~seed:7 ~steps:24;
  let play = harness ~seed:99 in
  let first = play ~steps:12 in
  let again = play ~steps:12 in
  Stdio.printf
    "%d messages, the second run repeats: %b\n"
    (List.length first)
    (same_messages first again)
;;
