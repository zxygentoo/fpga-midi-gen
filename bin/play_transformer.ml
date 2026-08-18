(* play_transformer: samples the step-frame model, prints the steps, and with -play sends
   them to the synthesizer.

   The model states a frame for each step, and [Frame.events_of_frames] states the events
   of the wire: the releases of a step before its strikes, and no note held that the next
   frame does not ask for. Therefore the player is a clock and a wire encoder — no
   grammar, no seats, no gate.

   The line format is the one of jax/transformer/infer.py, thus a walk of this player and
   a walk of the twin compare with `diff`. That comparison is the walk gate of
   docs/transformer_model.md.

   The configuration flags must equal the flags of the training run; the checkpoint holds
   only tensors. The audition of the integer twin returns with the twin. *)

open Core
module Control_intf = Mgen_core.Control_intf
module Frame = Mgen_core.Frame
module Midi = Mgen_core.Midi
module Transformer = Mgen_transformer.Transformer

(* the argument check of play_pink: the range of a register is the range of the flag *)
let ranged name address =
  let { Control_intf.Reg.lower; upper } =
    Option.value_exn
      (Control_intf.Reg.bounds_of address)
      ~message:(name ^ " has no range in Control_intf.Reg")
  in
  Command.Arg_type.create (fun s ->
    let value = Int.of_string s in
    if value < lower || value > upper
    then (
      Printf.eprintf "%s must be %d to %d, not %d\n" name lower upper value;
      exit 2);
    value)
;;

let channel_arg = ranged "the channel" Control_intf.Reg.channel
let velocity_arg = ranged "the velocity" Control_intf.Reg.velocity

(* step_ms has no range in the register table; the player only refuses a step of zero *)
let step_ms_arg =
  Command.Arg_type.create (fun s ->
    let value = Int.of_string s in
    if value < 1
    then (
      Printf.eprintf "the step must be 1 ms or more, not %d\n" value;
      exit 2);
    value)
;;

let sleep_ms ms = ignore (Core_unix.nanosleep (Float.of_int ms /. 1000.) : float)
let default_device = "/dev/snd/midiC2D0"

(* the STEP_MS power-on value: the board and the audition share one tempo *)
let default_step_ms = Control_intf.Default.step_ms

let show_events events =
  if List.is_empty events
  then "-"
  else
    String.concat
      ~sep:" "
      (List.map events ~f:(function
        | Frame.Event.On pitch -> sprintf "on:%d" pitch
        | Frame.Event.Off pitch -> sprintf "off:%d" pitch))
;;

let print_step index events = printf "step %3d  %s\n%!" index (show_events events)

let play music ~device ~step_ms ~channel ~velocity =
  let fd =
    try Midi.open_device device with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" device (Core_unix.Error.message error);
      exit 1
  in
  let sounding = ref (Set.empty (module Int)) in
  (* Ctrl-C must not leave a chord ringing on the synthesizer, and an exception cannot
     carry that rule here.

     Measured on 5.2.0+ox, 2026-08-18: a [Sys.Break] raised out of the blocking sleep of a
     step is NOT caught by an enclosing handler. It reaches the top level with the notes
     still sounding, thus [Stdlib.Sys.catch_break] with a [try] around the loop drains
     nothing — and that is the idiom [play_pink] carries today.

     The handler therefore states the exit code and nothing else. It touches no note: a
     handler must be portable, thus it cannot hold the set of sounding pitches, and it has
     no business writing to the wire while a step may be halfway through a message. The
     loop reads the code at its next step and leaves by the ordinary road, where the drain
     stands. The wait is one step at the most. *)
  let stopped = Atomic.make 0 in
  List.iter
    [ Stdlib.Sys.sigint, 130; Stdlib.Sys.sigterm, 143 ]
    ~f:(fun (signal, code) ->
      ignore
        (Stdlib.Sys.Safe.signal
           signal
           (Stdlib.Sys.Signal_handle (fun (_ : int) -> Atomic.set stopped code))
         : Stdlib.Sys.signal_behavior));
  Exn.protect
    ~f:(fun () ->
      With_return.with_return (fun { return } ->
        List.iteri music ~f:(fun index events ->
          if Atomic.get stopped <> 0 then return ();
          print_step index events;
          List.iter events ~f:(function
            | Frame.Event.On note ->
              Midi.send_note_on fd ~channel ~note ~velocity;
              sounding := Set.add !sounding note
            | Frame.Event.Off note ->
              Midi.send_note_off fd ~channel ~note;
              sounding := Set.remove !sounding note);
          sleep_ms step_ms)))
    ~finally:(fun () ->
      (* the drain: the walk ends with its chord still sounding, as a piece does, and the
         sequencer of the board releases the same way at a stop *)
      Set.iter !sounding ~f:(fun note -> Midi.send_note_off fd ~channel ~note));
  match Atomic.get stopped with
  | 0 -> ()
  | code -> exit code
;;

let command =
  Command.basic
    ~summary:"sample the transformer checkpoint; print the steps, or play them"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
     and slope_span =
       flag
         "-alibi-span"
         (optional_with_default Transformer.Config.(baseline.slope_span) int)
         ~doc:"N the ALiBi exponent span; it must equal the span of the training run"
     and heads =
       flag
         "-heads"
         (optional_with_default Transformer.Config.(baseline.heads) int)
         ~doc:"N the heads; they must equal the heads of the training run"
     and context =
       flag
         "-context"
         (optional_with_default Transformer.Config.(baseline.context) int)
         ~doc:"N the window, in steps. A model trained long can sample short"
     and seed = flag "-seed" (optional_with_default 1 int) ~doc:"N the seed"
     and steps =
       flag
         "-steps"
         (optional_with_default 64 int)
         ~doc:"N the steps to draw, the silent lead-in of one bar inside"
     and temperature =
       flag
         "-temperature"
         (optional_with_default 1.0 float)
         ~doc:
           "F the temperature. The default is elected by ear over a sweep of 0.7 to 1.3, \
            against min-p 0.0039 to 0.15"
     and min_p =
       flag
         "-min-p"
         (optional_with_default 0.05 float)
         ~doc:
           "F drop the classes under this share of the peak; 0 turns the filter off. The \
            peak always stays, thus a draw always exists"
     and send = flag "-play" no_arg ~doc:" send the steps to the synthesizer"
     and device =
       flag
         "-device"
         (optional_with_default default_device string)
         ~doc:"PATH the rawmidi device of the synth"
     and step_ms =
       flag
         "-step-ms"
         (optional_with_default default_step_ms step_ms_arg)
         ~doc:"N the step, in milliseconds (default: 75 quarters each minute)"
     and channel =
       flag
         "-channel"
         (optional_with_default Control_intf.Default.channel channel_arg)
         ~doc:"N MIDI channel 0 to 15 (default: the S-1 channel 3)"
     and velocity =
       flag
         "-velocity"
         (optional_with_default Control_intf.Default.velocity velocity_arg)
         ~doc:"N the Note On velocity"
     in
     fun () ->
       (* The rules of the draw and of the shapes live with the model, thus the player
          does not restate them; it only says what the model refused, and it says it the
          way the range flags above do. [Nx_io] states a missing or unreadable checkpoint
          as a [Failure], and that message names the file the caller named. *)
       let music =
         try
           let config =
             Transformer.Config.of_checkpoint checkpoint ~heads ~context ~slope_span
           in
           let params = Transformer.Params.load config ~path:checkpoint in
           Transformer.sample config params ~seed ~steps ~temperature ~min_p
           |> Frame.events_of_frames
         with
         | Invalid_argument message | Failure message ->
           Printf.eprintf "%s\n" message;
           exit 2
       in
       if send
       then play music ~device ~step_ms ~channel ~velocity
       else List.iteri music ~f:print_step)
;;

let () = Command_unix.run command
