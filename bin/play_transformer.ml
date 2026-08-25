(* play_transformer: samples the step-frame model, prints the steps, and with -play sends
   them to the synthesizer.

   The model states a frame for each step, and [Frame.events_of_frames] states the events
   of the wire: the releases of a step before its strikes, and no note held that the next
   frame does not ask for. Therefore the player is a clock and a wire encoder — no
   grammar, no seats, no gate.

   The line format is the one of jax/transformer/infer.py, thus a walk of this player and
   a walk of the twin compare with `diff`. That comparison is the walk gate of
   docs/transformer.md, and [Player] holds the format and the play loop that answer it.

   The float model draws by default; [-quantized] draws the integer twin — the piece the
   board plays at this seed, heard off the board. The configuration flags must equal the
   flags of the training run; the checkpoint holds only tensors. *)

open Core
module Control_intf = Mgen_core.Control_intf
module Frame = Mgen_core.Frame
module Player = Mgen_core.Player
module Quantized = Mgen_transformer.Quantized
module Transformer = Mgen_transformer.Transformer

(* the STEP_MS power-on value: the board and the audition share one tempo *)
let default_step_ms = Control_intf.Default.step_ms

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
         (optional_with_default Transformer.elected_temperature float)
         ~doc:
           "F the temperature. The default is elected by ear over a sweep of 0.7 to 1.3, \
            against min-p 0.0039 to 0.15"
     and min_p =
       flag
         "-min-p"
         (optional_with_default Transformer.elected_min_p float)
         ~doc:
           "F drop the classes under this share of the peak; 0 turns the filter off. The \
            peak always stays, thus a draw always exists"
     and quantized =
       flag
         "-quantized"
         no_arg
         ~doc:
           " sample the integer twin of the circuit: the piece the board plays at this \
            seed. Every sampling flag applies as in the float path, and the policy bakes \
            into the twin, as the bitstream carries it"
     and send = flag "-play" no_arg ~doc:" send the steps to the synthesizer"
     and device =
       flag
         "-device"
         (optional_with_default Player.default_device string)
         ~doc:"PATH the rawmidi device of the synth"
     and step_ms =
       flag
         "-step-ms"
         (optional_with_default default_step_ms Player.step_ms_arg)
         ~doc:"N the step, in milliseconds (default: 75 quarters each minute)"
     and channel =
       flag
         "-channel"
         (optional_with_default Control_intf.Default.channel Player.channel_arg)
         ~doc:"N MIDI channel 0 to 15 (default: the S-1 channel 3)"
     and velocity =
       flag
         "-velocity"
         (optional_with_default Control_intf.Default.velocity Player.velocity_arg)
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
           let frames =
             if quantized
             then (
               let model =
                 Quantized.Model.of_checkpoint ~temperature ~min_p config checkpoint
               in
               List.folding_map
                 (List.range 0 steps)
                 ~init:(Quantized.Engine.init model ~seed)
                 ~f:(fun engine (_ : int) ->
                   let next, (step : Quantized.Engine.step) =
                     Quantized.Engine.next_step engine
                   in
                   next, step.frame)
               |> Array.of_list)
             else (
               let params = Transformer.Params.load config ~path:checkpoint in
               Transformer.sample config params ~seed ~steps ~temperature ~min_p)
           in
           Frame.events_of_frames frames
         with
         | Invalid_argument message | Failure message ->
           Printf.eprintf "%s\n" message;
           exit 2
       in
       if send
       then Player.play music ~device ~step_ms ~channel ~velocity
       else List.iteri music ~f:(fun step events -> Player.print_step ~step events))
;;

let () = Command_unix.run command
