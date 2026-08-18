(* play_transformer: samples the trained transformer, prints the steps, and with -play
   sends them to the synthesizer. The model states its own Note Offs, thus the player is
   only a clock and a wire encoder: no gate and no seat logic. The config flags must equal
   the flags of the training run; the checkpoint holds only tensors.

   -quantized samples the reference of the circuit instead: the same seed then gives the
   piece the board plays, note for note — the audition of the quantized model. *)

open Core
module Control_intf = Mgen_core.Control_intf
module Quantized = Mgen_transformer.Quantized
module Midi = Mgen_core.Midi
module Token = Mgen_core.Token
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

let show_sentence sentence =
  match sentence with
  | [] -> "-"
  | events ->
    String.concat
      ~sep:" "
      (List.map events ~f:(function
        | Token.Start -> "start"
        | Token.On pitch -> sprintf "on:%d" pitch
        | Token.Off pitch -> sprintf "off:%d" pitch
        | Token.End -> "end"))
;;

let print_step index sentence = printf "step %3d  %s\n%!" index (show_sentence sentence)

let play music ~device ~step_ms ~channel ~velocity =
  let fd =
    try Midi.open_device device with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" device (Core_unix.Error.message error);
      exit 1
  in
  let sounding = ref (Set.empty (module Int)) in
  Exn.protect
    ~f:(fun () ->
      List.iteri music ~f:(fun index sentence ->
        print_step index sentence;
        List.iter sentence ~f:(function
          | Token.Start -> ()
          | Token.On note ->
            Midi.send_note_on fd ~channel ~note ~velocity;
            sounding := Set.add !sounding note
          | Token.Off note ->
            Midi.send_note_off fd ~channel ~note;
            sounding := Set.remove !sounding note
          | Token.End -> ());
        sleep_ms step_ms))
    ~finally:(fun () ->
      (* the drain: each open note closes, as the sequencer does at a stop *)
      Set.iter !sounding ~f:(fun note -> Midi.send_note_off fd ~channel ~note))
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
         ~doc:"N the heads"
     and context =
       flag
         "-context"
         (optional_with_default Transformer.Config.(baseline.context) int)
         ~doc:"N the window, in tokens"
     and seed = flag "-seed" (optional_with_default 1 int) ~doc:"N the seed"
     and steps = flag "-steps" (optional_with_default 64 int) ~doc:"N the steps to draw"
     and temperature =
       flag
         "-temperature"
         (optional_with_default 0.9 float)
         ~doc:
           "F the temperature. The default holds: 0.6 measured three times the \
            self-repetition, and the ear of 2026-08-07 heard it as dull — the \
            restatement of a cold draw is monotony, not melody"
     and min_p =
       flag
         "-min-p"
         (optional_with_default 0.00390625 float)
         ~doc:
           "F drop tokens under this share of the peak; 0 turns the filter off. The \
            default is 1/256: wider filters eat the melody, per the sweep of 2026-08-05"
     and piece_steps =
       flag
         "-piece-steps"
         (optional_with_default Transformer.piece_steps int)
         ~doc:
           "N re-anchor the walk every N steps: release the sounding pitches, clear the \
            context and feed START, carrying the step count and the PRNG. It applies to \
            -quantized alone: the float sampler is the improviser of \
            docs/transformer_model.md, which knows no piece. 0 is the endless walk"
     and quantized =
       flag
         "-quantized"
         no_arg
         ~doc:
           " sample the reference of the circuit: the piece the board plays at this \
            seed. Every configuration and sampling flag applies as in the float path; \
            the board commits to the values its bitstream was elaborated with"
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
       let config =
         Transformer.Config.of_checkpoint checkpoint ~heads ~context ~slope_span
       in
       let music, stats =
         if quantized
         then (
           (* the reference takes the rule of the SEED cell, as the board does *)
           if seed < 1 || seed > 0xFFFF_FFFF
           then (
             Printf.eprintf
               "-quantized takes the seed of the SEED cell: 1 to 0xFFFFFFFF\n";
             exit 2);
           (* the circuit takes the boundary as a bit-slice of its step counter *)
           if piece_steps > 0 && ((not (Int.is_pow2 piece_steps)) || piece_steps = 1)
           then (
             Printf.eprintf
               "-quantized takes -piece-steps as a power of two above 1, or 0\n";
             exit 2);
           let model =
             Quantized.Model.of_checkpoint
               ~temperature
               ~min_p
               ~piece_steps:(if piece_steps > 0 then Some piece_steps else None)
               config
               checkpoint
           in
           let engine = Quantized.Engine.init model ~seed in
           let sentence events =
             List.map events ~f:(fun { Quantized.Engine.voice = (_ : int); pitch; on } ->
               if on then Token.On pitch else Token.Off pitch)
           in
           (* the fold threads the engine through the steps in the drawn order *)
           let music =
             let (_ : Quantized.Engine.t), reversed =
               List.fold
                 (List.range 0 steps)
                 ~init:(engine, [])
                 ~f:(fun (engine, acc) (_ : int) ->
                   let engine, events = Quantized.Engine.next_step engine in
                   engine, sentence events :: acc)
             in
             List.rev reversed
           in
           music, None)
         else (
           let params = Transformer.Params.load config ~path:checkpoint in
           let ~music, ~stats =
             Transformer.sample config params ~seed ~steps ~temperature ~min_p
           in
           music, Some stats)
       in
       if send
       then play music ~device ~step_ms ~channel ~velocity
       else List.iteri music ~f:print_step;
       Option.iter stats ~f:(fun (stats : Transformer.sample_stats) ->
         printf
           "min-p refused %.4f of the legal mass; guard held %.4f of the raw mass, %.4f \
            of the raw top choices, over %d draws\n\
            %!"
           stats.Transformer.refused
           stats.illegal_mass
           stats.illegal_top
           stats.draws))
;;

let () = Command_unix.run command
