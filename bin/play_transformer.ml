(* play_transformer: samples the trained transformer, prints the steps, and with -play
   sends them to the synthesizer. The model states its own Note Offs, thus the player is
   only a clock and a wire encoder: no gate and no seat logic. The config flags must equal
   the flags of the training run; the checkpoint holds only tensors.

   -quantized samples the integer twin of the circuit instead: the same seed then gives
   the piece the board plays, note for note — the audition of the quantized model. *)

open Core
module Control_intf = Mgen_core.Control_intf
module Fixed = Mgen_transformer.Fixed
module Jsb = Mgen_corpus.Jsb
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

(* The audition metrics: the texture and the register of a sample. The corpus references,
   on the same walk: single-ON 11%, four-voice 92%, top pitch 71/75/81, above-87 0%. The
   share metrics read over all steps; the register reads over the steps in which something
   sounds. The finer harmony metrics of 2026-08-05 are cut: none of them ranked what the
   ear ranks. The loss ranks the checkpoints, these two lines catch the knob pathologies,
   and the ear decides the rest. *)
let print_stats music =
  let steps = List.length music in
  let sounding = ref (Set.empty (module Int)) in
  let tops = ref [] in
  let fours = ref 0 in
  let singles = ref 0 in
  List.iter music ~f:(fun sentence ->
    let ons =
      List.count sentence ~f:(function
        | Token.Start -> false
        | On _ -> true
        | Off _ | End -> false)
    in
    List.iter sentence ~f:(function
      | Token.Start -> ()
      | Token.On pitch -> sounding := Set.add !sounding pitch
      | Token.Off pitch -> sounding := Set.remove !sounding pitch
      | Token.End -> ());
    if ons = 1 then incr singles;
    if Set.length !sounding = 4 then incr fours;
    Option.iter (Set.max_elt !sounding) ~f:(fun top -> tops := top :: !tops));
  let share count = 100. *. Float.of_int count /. Float.of_int (max 1 steps) in
  printf "single-ON %.0f%%  four-voice %.0f%%\n" (share !singles) (share !fours);
  let tops = Array.of_list (List.sort !tops ~compare:Int.compare) in
  let sounded = Array.length tops in
  if sounded > 0
  then (
    let pct p = tops.(min (sounded - 1) (p * sounded / 100)) in
    let high = Array.count tops ~f:(fun top -> top > 87) in
    printf
      "top pitch p50/p90/max %d/%d/%d  above 87: %.1f%%\n"
      (pct 50)
      (pct 90)
      tops.(sounded - 1)
      (100. *. Float.of_int high /. Float.of_int sounded))
;;

(* The self-repetition of the top voice: the share of one-bar windows that come again in
   the sample, exactly and as a transposed contour. A melody is restated material, and
   this is the one number that agreed with the ear about its absence — the corpus restates
   36% of its bars and the models of 2026-08-07 restate 0 to 10%. A window of a held note
   is no motif, thus a window counts only with three distinct pitches inside it: without
   that filter a drone reads as 24% repetition. The spread over seeds is six or seven
   points, so read this over several seeds and never over one.

   The number counts restatement, which is necessary for a melody and not sufficient. A
   cold draw raises it — temperature 0.6 gave three times the restatement of 0.9 — and the
   ear of 2026-08-07 called the result dull: the model repeats because it has nothing else
   to say, not because it holds a motif. Read a high number with the temperature beside
   it, and never buy repetition with entropy. *)
let print_repetition music =
  let bar = Jsb.bar_steps in
  let sounding = ref (Set.empty (module Int)) in
  let tops =
    Array.of_list_map music ~f:(fun sentence ->
      List.iter sentence ~f:(function
        | Token.Start | Token.End -> ()
        | Token.On pitch -> sounding := Set.add !sounding pitch
        | Token.Off pitch -> sounding := Set.remove !sounding pitch);
      Set.max_elt !sounding)
  in
  let windows =
    List.filter_map
      (List.range 0 (max 0 (Array.length tops - bar)))
      ~f:(fun at ->
        match Option.all (Array.to_list (Array.sub tops ~pos:at ~len:bar)) with
        | None -> None
        | Some pitches ->
          if List.length (List.dedup_and_sort pitches ~compare:Int.compare) < 3
          then None
          else Some pitches)
  in
  let recurring windows =
    let compare = List.compare Int.compare in
    List.sort windows ~compare
    |> List.group ~break:(fun a b -> compare a b <> 0)
    |> List.sum (module Int) ~f:(fun group ->
      let size = List.length group in
      if size > 1 then size else 0)
  in
  let contour pitches =
    List.map2_exn (List.drop_last_exn pitches) (List.tl_exn pitches) ~f:(fun a b -> b - a)
  in
  match List.length windows with
  | 0 -> printf "one-bar repeats: no window moves enough to count\n"
  | count ->
    let share recurring = 100. *. Float.of_int recurring /. Float.of_int count in
    printf
      "one-bar repeats %.0f%% exact, %.0f%% contour, over %d windows (the corpus: 36%%, \
       40%%)\n"
      (share (recurring windows))
      (share (recurring (List.map windows ~f:contour)))
      count
;;

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
     and quantized =
       flag
         "-quantized"
         no_arg
         ~doc:
           " sample the integer twin of the circuit: the piece the board plays at this \
            seed. Every configuration and sampling flag applies as in the float path; \
            the board commits to the values its bitstream was elaborated with"
     and send = flag "-play" no_arg ~doc:" send the steps to the synthesizer"
     and show_stats =
       flag "-stats" no_arg ~doc:" print the audition metrics of the sample"
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
           (* the twin takes the rule of the SEED cell, as the board does *)
           if seed < 1 || seed > 0xFFFF_FFFF
           then (
             Printf.eprintf
               "-quantized takes the seed of the SEED cell: 1 to 0xFFFFFFFF\n";
             exit 2);
           let model = Fixed.Model.of_checkpoint ~temperature ~min_p config checkpoint in
           let engine = Fixed.Engine.create model ~seed in
           let sentence events =
             List.map events ~f:(fun { Fixed.Engine.seat = (_ : int); pitch; on } ->
               if on then Token.On pitch else Token.Off pitch)
           in
           (* [List.init] applies [f] in the reverse index order, thus it cannot collect
              from the mutable engine; the fold steps in the true order *)
           let music =
             List.rev
               (List.fold (List.range 0 steps) ~init:[] ~f:(fun acc (_ : int) ->
                  sentence (Fixed.Engine.step_events engine) :: acc))
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
       else if not show_stats
       then List.iteri music ~f:print_step;
       if show_stats
       then (
         print_stats music;
         print_repetition music);
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
