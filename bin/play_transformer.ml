(* play_transformer: samples the trained transformer, prints the steps, and with -play
   sends them to the synthesizer. The model states its own Note Offs, thus the player is
   only a clock and a wire encoder: no gate and no seat logic. The config flags must equal
   the flags of the training run; the checkpoint holds only tensors. *)

open Core
module Control_intf = Mgen.Control_intf
module Midi = Mgen.Midi
module Token = Mgen.Token
module Transformer = Mgen.Transformer

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

(* The common chorale tempo is near 72 to 80 quarters each minute. One step is a
   sixteenth, thus 200 ms puts the quarter at exactly 75. The register default of 250 ms
   (60 quarters) serves the board, not the audition of this corpus. *)
let default_step_ms = 200

let show_sentence sentence =
  match sentence with
  | [] -> "-"
  | events ->
    String.concat
      ~sep:" "
      (List.map events ~f:(function
        | Token.Off pitch -> sprintf "off:%d" pitch
        | Token.On pitch -> sprintf "on:%d" pitch
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
        | Token.On _ -> true
        | Off _ | End -> false)
    in
    List.iter sentence ~f:(function
      | Token.Off pitch -> sounding := Set.remove !sounding pitch
      | Token.On pitch -> sounding := Set.add !sounding pitch
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
          | Token.Off note ->
            Midi.send_note_off fd ~channel ~note;
            sounding := Set.remove !sounding note
          | Token.On note ->
            Midi.send_note_on fd ~channel ~note ~velocity;
            sounding := Set.add !sounding note
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
     and d =
       flag
         "-d"
         (optional_with_default Transformer.Config.(baseline.d) int)
         ~doc:"N the residual width"
     and layers =
       flag
         "-layers"
         (optional_with_default Transformer.Config.(baseline.layers) int)
         ~doc:"N the layers"
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
           "F the temperature. The default won the ear test of 2026-08-05 with the min-p \
            default beside it; full temperature clashes more"
     and min_p =
       flag
         "-min-p"
         (optional_with_default 0.00390625 float)
         ~doc:
           "F drop tokens under this share of the peak; 0 turns the filter off. The \
            default is 1/256: wider filters eat the melody, per the sweep of 2026-08-05"
     and send = flag "-play" no_arg ~doc:" send the steps to the synthesizer"
     and stats = flag "-stats" no_arg ~doc:" print the audition metrics of the sample"
     and guard =
       flag
         "-guard"
         (optional_with_default
            Transformer.Guard.Grammar
            (Command.Arg_type.of_alist_exn
               [ "grammar", Transformer.Guard.Grammar
               ; "hazards", Transformer.Guard.Hazards
               ]))
         ~doc:
           "GUARD grammar for mask-era weights; hazards is the two-rule floor for the \
            unmasked era"
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
       let config = { Transformer.Config.d; layers; heads; context } in
       let like = Transformer.Params.to_ptree (Transformer.Params.draw config ~seed:0) in
       let tree = Kaun.Checkpoint.load checkpoint ~like in
       let params = Transformer.Params.of_ptree config tree in
       let music, sampled =
         Transformer.sample config params ~seed ~steps ~temperature ~min_p ~guard
       in
       if send
       then play music ~device ~step_ms ~channel ~velocity
       else if not stats
       then List.iteri music ~f:print_step;
       if stats then print_stats music;
       printf
         "min-p refused %.4f of the legal mass; guard held %.4f of the raw mass, %.4f of \
          the raw top choices, over %d draws\n\
          %!"
         sampled.Transformer.Sample_stats.refused
         sampled.illegal_mass
         sampled.illegal_top
         sampled.draws)
;;

let () = Command_unix.run command
