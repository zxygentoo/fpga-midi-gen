(* play_diffusion: draws canvases of the masked-canvas checkpoint, prints the steps, and
   with -play sends them to the synthesizer. The float reference draws by default;
   [-quantized] draws the integer twin instead — THE BOARD'S MUSIC, HEARD BEFORE THE BOARD
   EXISTS, because the circuit of the next round must equal that engine draw for draw. One
   player, one set of flags, as era three's player held it.

   THE PLAYER IS THE BOARD'S PLAYBACK BEHAVIOR IN SOFTWARE, gesture for gesture. A batch
   of seeds is several whole pieces and not one piece in parts: each canvas plays in turn,
   [-gap] rests between two of them as a performer breathes between two chorales, and
   [-fade] takes the velocity down over the last bar of each — [Midi.fading] holds the
   rule and the reasons, and the JAX audition applies the same numbers. Neither gesture
   makes a crop ARRIVE; the ending is the whole-piece round's.

   The line format is the one of jax/diffusion/infer.py, thus a walk of this player and a
   walk of the twin compare with `diff`. That comparison is Gate C of the JAX seam, in
   jax/tests/test_parity.py: the two walks draw every uniform from the same generator in
   the same order, thus one seed is one piece on both sides. The fade moves velocities and
   never events, thus the step lines do not feel it.

   THE DRAW COMES BEFORE THE FIRST SOUND. Every canvas is drawn whole — [walk] forward
   passes for each — and only then plays; at the elected shape and the full budget that is
   minutes of compute for each canvas, thus a small -walk auditions the low-N regime the
   board negotiates with its lanes. The checkpoint states its own shape and no flag
   restates the training run. *)

open Core
module Control_intf = Mgen_core.Control_intf
module Frame = Mgen_core.Frame
module Midi = Mgen_core.Midi
module Signal = Mgen_core.Signal
module Diffusion = Mgen_diffusion.Diffusion
module Quantized = Mgen_diffusion.Quantized

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

(* a list, or LOW-HIGH: the rule of jax/midi.py's parse_seeds, thus one spelling names the
   same walks on both sides of the seam *)
let seeds_arg =
  Command.Arg_type.create (fun text ->
    match String.lsplit2 text ~on:'-' with
    | Some (low, high) ->
      List.range (Int.of_string low) (Int.of_string high) ~stop:`inclusive
    | None -> List.map (String.split text ~on:',') ~f:Int.of_string)
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

(* a canvas is a whole piece, thus several of them are several pieces: the separator
   stands only when there are several, and it does not start with "step", thus the walk
   gate's grep does not feel it *)
let print_banner ~at ~count = if count > 1 then printf "# canvas %d\n%!" at

(* One canvas to the wire, the fade on its note-ons. The drain releases what still sounds
   when the canvas ends or the stop arrives — a crop ends with its chord ringing, and
   Ctrl-C must not leave one on the synthesizer; [Signal] holds that rule. *)
let play_canvas fd stopped music ~step_ms ~fade ~channel ~velocity =
  let steps = List.length music in
  let sounding = ref (Set.empty (module Int)) in
  Exn.protect
    ~f:(fun () ->
      With_return.with_return (fun { return } ->
        List.iteri music ~f:(fun index events ->
          if Signal.stop_requested stopped then return ();
          print_step index events;
          let struck = Midi.faded_velocity ~velocity ~step:index ~steps ~fade in
          List.iter events ~f:(function
            | Frame.Event.On note ->
              Midi.send_note_on fd ~channel ~note ~velocity:struck;
              sounding := Set.add !sounding note
            | Frame.Event.Off note ->
              Midi.send_note_off fd ~channel ~note;
              sounding := Set.remove !sounding note);
          sleep_ms step_ms)))
    ~finally:(fun () ->
      Set.iter !sounding ~f:(fun note -> Midi.send_note_off fd ~channel ~note))
;;

let play canvases ~device ~step_ms ~gap ~fade ~channel ~velocity =
  let fd =
    try Midi.open_device device with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" device (Core_unix.Error.message error);
      exit 1
  in
  let stopped = Signal.watch_stop_play () in
  List.iteri canvases ~f:(fun at music ->
    if not (Signal.stop_requested stopped)
    then (
      (* the rest between two pieces: the drain of the one before has already run, thus
         this is a wait and not a message, as a performer breathes between two chorales *)
      if at > 0 then sleep_ms (gap * step_ms);
      print_banner ~at ~count:(List.length canvases);
      play_canvas fd stopped music ~step_ms ~fade ~channel ~velocity));
  Signal.exit_if_stopped stopped
;;

let command =
  Command.basic
    ~summary:"draw canvases of the masked-canvas checkpoint; print the steps, or play"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint; it states its own shape"
     and seeds =
       flag
         "-seeds"
         (optional_with_default [ 1 ] seeds_arg)
         ~doc:"SEEDS a list, or LOW-HIGH; each seed is one canvas, one whole piece"
     and steps =
       flag
         "-steps"
         (optional_with_default 128 int)
         ~doc:"T the steps of one canvas: eight measures at the training crop"
     and walk =
       flag
         "-walk"
         (optional_with_default 512 int)
         ~doc:"N the Gibbs passes, the paper's I times T"
     and temperature =
       flag
         "-temperature"
         (optional_with_default 1.0 float)
         ~doc:
           "F the temperature; 1.0 is the policy of the era, and no min-p floor exists \
            in this draw"
     and quantized =
       flag
         "-quantized"
         no_arg
         ~doc:
           " sample the integer twin of the circuit: the piece the board plays at this \
            seed. Every sampling flag applies as in the float path, and the temperature \
            bakes into the twin, as the bitstream will carry it"
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
     and gap =
       flag
         "-gap"
         (optional_with_default 32 int)
         ~doc:"N steps of silence between two canvases; 32 is two bars, 0 is none"
     and fade =
       flag
         "-fade"
         (optional_with_default 16 int)
         ~doc:"N steps of diminuendo at the end of a canvas; 16 is one bar, 0 is none"
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
          does not restate them; it only says what the model refused. [Nx_io] states a
          missing or unreadable checkpoint as a [Failure], and that message names the file
          the caller named. *)
       let canvases =
         try
           let config = Diffusion.Config.of_checkpoint checkpoint in
           let params = Diffusion.Params.load config ~path:checkpoint in
           (* one quantization serves every seed of the batch, as one bitstream will *)
           let model =
             if quantized
             then Some (Quantized.Model.of_params ~temperature params)
             else None
           in
           (* The float walk folds the seed and the twin takes it as the SEED cell will; a
              seed inside 32 bits opens the two on ONE generator state, thus the A/B at
              one seed hears the quantization and nothing else. *)
           let draw seed =
             let canvas =
               match model with
               | Some model ->
                 Quantized.Engine.run (Quantized.Engine.init model ~steps ~walk ~seed)
               | None -> Diffusion.gibbs params ~steps ~walk ~temperature ~seed
             in
             canvas |> Diffusion.frames_of_canvas |> Frame.events_of_frames
           in
           List.map seeds ~f:draw
         with
         | Invalid_argument message | Failure message ->
           Printf.eprintf "%s\n" message;
           exit 2
       in
       if send
       then play canvases ~device ~step_ms ~gap ~fade ~channel ~velocity
       else
         List.iteri canvases ~f:(fun at music ->
           print_banner ~at ~count:(List.length canvases);
           List.iteri music ~f:print_step))
;;

let () = Command_unix.run command
