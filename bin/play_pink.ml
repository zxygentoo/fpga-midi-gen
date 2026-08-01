(* play_pink: the reference pink-noise model plays on the synthesizer over USB MIDI.

   play_pink [--device PATH] [--seed N] [--notes N] [--step-ms N] [--gate-ms N]
   [--velocity N] [--channel N] [--rows N] [--stretch N] [--root N]

   The defaults are the power-on values of the control cells, thus the tool with no
   arguments is a preview of the board at power-on plus RUN. --notes 0 plays with no end.
   The device default is the one rawmidi device in /dev/snd, the S-1. The FPGA is not in
   this loop. *)

open Core
module Control = Mgen.Control
module Pink = Mgen.Pink

let usage () : 'a =
  prerr_endline
    "usage: play_pink [--device PATH] [--seed N] [--notes N] [--step-ms N] [--gate-ms N] \
     [--velocity N] [--channel N] [--rows N] [--stretch N] [--root N]";
  exit 2
;;

type options =
  { device : string option
  ; seed : int
  ; notes : int
  ; step_ms : int
  ; gate_ms : int
  ; velocity : int
  ; channel : int
  ; params : Pink.Params.t
  }

let defaults =
  { device = None
  ; seed = Control.Default.seed
  ; notes = 64
  ; step_ms = Control.Default.step_ms
  ; gate_ms = Control.Default.gate_ms
  ; velocity = Control.Default.velocity
  ; channel = Control.Default.channel
  ; params = Pink.Params.default
  }
;;

let int_arg s =
  match Option.try_with (fun () -> Int.of_string s) with
  | Some v -> v
  | None -> usage ()
;;

let rec parse options = function
  | [] -> options
  | "--device" :: v :: rest -> parse { options with device = Some v } rest
  | "--seed" :: v :: rest -> parse { options with seed = int_arg v } rest
  | "--notes" :: v :: rest -> parse { options with notes = int_arg v } rest
  | "--step-ms" :: v :: rest -> parse { options with step_ms = int_arg v } rest
  | "--gate-ms" :: v :: rest -> parse { options with gate_ms = int_arg v } rest
  | "--velocity" :: v :: rest -> parse { options with velocity = int_arg v } rest
  | "--channel" :: v :: rest -> parse { options with channel = int_arg v } rest
  | "--rows" :: v :: rest ->
    parse
      { options with params = { options.params with Pink.Params.rows = int_arg v } }
      rest
  | "--stretch" :: v :: rest ->
    parse
      { options with params = { options.params with Pink.Params.stretch = int_arg v } }
      rest
  | "--root" :: v :: rest ->
    parse
      { options with params = { options.params with Pink.Params.root = int_arg v } }
      rest
  | _ -> usage ()
;;

(* the one rawmidi device is the S-1; more than one needs --device *)
let find_device () =
  let dir = Core_unix.opendir "/dev/snd" in
  let rec entries acc =
    match Core_unix.readdir_opt dir with
    | None -> acc
    | Some name ->
      entries
        (if String.is_prefix name ~prefix:"midiC"
         then ("/dev/snd/" ^ name) :: acc
         else acc)
  in
  let found = entries [] in
  Core_unix.closedir dir;
  match found with
  | [ device ] -> device
  | [] ->
    prerr_endline "no rawmidi device in /dev/snd: is the synthesizer on?";
    exit 1
  | _ ->
    Printf.eprintf
      "more than one rawmidi device, give --device: %s\n"
      (String.concat ~sep:", " (List.sort found ~compare:String.compare));
    exit 1
;;

let send fd bytes =
  let buf = Bytes.of_char_list (List.map bytes ~f:Char.of_int_exn) in
  ignore (Core_unix.write fd ~buf : int)
;;

let note_on fd ~channel ~note ~velocity = send fd [ 0x90 lor channel; note; velocity ]
let note_off fd ~channel ~note = send fd [ 0x80 lor channel; note; 0x40 ]
let sleep_ms ms = ignore (Core_unix.nanosleep (Float.of_int ms /. 1000.) : float)

(* the note now on the line, so that an interrupt can silence it before the exit: state at
   the process edge *)
let sounding : int option ref = ref None

let play options fd =
  let gate = Int.min options.gate_ms options.step_ms in
  let pause = options.step_ms - gate in
  let rec loop model remaining =
    if remaining <> 0
    then (
      let model, note = Pink.next_note model in
      note_on fd ~channel:options.channel ~note ~velocity:options.velocity;
      sounding := Some note;
      sleep_ms gate;
      note_off fd ~channel:options.channel ~note;
      sounding := None;
      sleep_ms pause;
      loop model (remaining - 1))
  in
  let model =
    try Pink.create options.params ~seed:options.seed with
    | Invalid_argument message ->
      prerr_endline message;
      exit 1
  in
  (* Ctrl-C raises [Break] here, in the main flow, and the handler below silences the note
     that is on: no signal handler and no shared state *)
  Stdlib.Sys.catch_break true;
  try loop model (if options.notes <= 0 then -1 else options.notes) with
  | Stdlib.Sys.Break ->
    Option.iter !sounding ~f:(fun note -> note_off fd ~channel:options.channel ~note);
    exit 130
;;

let () =
  let options = parse defaults (List.tl_exn (Array.to_list (Sys.get_argv ()))) in
  if options.channel < 0
     || options.channel > 15
     || options.velocity < 1
     || options.velocity > 127
     || options.step_ms < 1
     || options.gate_ms < 0
  then (
    prerr_endline
      "the channel is 0 to 15, the velocity 1 to 127, step-ms at least 1, gate-ms at \
       least 0";
    exit 1);
  let device =
    match options.device with
    | Some device -> device
    | None -> find_device ()
  in
  let fd =
    try Core_unix.openfile device ~mode:[ O_WRONLY ] with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" device (Core_unix.Error.message error);
      exit 1
  in
  let { Pink.Params.rows; root; stretch; degrees = _; scale = _ } = options.params in
  Printf.printf
    "%s: seed %d, rows %d, stretch %d, root %d, step %d ms, gate %d ms, velocity %d, \
     channel %d, notes %d\n\
     %!"
    device
    options.seed
    rows
    stretch
    root
    options.step_ms
    options.gate_ms
    options.velocity
    options.channel
    options.notes;
  play options fd
;;
