(* The shared half of the era players — see player.mli for the contract and the design.
   The step line is a gate contract: jax/midi.py prints the same characters. *)

(* [Core] carries a [Signal] of its own — the POSIX numbers — thus the stop request of
   this library takes a second name here, before the open shadows the first. *)
module Stop = Signal
open Core

let default_device = "/dev/snd/midiC2D0"

(* These flags carry the value of a control cell, thus [Control_intf.Reg] states their
   range and a player does not repeat it. A value outside the range either raises out of
   the library — [Prng.create] for the seed, [Char.of_int_exn] for a velocity above a byte
   — or makes a wrong MIDI status byte, and neither is a diagnostic for a person at a
   command line. The check is on the argument, thus it comes before the tool opens the
   device. *)
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

(* the events of one step, as the gate reads them: a step that moves nothing is a dash and
   never an empty tail, thus every line carries a field there *)
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

let step_line ~step events = sprintf "step %3d  %s" step (show_events events)
let print_step ~step events = printf "%s\n%!" (step_line ~step events)

let open_or_die path =
  try Midi.open_device path with
  | Core_unix.Unix_error (error, _, _) ->
    Printf.eprintf "cannot open %s: %s\n" path (Core_unix.Error.message error);
    exit 1
;;

(* The set of sounding pitches is the state the drain needs, thus it stands where the
   [finally] can read it: an exception or a stop leaves through there, and a handler that
   cannot see the set releases nothing. *)
let play_piece fd stopped music ~step_ms ~channel ~velocity_at =
  let sounding = ref (Set.empty (module Int)) in
  Exn.protect
    ~f:(fun () ->
      With_return.with_return (fun { return } ->
        List.iteri music ~f:(fun step events ->
          if Stop.stop_requested stopped then return ();
          print_step ~step events;
          let struck = velocity_at ~step in
          List.iter events ~f:(function
            | Frame.Event.On note ->
              Midi.send_note_on fd ~channel ~note ~velocity:struck;
              sounding := Set.add !sounding note
            | Frame.Event.Off note ->
              Midi.send_note_off fd ~channel ~note;
              sounding := Set.remove !sounding note);
          sleep_ms step_ms)))
    ~finally:(fun () ->
      (* the drain: the piece ends with its chord still sounding, as a chorale does, and
         the sequencer of the board releases the same way at a stop *)
      Set.iter !sounding ~f:(fun note -> Midi.send_note_off fd ~channel ~note))
;;

let play music ~device ~step_ms ~channel ~velocity =
  let fd = open_or_die device in
  (* Ctrl-C must not leave a chord ringing on the synthesizer; [Signal] holds the rule and
     the measurement behind it. The drain of [play_piece] is the road out. *)
  let stopped = Stop.watch_stop_play () in
  play_piece fd stopped music ~step_ms ~channel ~velocity_at:(fun ~step:(_ : int) ->
    velocity);
  Stop.exit_if_stopped stopped
;;

(* The gate reads these lines as text, thus the format is pinned here and not only at the
   seam. The event order is the business of [Frame.events_of_frames] — the releases of a
   step before its strikes — and this pins the join. *)
let%expect_test "the step line: a step that moves nothing, and a step of offs and ons" =
  let show ~step events = print_endline (step_line ~step events) in
  show ~step:0 [];
  show ~step:7 [ Frame.Event.On 60 ];
  show ~step:16 [ Frame.Event.Off 55; Frame.Event.Off 72; Frame.Event.On 57 ];
  show ~step:128 [ Frame.Event.On 48 ];
  [%expect
    {|
    step   0  -
    step   7  on:60
    step  16  off:55 off:72 on:57
    step 128  on:48
    |}]
;;
