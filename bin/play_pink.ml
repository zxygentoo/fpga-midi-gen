(* play_pink: the pink model plays on the synthesizer over USB MIDI. Four voices from the
   row groups: the bass drifts, the tenor moves, the alto walks, the soprano dances. The
   FPGA is not in this loop, and the defaults of the run come from the control cells, thus
   the bare tool previews the model at the power-on values.

   One step of music is one frame, and [Frame.events_of_frames] states what a new frame
   does to the frame before it. Therefore this tool is a clock and a wire encoder, and it
   plays what the board plays: there is no gate and no re-strike policy, because a frame
   can state neither. *)

open Core
module Control_intf = Mgen_core.Control_intf
module Frame = Mgen_core.Frame
module Midi = Mgen_core.Midi
module Pink = Mgen_pink.Pink
module Signal = Mgen_core.Signal

(* These flags set a control cell, thus [Control_intf.Reg] states their range and this
   tool does not repeat it. A value outside the range either raises out of the library —
   [Prng.create] for the seed, [Char.of_int_exn] for a velocity above a byte — or makes a
   wrong MIDI status byte, and neither is a diagnostic for a person at a command line. The
   check is on the argument, thus it comes before the tool opens the device. *)
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

let seed_arg = ranged "the seed" Control_intf.Reg.seed
let channel_arg = ranged "the channel" Control_intf.Reg.channel
let velocity_arg = ranged "the velocity" Control_intf.Reg.velocity
let sleep_ms ms = ignore (Core_unix.nanosleep (Float.of_int ms /. 1000.) : float)
let default_device = "/dev/snd/midiC2D0"

(* the events of one step: the rule of [Frame] over the pair. The fold starts from
   silence, thus the second element is what [frame] does to the frame before it. *)
let events_after previous frame =
  List.last_exn (Frame.events_of_frames [| previous; frame |])
;;

let play ~device ~seed ~steps ~step_ms ~channel ~velocity =
  let fd =
    let path = Option.value device ~default:default_device in
    try Midi.open_device path with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" path (Core_unix.Error.message error);
      exit 1
  in
  (* the frame that sounds after the send: the walk carries it into the next step *)
  let send ~sounding frame =
    List.iter (events_after sounding frame) ~f:(function
      | Frame.Event.On note -> Midi.send_note_on fd ~channel ~note ~velocity
      | Frame.Event.Off note -> Midi.send_note_off fd ~channel ~note);
    frame
  in
  (* Ctrl-C must not leave the four voices ringing; [Signal] holds the rule and the
     measurement behind it. The silent frame below is the road out, and it is the rule the
     board keeps at the run stop. *)
  let stopped = Signal.watch_stop_play () in
  (* [steps] of 0 plays with no end, thus only the stop ends that walk *)
  let rec play_steps walk ~sounding ~step =
    if Signal.stop_requested stopped || (steps > 0 && step >= steps)
    then sounding
    else (
      let walk, frame = Pink.next_frame walk in
      let sounding = send ~sounding frame in
      sleep_ms step_ms;
      play_steps walk ~sounding ~step:(step + 1))
  in
  let sounding =
    play_steps (Pink.create ~model:Pink.default ~seed) ~sounding:Frame.silent ~step:0
  in
  let (_ : int) = send ~sounding Frame.silent in
  Signal.exit_if_stopped stopped
;;

let command =
  Command.basic
    ~summary:"play the pink model on the S-1 over USB MIDI"
    (let%map_open.Command device =
       flag
         "-device"
         (optional string)
         ~doc:"PATH the rawmidi device (default /dev/snd/midiC2D0)"
     and seed =
       flag
         "-seed"
         (optional_with_default Control_intf.Default.seed seed_arg)
         ~doc:"N the PRNG seed, 32 bits, not 0"
     and steps =
       flag
         "-steps"
         (optional_with_default 128 int)
         ~doc:"N steps to play; 0 plays with no end"
     and step_ms =
       flag
         "-step-ms"
         (optional_with_default Control_intf.Default.step_ms int)
         ~doc:"MS the step period"
     and channel =
       flag
         "-channel"
         (optional_with_default Control_intf.Default.channel channel_arg)
         ~doc:"N MIDI channel 0 to 15 (default: the S-1 channel 3)"
     and velocity =
       flag
         "-velocity"
         (optional_with_default Control_intf.Default.velocity velocity_arg)
         ~doc:"N note velocity 1 to 127"
     in
     fun () -> play ~device ~seed ~steps ~step_ms ~channel ~velocity)
;;

let () = Command_unix.run command
