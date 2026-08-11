(* play_pink: the pink model plays on the synthesizer over USB MIDI. Four voices from the
   row groups: the bass drifts, the tenor moves, the alto walks, the soprano dances. The
   FPGA is not in this loop, and the defaults of the run come from the control cells, thus
   the bare tool previews the model at the power-on values. *)

open Core
module Control_intf = Mgen_core.Control_intf
module Midi = Mgen_core.Midi
module Pink = Mgen_pink.Pink
module Player = Mgen_pink.Player

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

let play ~device ~seed ~steps ~step_ms ~gate_ms ~channel ~velocity ~hold =
  let fd =
    let path = Option.value device ~default:default_device in
    try Midi.open_device path with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" path (Core_unix.Error.message error);
      exit 1
  in
  (* -hold makes every voice speak only when it moves *)
  let model =
    if hold
    then
      { Pink.default with
        voices =
          List.map Pink.default.voices ~f:(fun v ->
            { v with Pink.Voice.restrike = false })
      }
    else Pink.default
  in
  let gate = Int.min gate_ms step_ms in
  let player = ref (Player.create ~model ~seed) in
  let advance f =
    let player', events = f !player in
    player := player';
    List.iter events ~f:(function
      | Player.Event.On note -> Midi.send_note_on fd ~channel ~note ~velocity
      | Player.Event.Off note -> Midi.send_note_off fd ~channel ~note)
  in
  Stdlib.Sys.catch_break true;
  (try
     let step = ref 0 in
     while steps = 0 || !step < steps do
       Int.incr step;
       advance Player.step;
       sleep_ms gate;
       (* the gate closes the highest voice only. When the gate is not less than the step
          it never comes, and that voice closes at its next articulation. *)
       if gate_ms < step_ms then advance Player.gate;
       sleep_ms (step_ms - gate)
     done
   with
   | Stdlib.Sys.Break ->
     advance Player.stop;
     exit 130);
  advance Player.stop
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
     and gate_ms =
       flag
         "-gate-ms"
         (optional_with_default Control_intf.Default.gate_ms int)
         ~doc:"MS the soprano gate time"
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
     and hold =
       flag
         "-hold"
         no_arg
         ~doc:" every voice re-strikes only at a pitch change, not only the low ones"
     in
     fun () -> play ~device ~seed ~steps ~step_ms ~gate_ms ~channel ~velocity ~hold)
;;

let () = Command_unix.run command
