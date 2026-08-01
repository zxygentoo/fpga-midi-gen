(* play_voices: the register-decomposition experiment plays on the synthesizer over USB
   MIDI. Three voices from the row groups of the pink model: the bass drifts, the middle
   walks, the treble dances. The FPGA is not in this loop. *)

open Core
module Control = Mgen.Control
module Midi = Mgen.Midi
module Pink_voices = Mgen.Pink_voices

let sleep_ms ms = ignore (Core_unix.nanosleep (Float.of_int ms /. 1000.) : float)

(* the open note of each voice, so that an interrupt and the exit can silence them: state
   at the process edge *)
let sounding : int option array = Array.create ~len:Pink_voices.voices None

let silence fd ~channel =
  Array.iteri sounding ~f:(fun k note ->
    Option.iter note ~f:(fun note -> Midi.Host.note_off fd ~channel ~note);
    sounding.(k) <- None)
;;

let default_device = "/dev/snd/midiC2D0"

let play ~device ~seed ~steps ~step_ms ~gate_ms ~channel ~velocity ~hold =
  let fd =
    let path = Option.value device ~default:default_device in
    try Midi.Host.open_device path with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" path (Core_unix.Error.message error);
      exit 1
  in
  let gate = Int.min gate_ms step_ms in
  let model = ref (Pink_voices.create ~seed) in
  Stdlib.Sys.catch_break true;
  (try
     let step = ref 0 in
     while steps = 0 || !step < steps do
       Int.incr step;
       let model', states = Pink_voices.next_step !model in
       model := model';
       (* strike from the bass upward; each due voice closes its note first, thus at most
          three of the four S-1 voices are ever held *)
       List.iteri states ~f:(fun k (s : Pink_voices.state) ->
         let restrike =
           s.due && ((not hold) || not (Option.equal Int.equal sounding.(k) (Some s.note)))
         in
         if restrike
         then (
           Option.iter sounding.(k) ~f:(fun note -> Midi.Host.note_off fd ~channel ~note);
           Midi.Host.note_on fd ~channel ~note:s.note ~velocity;
           sounding.(k) <- Some s.note));
       sleep_ms gate;
       (* the treble — the last voice — is the gated one; the bass and the middle sustain
          to their next articulation *)
       let treble = Pink_voices.voices - 1 in
       Option.iter sounding.(treble) ~f:(fun note ->
         Midi.Host.note_off fd ~channel ~note;
         sounding.(treble) <- None);
       sleep_ms (step_ms - gate)
     done
   with
   | Stdlib.Sys.Break ->
     silence fd ~channel;
     exit 130);
  silence fd ~channel
;;

let command =
  Command.basic
    ~summary:"play the register-decomposition experiment on the S-1 over USB MIDI"
    (let%map_open.Command device =
       flag
         "-device"
         (optional string)
         ~doc:"PATH the rawmidi device (default /dev/snd/midiC2D0)"
     and seed =
       flag
         "-seed"
         (optional_with_default Control.Default.seed int)
         ~doc:"N the PRNG seed, 32 bits, not 0"
     and steps =
       flag
         "-steps"
         (optional_with_default 128 int)
         ~doc:"N steps to play; 0 plays with no end"
     and step_ms =
       flag
         "-step-ms"
         (optional_with_default Control.Default.step_ms int)
         ~doc:"MS the step period"
     and gate_ms =
       flag
         "-gate-ms"
         (optional_with_default Control.Default.gate_ms int)
         ~doc:"MS the treble gate time"
     and channel =
       flag
         "-channel"
         (optional_with_default Control.Default.channel int)
         ~doc:"N MIDI channel 0 to 15 (default: the S-1 channel 3)"
     and velocity =
       flag
         "-velocity"
         (optional_with_default Control.Default.velocity int)
         ~doc:"N note velocity 1 to 127"
     and hold =
       flag "-hold" no_arg ~doc:" re-strike a voice only when its pitch changes"
     in
     fun () -> play ~device ~seed ~steps ~step_ms ~gate_ms ~channel ~velocity ~hold)
;;

let () = Command_unix.run command
