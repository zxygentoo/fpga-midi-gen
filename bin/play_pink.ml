(* play_pink: the pink model plays on the synthesizer over USB MIDI. Four voices from the
   row groups: the bass drifts, the tenor moves, the alto walks, the soprano dances. The
   FPGA is not in this loop, and the defaults of the run come from the control cells, thus
   the bare tool previews the model at the power-on values. *)

open Core
module Control = Mgen.Control
module Midi = Mgen.Midi
module Pink = Mgen.Pink

let voices = List.length Pink.default_voices

(* bass first, aligned with the states of [Pink.next_step] *)
let restrikes =
  Array.of_list (List.rev_map Pink.default_voices ~f:(fun v -> v.Pink.Voice.restrike))
;;

let sleep_ms ms = ignore (Core_unix.nanosleep (Float.of_int ms /. 1000.) : float)

(* the open note of each voice, so that an interrupt and the exit can silence them: state
   at the process edge *)
let sounding : int option array = Array.create ~len:voices None

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
  let model = ref (Pink.create ~voices:Pink.default_voices ~seed) in
  Stdlib.Sys.catch_break true;
  (try
     let step = ref 0 in
     while steps = 0 || !step < steps do
       Int.incr step;
       let model', states = Pink.next_step !model in
       model := model';
       (* strike from the bass upward; each due voice closes its note first, thus the four
          voices of the S-1 are never exceeded *)
       List.iteri states ~f:(fun k (s : Pink.state) ->
         (* a due voice strikes when nothing is open, when its pitch moved, or when its
            policy re-strikes held pitches (and -hold does not override it) *)
         let restrike =
           s.due
           &&
           match sounding.(k) with
           | None -> true
           | Some open_note -> open_note <> s.note || (restrikes.(k) && not hold)
         in
         if restrike
         then (
           Option.iter sounding.(k) ~f:(fun note -> Midi.Host.note_off fd ~channel ~note);
           Midi.Host.note_on fd ~channel ~note:s.note ~velocity;
           sounding.(k) <- Some s.note));
       sleep_ms gate;
       (* the soprano — the last voice — is the gated one; the others sustain to their
          next articulation *)
       let soprano = voices - 1 in
       Option.iter sounding.(soprano) ~f:(fun note ->
         Midi.Host.note_off fd ~channel ~note;
         sounding.(soprano) <- None);
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
    ~summary:"play the pink model on the S-1 over USB MIDI"
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
         ~doc:"MS the soprano gate time"
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
       flag
         "-hold"
         no_arg
         ~doc:" every voice re-strikes only at a pitch change, not only the low ones"
     in
     fun () -> play ~device ~seed ~steps ~step_ms ~gate_ms ~channel ~velocity ~hold)
;;

let () = Command_unix.run command
