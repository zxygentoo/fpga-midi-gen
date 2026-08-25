open Core
open Hardcaml

let max_message_bytes = 3
let note_on = 0x90
let note_off = 0x80
let release_velocity = 0x40

(* the bytes of a channel voice message: the status low nibble carries the channel *)
let note_on_bytes ~channel ~note ~velocity = [ note_on lor channel; note; velocity ]
let note_off_bytes ~channel ~note = [ note_off lor channel; note; release_velocity ]
let open_device path = Core_unix.openfile path ~mode:[ O_WRONLY ]

let send fd bytes =
  let buf = Bytes.of_char_list (List.map bytes ~f:Char.of_int_exn) in
  ignore (Core_unix.write fd ~buf : int)
;;

let send_note_on fd ~channel ~note ~velocity =
  send fd (note_on_bytes ~channel ~note ~velocity)
;;

let send_note_off fd ~channel ~note = send fd (note_off_bytes ~channel ~note)

(* The share of its velocity a note keeps at the last step of a fade. It is not zero, and
   not because a fade should end loud: A NOTE-ON OF VELOCITY ZERO IS A NOTE-OFF on the
   wire, and a note the fade silenced would never be released. A quarter is about twelve
   decibels under the full stroke, which the ear reads as an ending and the synth still
   sounds. The rule is the twin of jax/midi.py's, number for number. *)
let fade_floor = 0.25

let fading ~step ~steps ~fade =
  (* a canvas shorter than the window fades across the whole of itself; without the clamp
     it would OPEN partway down the ramp and never sound its full stroke *)
  let fade = min fade steps in
  let left = steps - step in
  if fade <= 0 || left > fade
  then 1.0
  else
    fade_floor
    +. ((1.0 -. fade_floor) *. Float.of_int (left - 1) /. Float.of_int (max (fade - 1) 1))
;;

let faded_velocity ~velocity ~step ~steps ~fade =
  max 1 (Float.iround_nearest_exn (Float.of_int velocity *. fading ~step ~steps ~fade))
;;

let%expect_test "the fade leaves the music before it alone" =
  (* the gesture is the last bar of a canvas and nothing else: a step outside the window
     keeps the whole stroke, and the window opens at the step whose distance from the end
     is the fade itself *)
  let at step = fading ~step ~steps:128 ~fade:16 in
  printf
    "%b %b %b %b\n"
    Float.(at 0 = 1.0)
    Float.(at 111 = 1.0)
    Float.(at 112 = 1.0)
    Float.(at 113 < 1.0);
  [%expect {| true true true true |}]
;;

let%expect_test "the fade falls to the floor and never below it" =
  (* A NOTE-ON OF VELOCITY ZERO IS A NOTE-OFF: a fade that reached zero would state a
     release the player never made, and the note would keep sounding until the drain *)
  printf "%b\n" Float.(fading ~step:127 ~steps:128 ~fade:16 = fade_floor);
  let struck =
    List.map (List.range 112 128) ~f:(fun step ->
      faded_velocity ~velocity:100 ~step ~steps:128 ~fade:16)
  in
  printf
    "%d down to %d, none under 1: %b\n"
    (List.hd_exn struck)
    (List.last_exn struck)
    (List.for_all struck ~f:(fun v -> v >= 1));
  [%expect {|
    true
    100 down to 25, none under 1: true
    |}]
;;

let%expect_test "the fade only falls, none is the full stroke, and a short canvas ramps \
                 whole"
  =
  let falling steps fade =
    let scale =
      List.map (List.range 0 steps) ~f:(fun step -> fading ~step ~steps ~fade)
    in
    List.for_all2_exn (List.drop_last_exn scale) (List.tl_exn scale) ~f:(fun a b ->
      Float.(b <= a))
  in
  (* a diminuendo that rose anywhere would read as a phrase and not as an ending *)
  printf "only falls: %b\n" (falling 128 16);
  (* -fade 0 must be the player as it stood, and not a fade of one step *)
  printf
    "no fade is the full stroke: %b\n"
    (List.for_all (List.range 0 128) ~f:(fun step ->
       Float.(fading ~step ~steps:128 ~fade:0 = 1.0)));
  (* the window is the last [fade] steps or the whole canvas, whichever is shorter; a walk
     of four steps must not divide by a fade of sixteen *)
  printf
    "a short canvas opens whole and ends on the floor: %b %b %b\n"
    Float.(fading ~step:0 ~steps:4 ~fade:16 = 1.0)
    Float.(fading ~step:3 ~steps:4 ~fade:16 = fade_floor)
    (falling 4 16);
  [%expect
    {|
    only falls: true
    no fade is the full stroke: true
    a short canvas opens whole and ends on the floor: true true true
    |}]
;;

module Rtl = struct
  open Signal

  module Message = struct
    type 'a t =
      { data : 'a [@bits max_message_bytes * 8]
      ; len : 'a [@bits 8]
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  (* the same layout as [note_on_bytes] and [note_off_bytes], in the order of
     [Message.data]: the first byte is in the low 8 bits *)
  let channel_voice_data ~status ~channel ~data1 ~data2 =
    concat_lsb
      [ concat_msb [ of_unsigned_int ~width:4 (status lsr 4); channel ]; data1; data2 ]
  ;;

  let note_on_data ~channel ~pitch ~velocity =
    channel_voice_data ~status:note_on ~channel ~data1:pitch ~data2:velocity
  ;;

  let note_off_data ~channel ~pitch =
    channel_voice_data
      ~status:note_off
      ~channel
      ~data1:pitch
      ~data2:(of_unsigned_int ~width:8 release_velocity)
  ;;
end
