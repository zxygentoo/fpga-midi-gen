(* What the gates of era six share — see fuzz.mli. The uniform of [Prng] stands on a
   24-bit grid, thus a draw is that grid mapped onto the range and never a modulo of a
   byte. *)

open Core
module Bits = Hardcaml.Bits

let draw state ~limit =
  let state, u = Prng.run Prng.uniform state in
  state, Int.of_float (u *. Float.of_int ((2 * limit) + 1)) - limit
;;

let draw_between state ~low ~high =
  let state, u = Prng.run Prng.uniform state in
  state, low + Int.of_float (u *. Float.of_int (high - low + 1))
;;

let draw_array state ~len ~limit =
  Array.fold_map (Array.create ~len 0) ~init:state ~f:(fun state (_ : int) ->
    draw state ~limit)
;;

let pack values ~width =
  Bits.concat_lsb (List.map (Array.to_list values) ~f:(Bits.of_signed_int ~width))
;;

let set port value = port := Bits.of_unsigned_int ~width:(Bits.width !port) value
