(* The draws a gate takes — see fuzz.mli. The uniform of [Prng] stands on a 24-bit grid,
   thus a draw is that grid mapped onto the range and never a modulo of a byte. *)

open Core

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
