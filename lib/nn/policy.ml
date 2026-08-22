(* The sampling policy — see policy.mli for the contract. *)

open Base

let tempered raw ~temperature =
  let peak = Array.fold raw ~init:Float.neg_infinity ~f:Float.max in
  Array.map raw ~f:(fun value -> Float.exp ((value -. peak) /. temperature))
;;

let above_min_p weights ~min_p =
  if Float.(min_p <= 0.0)
  then weights
  else
    Array.map weights ~f:(fun weight -> if Float.(weight >= min_p) then weight else 0.0)
;;

(* the running totals of the weights, left to right; the last of them is the total *)
let running_totals weights =
  Array.folding_map weights ~init:0.0 ~f:(fun total weight ->
    let total = total +. weight in
    total, total)
;;

(* The class whose running total passes the draw.

   It takes the uniform and not a draw, thus one function owns both sums and the total is
   the last running total — never a second sum of the same weights. Two sums of one array
   differ in the last bits, and a draw made against the other sum can land above every
   running total, where no class passes at all. That case is real in the integer twins,
   which add pairwise in [sum] and left to right in the running totals.

   Against this total the draw is strictly below it: the uniform falls under 1 by 2 ** -24
   at the least, thus the exact product falls short by about 2 ** 29 units in the last
   place, where rounding moves a result by half of one. Therefore the walk always ends on
   a class, and that class always holds weight — to reach the last index is to know that
   no earlier total passed, thus the weight there is the difference of two totals across
   the draw. No fallback is necessary, and none is written. *)
let pick weights ~uniform =
  let running = running_totals weights in
  let last = Array.length running - 1 in
  let draw = uniform *. running.(last) in
  let rec walk index =
    if index = last || Float.(running.(index) > draw) then index else walk (index + 1)
  in
  walk 0
;;

let draw_class raw ~temperature ~min_p ~uniform =
  pick (above_min_p (tempered raw ~temperature) ~min_p) ~uniform
;;

(* The bounds of the draw. The integer twins state the same two, thus one module owns them
   and a reader finds one message for each. *)
let check_policy ~temperature ~min_p =
  if Float.(temperature <= 0.0) then invalid_arg "the temperature is positive";
  if Float.(min_p < 0.0 || min_p >= 1.0) then invalid_arg "min_p is 0 up to 1"
;;

let elected_temperature = 1.0
let elected_min_p = 0.05

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

(* the widest row the eras draw over — the vocabulary of the corpus is one case of this
   pipeline and not a rule of it, thus the gate states a width and no corpus enters *)
let widest_row = 48

(* Both integer twins draw through these rules and never through a copy of them, thus a
   rule that moved here would move the circuits' walks with it. The prose of [pick] argues
   that the walk always ends on a class that holds weight; this fuzz measures it. *)
let%expect_test "the fuzz: a pick lands on a class that holds weight" =
  (* The seed is fixed, thus the gate is the same gate on every machine, and the report is
     a verdict: a case that lands on a zero prints itself.

     [pick] takes the weights a draw gives it, thus one class always holds weight — the
     tempered peak weighs one and a floor under one keeps it. An array of zeros is not a
     case of this function, and no floor can make one. *)
  let state = Random.State.make [| 20260819 |] in
  let drawn_weights (_ : int) =
    let classes = Random.State.int_incl state 1 widest_row in
    let raw =
      Array.init classes ~f:(fun (_ : int) -> Random.State.float_range state (-20.) 20.)
    in
    let temperature = Random.State.float_range state 0.5 1.5 in
    (* a floor high enough to zero most of a wide row, thus the walk crosses long runs of
       refused classes *)
    let min_p = Random.State.float_range state 0.0 0.9 in
    above_min_p (tempered raw ~temperature) ~min_p
  in
  (* the shapes a draw makes rarely: the one class, and the weight at each end of the row *)
  let edges = [ [| 1.0 |]; [| 1.0; 0.0 |]; [| 0.0; 1.0 |]; [| 0.0; 0.0; 1.0; 0.0 |] ] in
  let cases = edges @ List.map (List.range 0 200) ~f:drawn_weights in
  (* the two ends of the grid of the 24-bit uniform and one draw between them: at the top
     end a total summed a second time would send the walk past every running total *)
  let uniforms = [ 0.0; 0.5; Float.of_int 0xFFFFFF *. 0x1p-24 ] in
  let fault weights =
    List.find_map uniforms ~f:(fun uniform ->
      let index = pick weights ~uniform in
      if Float.(weights.(index) > 0.0) then None else Some (Array.length weights, index))
  in
  (match List.filter_map cases ~f:fault with
   | [] ->
     Stdio.printf
       "%d weight rows over %d uniforms: every pick holds weight\n"
       (List.length cases)
       (List.length uniforms)
   | (classes, index) :: (_ : (int * int) list) ->
     Stdio.printf "a row of %d classes picked %d, which holds no weight\n" classes index);
  [%expect {| 204 weight rows over 3 uniforms: every pick holds weight |}]
;;

let%expect_test "the min-p floor: 0 removes nothing, and the peak stands under any floor" =
  let state = Random.State.make [| 20260819 |] in
  let raw (_ : int) =
    Array.init widest_row ~f:(fun (_ : int) -> Random.State.float_range state (-20.) 20.)
  in
  let rows = List.map (List.range 0 100) ~f:raw in
  let floors = [ 0.0; elected_min_p; 0.5; 0.999 ] in
  let removes_nothing row =
    let weights = tempered row ~temperature:1.0 in
    Array.equal Float.equal (above_min_p weights ~min_p:0.0) weights
  in
  (* The tempered peak weighs exactly one, thus every floor under one keeps it. This is
     why a draw always exists and why one compare holds the whole filter. *)
  let peak_stands row =
    let weights = tempered row ~temperature:1.0 in
    List.for_all floors ~f:(fun min_p ->
      Array.exists (above_min_p weights ~min_p) ~f:(fun w -> Float.(w = 1.0)))
  in
  Stdio.printf
    "%d rows: the floor 0 removes nothing: %b, the peak stands under every floor: %b\n"
    (List.length rows)
    (List.for_all rows ~f:removes_nothing)
    (List.for_all rows ~f:peak_stands);
  (* A class ten nats under the peak weighs 4.5e-5 of it. With no floor the walk reaches
     it at the top of the grid; the elected floor removes it, and no uniform can name it. *)
  let far_under = [| 0.0; -10.0 |] in
  let draw ~min_p ~uniform = draw_class far_under ~temperature:1.0 ~min_p ~uniform in
  let top = Float.of_int 0xFFFFFF *. 0x1p-24 in
  Stdio.printf
    "no floor: %d at 0.5, %d at the top of the grid\n"
    (draw ~min_p:0.0 ~uniform:0.5)
    (draw ~min_p:0.0 ~uniform:top);
  Stdio.printf
    "the elected floor: %d at 0.5, %d at the top of the grid\n"
    (draw ~min_p:elected_min_p ~uniform:0.5)
    (draw ~min_p:elected_min_p ~uniform:top);
  [%expect
    {|
    100 rows: the floor 0 removes nothing: true, the peak stands under every floor: true
    no floor: 0 at 0.5, 1 at the top of the grid
    the elected floor: 0 at 0.5, 0 at the top of the grid
    |}]
;;

let%expect_test "the policy bounds: one message for each" =
  let policy ~temperature ~min_p =
    Checkpoint.refusal (fun () -> check_policy ~temperature ~min_p)
  in
  Stdio.printf "temperature 0: %s\n" (policy ~temperature:0.0 ~min_p:elected_min_p);
  Stdio.printf "min_p 1: %s\n" (policy ~temperature:elected_temperature ~min_p:1.0);
  Stdio.printf
    "min_p below 0: %s\n"
    (policy ~temperature:elected_temperature ~min_p:(-0.1));
  Stdio.printf
    "the elected policy: %s\n"
    (policy ~temperature:elected_temperature ~min_p:elected_min_p);
  [%expect
    {|
    temperature 0: the temperature is positive
    min_p 1: min_p is 0 up to 1
    min_p below 0: min_p is 0 up to 1
    the elected policy: no raise
    |}]
;;
