open Core

type t =
  { sounding : Set.M(Int).t
  ; last_on : int option
  ; last_off : int option
  }

let silence = { sounding = Set.empty (module Int); last_on = None; last_off = None }

(* The first ON of a sentence opens the run and the rest fall below it. The fall is the
   melody leading: the top voice is chosen before the voices under it and conditions on
   none of them. *)
let below_last_on t pitch =
  match t.last_on with
  | None -> true
  | Some last -> pitch < last
;;

(* The OFFs climb, thus the two runs meet in the middle: the release of the top moving
   voice sits beside its attack, and one melodic step is two adjacent tokens. *)
let above_last_off t pitch =
  match t.last_off with
  | None -> true
  | Some last -> pitch > last
;;

let is_legal t (token : Token.t) =
  match token with
  | Start -> false
  | On pitch ->
    (not (Set.mem t.sounding pitch))
    && Set.length t.sounding < Token.seats
    && below_last_on t pitch
  | Off pitch ->
    Set.mem t.sounding pitch && Option.is_none t.last_on && above_last_off t pitch
  | End -> true
;;

let step t (token : Token.t) =
  match token with
  | Start -> t
  | On pitch -> { t with sounding = Set.add t.sounding pitch; last_on = Some pitch }
  | Off pitch -> { t with sounding = Set.remove t.sounding pitch; last_off = Some pitch }
  | End -> { t with last_on = None; last_off = None }
;;

let legal_mask t = Array.init Token.vocab ~f:(fun code -> is_legal t (Token.of_code code))

let%expect_test "the legal mask enforces the sentence rules" =
  let walk tokens = List.fold tokens ~init:silence ~f:step in
  let check state token =
    printf "%-10s %b\n" (Sexp.to_string (Token.sexp_of_t token)) (is_legal state token)
  in
  (* inside a sentence, after two ONs: the run falls, thus 64 is the ceiling of the rest *)
  let state = walk [ Token.On 67; On 64 ] in
  check state (On 60);
  check state (On 65);
  check state (On 64);
  check state (Off 67);
  check state End;
  [%expect
    {|
    (On 60)    true
    (On 65)    false
    (On 64)    false
    (Off 67)   false
    End        true
    |}];
  (* the next sentence: the OFFs open again, and they climb *)
  let state = walk [ Token.On 67; On 64; End ] in
  check state (Off 64);
  check state (Off 67);
  check state (On 71);
  [%expect {|
    (Off 64)   true
    (Off 67)   true
    (On 71)    true
    |}];
  (* after the first OFF, only an OFF above it *)
  let state = walk [ Token.On 67; On 64; End; Off 64 ] in
  check state (Off 67);
  [%expect {| (Off 67)   true |}];
  let state = walk [ Token.On 67; On 64; End; Off 67 ] in
  check state (Off 64);
  [%expect {| (Off 64)   false |}];
  (* the seats are full: 55 keeps the fall and the mask still refuses it *)
  let state = walk [ Token.On 71; On 67; On 64; On 60 ] in
  check state (On 55);
  check state End;
  [%expect {|
    (On 55)    false
    End        true
    |}]
;;

let%expect_test "START never, and an OFF needs a sounding pitch" =
  (* the sentence closes, thus the refusal below rests on the silent pitch alone *)
  let state = List.fold [ Token.On 64; On 60; End ] ~init:silence ~f:step in
  let mask = legal_mask state in
  let show token =
    printf
      "%-10s %b\n"
      (Sexp.to_string (Token.sexp_of_t token))
      mask.(Token.to_code token)
  in
  (* START is input only: the model never draws it *)
  show Start;
  (* an OFF of a silent pitch is a wire no-op, and the grammar still refuses it *)
  show (Off 50);
  [%expect {|
    Start      false
    (Off 50)   false
    |}]
;;
