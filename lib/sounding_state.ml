open Core

type t =
  { sounding : Set.M(Int).t
  ; last_on : int option
  }

let silence = { sounding = Set.empty (module Int); last_on = None }

let is_legal t (token : Token.t) =
  match token with
  | Start -> false
  | On pitch ->
    (not (Set.mem t.sounding pitch))
    && Set.length t.sounding < Token.seats
    &&
      (match t.last_on with
      | None -> true
      | Some last -> pitch > last)
  | Off pitch -> Set.mem t.sounding pitch && Option.is_none t.last_on
  | End -> true
;;

let step t (token : Token.t) =
  match token with
  | Start -> t
  | On pitch -> { sounding = Set.add t.sounding pitch; last_on = Some pitch }
  | Off pitch -> { t with sounding = Set.remove t.sounding pitch }
  | End -> { t with last_on = None }
;;

let legal_mask t = Array.init Token.vocab ~f:(fun code -> is_legal t (Token.of_byte code))

let%expect_test "the legal mask enforces the sentence rules" =
  let walk tokens = List.fold tokens ~init:silence ~f:step in
  let check state token =
    printf "%-10s %b\n" (Sexp.to_string (Token.sexp_of_t token)) (is_legal state token)
  in
  (* inside a sentence, after two ONs *)
  let state = walk [ Token.On 60; On 64 ] in
  check state (On 64);
  check state (On 62);
  check state (On 67);
  check state (Off 60);
  check state End;
  [%expect
    {|
    (On 64)    false
    (On 62)    false
    (On 67)    true
    (Off 60)   false
    End        true
    |}];
  (* the next sentence: the OFFs open again, the chord still sounds *)
  let state = walk [ Token.On 60; On 64; End ] in
  check state (Off 64);
  check state (On 60);
  check state (On 67);
  [%expect {|
    (Off 64)   true
    (On 60)    false
    (On 67)    true
    |}];
  (* the seats are full *)
  let state = walk [ Token.On 60; On 64; On 67; On 71 ] in
  check state (On 72);
  check state End;
  [%expect {|
    (On 72)    false
    End        true
    |}]
;;

let%expect_test "START never, and an OFF needs a sounding pitch" =
  let state = List.fold [ Token.On 60; On 64 ] ~init:silence ~f:step in
  let mask = legal_mask state in
  let show token =
    printf
      "%-10s %b\n"
      (Sexp.to_string (Token.sexp_of_t token))
      mask.(Token.to_byte token)
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
