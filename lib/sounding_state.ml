open Core

type t =
  { sounding : Set.M(Int).t
  ; last_on : int option
  }

let silence = { sounding = Set.empty (module Int); last_on = None }

let is_legal t (token : Token.t) =
  match token with
  | End -> true
  | Off pitch -> Set.mem t.sounding pitch && Option.is_none t.last_on
  | On pitch ->
    (not (Set.mem t.sounding pitch))
    && Set.length t.sounding < Token.seats
    &&
      (match t.last_on with
      | None -> true
      | Some last -> pitch > last)
;;

let is_safe t (token : Token.t) =
  match token with
  | End -> true
  | Off _ -> true
  | On pitch -> (not (Set.mem t.sounding pitch)) && Set.length t.sounding < Token.seats
;;

let step t (token : Token.t) =
  match token with
  | End -> { t with last_on = None }
  | Off pitch -> { t with sounding = Set.remove t.sounding pitch }
  | On pitch -> { sounding = Set.add t.sounding pitch; last_on = Some pitch }
;;

let legal_mask t = Array.init Token.vocab ~f:(fun code -> is_legal t (Token.of_byte code))
let safe_mask t = Array.init Token.vocab ~f:(fun code -> is_safe t (Token.of_byte code))

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

let%expect_test "the safety floor against the grammar" =
  let state = List.fold [ Token.On 60; On 64 ] ~init:silence ~f:step in
  let grammar = legal_mask state in
  let floor = safe_mask state in
  let show token =
    let code = Token.to_byte token in
    printf
      "%-10s grammar %-6b safe %b\n"
      (Sexp.to_string (Token.sexp_of_t token))
      grammar.(code)
      floor.(code)
  in
  (* the order rule is grammar; the synthesizer does not care *)
  show (Off 60);
  (* an OFF of a silent pitch is a wire no-op: only the grammar refuses *)
  show (Off 50);
  (* the cross-kill: both refuse *)
  show (On 60);
  (* below the last ON: convention, not damage *)
  show (On 62);
  [%expect
    {|
    (Off 60)   grammar false  safe true
    (Off 50)   grammar false  safe true
    (On 60)    grammar false  safe false
    (On 62)    grammar false  safe true
    |}]
;;
