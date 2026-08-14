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
    (* A pitch that sounds must be releasable, and [Off 0] has no code: code 0 is End.
       Therefore pitch 0 never starts, or its voice would hold for the rest of the walk
       with no token able to close it. [Jsb.escape_reserved] moves the same two pitches,
       thus the corpus never states one either. *)
    pitch <> 0
    && (not (Set.mem t.sounding pitch))
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
let sounding t = Set.to_list t.sounding

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

module Rtl = struct
  (* [open Signal] below shadows [t] with the signal type, thus the state of the grammar
     takes a name of its own for the block test. *)
  type sounding = t

  open Hardcaml
  open Signal

  (* The widths, from the encoding of [Token]: a code is one byte, its top bit is the type
     and the rest is the pitch. Therefore [Token.vocab] and [Token.seats] shape every
     register, and the two reserved codes come from the codec itself and cannot drift from
     it. [msb] of a code is the type bit, because the pitch is the rest of the byte. *)
  let code_bits = Int.ceil_log2 Token.vocab
  let pitch_bits = code_bits - 1
  let mask_bits = 1 lsl pitch_bits
  let count_bits = Int.ceil_log2 (Token.seats + 1)
  let start_code = Token.to_code Token.Start
  let end_code = Token.to_code Token.End

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; boot : 'a
      ; land_ : 'a
      ; code : 'a [@bits code_bits]
      ; query : 'a [@bits code_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { legal : 'a } [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let open Always in
    let mask = Variable.reg spec ~width:mask_bits in
    let count = Variable.reg spec ~width:count_bits in
    let last_on = Variable.reg spec ~width:pitch_bits in
    let lov = Variable.reg spec ~width:1 in
    let last_off = Variable.reg spec ~width:pitch_bits in
    let lofv = Variable.reg spec ~width:1 in
    let p = sel_bottom i.code ~width:pitch_bits in
    let hot = binary_to_onehot p in
    compile
      [ when_
          i.boot
          [ mask <-- zero mask_bits
          ; count <--. 0
          ; lov <--. 0
          ; lofv <--. 0
          ; last_on <--. 0
          ; last_off <--. 0
          ]
      ; when_
          (i.land_ &: ~:(i.boot))
          [ if_
              (i.code ==:. end_code)
              [ lov <--. 0; lofv <--. 0 ]
              [ when_
                  (i.code <>:. start_code)
                  [ if_
                      (msb i.code)
                      [ mask <-- (mask.value |: hot)
                      ; count <-- count.value +:. 1
                      ; last_on <-- p
                      ; lov <--. 1
                      ]
                      [ mask <-- (mask.value &: ~:hot)
                      ; count <-- count.value -:. 1
                      ; last_off <-- p
                      ; lofv <--. 1
                      ]
                  ]
              ]
          ]
      ];
    let q = sel_bottom i.query ~width:pitch_bits in
    let bit = mux q (bits_lsb mask.value) in
    let off_ok = bit &: ~:(lov.value) &: (~:(lofv.value) |: (q >: last_off.value)) in
    (* [q <>:. 0] is the rule of [is_legal]: [Off 0] has no code, thus pitch 0 never
       starts *)
    let on_ok =
      ~:bit
      &: (q <>:. 0)
      &: (count.value <:. Token.seats)
      &: (~:(lov.value) |: (q <: last_on.value))
    in
    { O.legal =
        mux2
          (i.query ==:. end_code)
          vdd
          (mux2 (i.query ==:. start_code) gnd (mux2 (msb i.query) on_ok off_ok))
    }
  ;;

  (* The two statements of the grammar, side by side. At each step the circuit answers
     every code of the vocabulary and [legal_mask] must agree, code for code. The walk
     lands a drawn legal token, thus the seats fill, the run of ONs falls, End clears the
     pair and the OFFs climb — the whole grammar, with no hand-written path. *)
  let%expect_test "the circuit answers as the mask does, code for code" =
    let module Sim = Cyclesim.With_interface (I) (O) in
    let sim = Sim.create create in
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs sim in
    (* the registers hold while [boot] and [land_] are low, thus a query moves nothing *)
    let legal_of code =
      inp.query := Bits.of_unsigned_int ~width:code_bits code;
      Cyclesim.cycle sim;
      Bits.to_bool !(out.legal)
    in
    let land_token code =
      inp.code := Bits.of_unsigned_int ~width:code_bits code;
      inp.land_ := Bits.vdd;
      Cyclesim.cycle sim;
      inp.land_ := Bits.gnd
    in
    let checks = ref 0 in
    let disagreements = ref 0 in
    let compare_row mask =
      Array.iteri mask ~f:(fun code allowed ->
        Int.incr checks;
        if not (Bool.equal allowed (legal_of code)) then Int.incr disagreements)
    in
    let draw prng mask =
      let allowed = Array.filter_mapi mask ~f:(fun code m -> Option.some_if m code) in
      let prng, byte = Prng.run Prng.next prng in
      prng, allowed.(byte % Array.length allowed)
    in
    let walk (state, prng, landed) (_ : int) =
      let mask = legal_mask state in
      compare_row mask;
      let prng, code = draw prng mask in
      land_token code;
      let token = Token.of_code code in
      step state token, prng, token :: landed
    in
    let is_on = function
      | Token.On (_ : int) -> true
      | Start | Off _ | End -> false
    in
    let is_off = function
      | Token.Off (_ : int) -> true
      | Start | On _ | End -> false
    in
    let is_end = function
      | Token.End -> true
      | Start | On _ | Off _ -> false
    in
    inp.boot := Bits.vdd;
    Cyclesim.cycle sim;
    inp.boot := Bits.gnd;
    let steps = 256 in
    let (_ : sounding), (_ : Prng.state), landed =
      List.fold (List.range 0 steps) ~init:(silence, Prng.create ~seed:7, []) ~f:walk
    in
    let count f = List.count landed ~f in
    printf
      "%d steps landed %d On, %d Off, %d End; %d checks, %d disagreements\n"
      steps
      (count is_on)
      (count is_off)
      (count is_end)
      !checks
      !disagreements;
    [%expect {| 256 steps landed 82 On, 78 Off, 96 End; 65536 checks, 0 disagreements |}]
  ;;
end
