open Core

(* The window of the corpus. The numbers are stated and derived from nothing: a checkpoint
   holds one row for each class, thus a window that followed the corpus would make every
   trained model wrong and say nothing. The expect test of [Jsb] holds the corpus inside
   them. *)
let classes = 48
let silence = 0
let pitch_low = 36

(* the highest pitch the window states, which is the spare class: the corpus stops one
   pitch below it *)
let pitch_high = pitch_low + classes - 2

let class_of_code code =
  match Frame.pitch_of_code code with
  | None -> silence
  | Some pitch ->
    if pitch < pitch_low || pitch > pitch_high
    then
      invalid_argf
        "the pitch %d is outside the window %d to %d of the vocabulary"
        pitch
        pitch_low
        pitch_high
        ();
    pitch - pitch_low + 1
;;

let code_of_class index =
  if index < 0 || index >= classes
  then invalid_argf "the class %d is outside 0 to %d" index (classes - 1) ();
  if index = silence
  then Frame.silent_code
  else Frame.code_of_pitch (pitch_low + index - 1)
;;

let classes_of_frame frame = List.map (Frame.codes frame) ~f:class_of_code
let frame_of_classes indices = indices |> List.map ~f:code_of_class |> Frame.of_codes

module Rtl = struct
  (* The map the circuit states, over any combinational type: [Bits] evaluates it in the
     test below and [Signal] elaborates it. It stands here and not in the source of a
     model, because the rule is the vocabulary's and one rule wants one definition — the
     expect test holds the two halves together over every class. *)
  module Make (Comb : Hardcaml.Comb.S) = struct
    open Comb

    (* [code_of_class index] is the voice code of a drawn class: the silent code for class
       0, and the pitch of the class with the sounding flag set for the others. One add
       and one bit, thus the circuit needs no table. *)
    let code_of_class index =
      let width = Frame.code_bits in
      let pitch = uresize index ~width +:. (pitch_low - 1) in
      mux2
        (index ==:. silence)
        (of_unsigned_int ~width Frame.silent_code)
        (pitch |: of_unsigned_int ~width 0x80)
    ;;
  end

  include Make (Hardcaml.Signal)
end

let%expect_test "the circuit states the code the software states" =
  (* One rule and two halves: the software maps a class at the seam of the corpus, and the
     circuit maps it at the seam of the wire. Every class of the vocabulary is checked,
     because the set is small enough that a sample would be the weaker test. *)
  let module Bits = Hardcaml.Bits in
  let module Map = Rtl.Make (Bits) in
  let index_bits = Int.ceil_log2 classes in
  let disagree =
    List.filter (List.range 0 classes) ~f:(fun index ->
      let circuit =
        Bits.to_unsigned_int
          (Map.code_of_class (Bits.of_unsigned_int ~width:index_bits index))
      in
      circuit <> code_of_class index)
  in
  printf "%d classes, %d disagree\n" classes (List.length disagree);
  [%expect {| 48 classes, 0 disagree |}]
;;

let%expect_test "the class of a voice code" =
  let show code = printf "0x%02x -> %d\n" code (class_of_code code) in
  (* the silent code, then the two ends of the corpus and the spare class above them *)
  show Frame.silent_code;
  show (Frame.code_of_pitch 36);
  show (Frame.code_of_pitch 81);
  show (Frame.code_of_pitch 82);
  [%expect {|
    0x00 -> 0
    0xa4 -> 1
    0xd1 -> 46
    0xd2 -> 47
    |}];
  (* a spare code of the frame reads as silence, which is what the wire says: the flag is
     clear, thus the pitch field carries nothing *)
  show 0x05;
  [%expect {| 0x05 -> 0 |}];
  (* a pitch outside the window refuses, because no table holds a row for it *)
  let refuse pitch =
    match class_of_code (Frame.code_of_pitch pitch) with
    | (_ : int) -> ()
    | exception Invalid_argument message -> print_endline message
  in
  refuse 35;
  refuse 83;
  [%expect
    {|
    the pitch 35 is outside the window 36 to 82 of the vocabulary
    the pitch 83 is outside the window 36 to 82 of the vocabulary
    |}]
;;

let%expect_test "the voice code of a class" =
  let show index = printf "%d -> 0x%02x\n" index (code_of_class index) in
  show silence;
  show 1;
  show 46;
  show (classes - 1);
  [%expect {|
    0 -> 0x00
    1 -> 0xa4
    46 -> 0xd1
    47 -> 0xd2
    |}];
  let refuse index =
    match code_of_class index with
    | (_ : int) -> ()
    | exception Invalid_argument message -> print_endline message
  in
  refuse (-1);
  refuse classes;
  [%expect
    {|
    the class -1 is outside 0 to 47
    the class 48 is outside 0 to 47
    |}]
;;

let%expect_test "the frame goes both ways" =
  let both frame =
    let indices = classes_of_frame frame in
    printf
      "%08x -> %s -> %08x\n"
      frame
      (Sexp.to_string ([%sexp_of: int list] indices))
      (frame_of_classes indices)
  in
  (* the chord of the reader test of [Jsb]: seat 0 first, thus the bass leads *)
  both 0xcac6c1ba;
  (* a rest in the alto seat *)
  both 0xca00c1ba;
  (* the silent frame is four silent classes, thus a cleared table reads as silence *)
  both Frame.silent;
  [%expect
    {|
    cac6c1ba -> (23 30 35 39) -> cac6c1ba
    ca00c1ba -> (23 30 0 39) -> ca00c1ba
    00000000 -> (0 0 0 0) -> 00000000
    |}];
  (* the seats of a frame are the seats of the synthesizer, and a shorter list refuses *)
  (match frame_of_classes [ 23; 30; 35 ] with
   | (_ : int) -> ()
   | exception Invalid_argument message -> print_endline message);
  [%expect {| a frame takes 4 codes, one for each seat |}]
;;
