open Core

type t =
  | End
  | Off of int
  | On of int
[@@deriving sexp_of]

let vocab = 256
let seats = 4

let to_byte = function
  | End -> 0
  | Off pitch ->
    if pitch < 1 || pitch > 127
    then invalid_argf "Off pitch %d is outside 1 to 127" pitch ();
    pitch
  | On pitch ->
    if pitch < 0 || pitch > 127
    then invalid_argf "On pitch %d is outside 0 to 127" pitch ();
    0x80 lor pitch
;;

let of_byte byte =
  if byte < 0 || byte > 255 then invalid_argf "%d is not a byte" byte ();
  if byte = 0 then End else if byte < 0x80 then Off byte else On (byte land 0x7f)
;;

let%expect_test "the byte codec goes both ways" =
  List.iter [ 0x00; 0x01; 0x3c; 0x7f; 0x80; 0xbc; 0xff ] ~f:(fun code ->
    let token = of_byte code in
    printf "0x%02x %-10s 0x%02x\n" code (Sexp.to_string (sexp_of_t token)) (to_byte token));
  [%expect
    {|
    0x00 End        0x00
    0x01 (Off 1)    0x01
    0x3c (Off 60)   0x3c
    0x7f (Off 127)  0x7f
    0x80 (On 0)     0x80
    0xbc (On 60)    0xbc
    0xff (On 127)   0xff
    |}]
;;
