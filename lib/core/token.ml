open Core

type t =
  | Start
  | On of int
  | Off of int
  | End
[@@deriving sexp_of]

let vocab = 256
let seats = 4

let to_code = function
  | Start -> 0xff
  | On pitch ->
    if pitch < 0 || pitch > 126
    then invalid_argf "On pitch %d is outside 0 to 126" pitch ();
    0x80 lor pitch
  | Off pitch ->
    if pitch < 1 || pitch > 127
    then invalid_argf "Off pitch %d is outside 1 to 127" pitch ();
    pitch
  | End -> 0
;;

let of_code code =
  if code < 0 || code > 255 then invalid_argf "the code %d is outside 0 to 255" code ();
  if code = 0xff
  then Start
  else if code >= 0x80
  then On (code land 0x7f)
  else if code > 0
  then Off code
  else End
;;

let%expect_test "the codec goes both ways" =
  List.iter [ 0xff; 0x80; 0xbc; 0xfe; 0x01; 0x3c; 0x7f; 0x00 ] ~f:(fun code ->
    let token = of_code code in
    printf "0x%02x %-10s 0x%02x\n" code (Sexp.to_string (sexp_of_t token)) (to_code token));
  [%expect
    {|
    0xff Start      0xff
    0x80 (On 0)     0x80
    0xbc (On 60)    0xbc
    0xfe (On 126)   0xfe
    0x01 (Off 1)    0x01
    0x3c (Off 60)   0x3c
    0x7f (Off 127)  0x7f
    0x00 End        0x00
    |}]
;;
