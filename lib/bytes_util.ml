open Base

let byte b i = Char.to_int (Bytes.get b i)

let set_byte b i v =
  if v < 0 || v > 0xff then invalid_arg (Printf.sprintf "byte value %d is out of range" v);
  Bytes.set b i (Char.unsafe_of_int v)
;;

let uint_le b ~pos ~width =
  List.init width ~f:(fun k -> byte b (pos + k) lsl (8 * k))
  |> List.fold ~init:0 ~f:(fun acc part -> acc lor part)
;;

let hex b =
  String.concat
    ~sep:" "
    (List.map (Bytes.to_list b) ~f:(fun c -> Printf.sprintf "%02x" (Char.to_int c)))
;;

let%expect_test "the accessors and the little-endian order" =
  let b = Bytes.create 4 in
  List.iteri [ 0xEF; 0xBE; 0xAD; 0xDE ] ~f:(fun i v -> set_byte b i v);
  Stdio.print_endline (hex b);
  [%expect {| ef be ad de |}];
  Stdio.printf
    "%x %x %x\n"
    (uint_le b ~pos:0 ~width:1)
    (uint_le b ~pos:0 ~width:2)
    (uint_le b ~pos:0 ~width:4);
  [%expect {| ef beef deadbeef |}];
  (* [set_byte] keeps the range check *)
  (match set_byte b 0 0x100 with
   | () -> Stdio.print_endline "accepted"
   | exception Invalid_argument m -> Stdio.print_endline m);
  [%expect {| byte value 256 is out of range |}]
;;
