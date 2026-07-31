open Base

let delimiter = '\000'

(* Base.Bytes has no byte-integer accessors; these state the intent one time *)
let byte b i = Char.to_int (Bytes.get b i)
let set_byte b i v = Bytes.set b i (Char.of_int_exn v)

let find_delimiter src ~from =
  let n = Bytes.length src in
  let rec go i =
    if i >= n
    then None
    else if Char.equal (Bytes.get src i) delimiter
    then Some i
    else go (i + 1)
  in
  go from
;;

let encode src =
  let n = Bytes.length src in
  let out = Bytes.create (n + (n / 254) + 2) in
  let rec group i out_pos =
    match find_delimiter src ~from:i with
    | Some z when z - i < 254 ->
      (* a short group, closed by a real zero at [z] *)
      set_byte out out_pos (z - i + 1);
      Bytes.blit ~src ~src_pos:i ~dst:out ~dst_pos:(out_pos + 1) ~len:(z - i);
      group (z + 1) (out_pos + (z - i) + 1)
    | _ when n - i >= 254 ->
      (* a full group of 254 bytes, no zero *)
      set_byte out out_pos 0xFF;
      Bytes.blit ~src ~src_pos:i ~dst:out ~dst_pos:(out_pos + 1) ~len:254;
      let i = i + 254 in
      if i = n then finish (out_pos + 255) else group i (out_pos + 255)
    | _ ->
      (* the last group, closed by the end of the input *)
      let len = n - i in
      set_byte out out_pos (len + 1);
      Bytes.blit ~src ~src_pos:i ~dst:out ~dst_pos:(out_pos + 1) ~len;
      finish (out_pos + len + 1)
  and finish out_pos =
    set_byte out out_pos 0;
    Bytes.sub out ~pos:0 ~len:(out_pos + 1)
  in
  group 0 0
;;

let decode frame =
  let n = Bytes.length frame in
  if n < 2
  then Error "the frame is too short"
  else (
    match find_delimiter frame ~from:0 with
    | None -> Error "the frame has no zero delimiter"
    | Some z when z < n - 1 -> Error "a zero byte is inside the frame"
    | Some _ ->
      (* the body [0, n-1) is now known to hold no zero byte *)
      let body = n - 1 in
      let out = Bytes.create body in
      let rec group i out_pos =
        if i = body
        then Ok (Bytes.sub out ~pos:0 ~len:out_pos)
        else (
          let code = byte frame i in
          if i + code > body
          then Error "the frame is too short"
          else (
            Bytes.blit
              ~src:frame
              ~src_pos:(i + 1)
              ~dst:out
              ~dst_pos:out_pos
              ~len:(code - 1);
            let i = i + code
            and out_pos = out_pos + code - 1 in
            if code < 0xFF && i < body
            then (
              Bytes.set out out_pos delimiter;
              group i (out_pos + 1))
            else group i out_pos))
      in
      group 0 0)
;;

let hex b =
  String.concat
    ~sep:" "
    (List.map (Bytes.to_list b) ~f:(fun c -> Printf.sprintf "%02x" (Char.to_int c)))
;;

let%expect_test "the examples from the paper" =
  let show s = Stdio.print_endline (hex (encode (Bytes.of_string s))) in
  show "";
  [%expect {| 01 00 |}];
  show "\x00";
  [%expect {| 01 01 00 |}];
  show "\x11\x22\x00\x33";
  [%expect {| 03 11 22 02 33 00 |}];
  show "\x11\x22\x33\x44";
  [%expect {| 05 11 22 33 44 00 |}]
;;

let%expect_test "the 254-byte group boundary keeps the minimal form" =
  let b254 = Bytes.init 254 ~f:(fun i -> Char.of_int_exn (i + 1)) in
  let b255 = Bytes.init 255 ~f:(fun i -> Char.of_int_exn (i + 1)) in
  let summary b =
    let e = encode b in
    let n = Bytes.length e in
    Stdio.printf
      "%d -> %d, head %02x, tail %02x %02x\n"
      (Bytes.length b)
      n
      (byte e 0)
      (byte e (n - 2))
      (byte e (n - 1))
  in
  summary b254;
  [%expect {| 254 -> 256, head ff, tail fe 00 |}];
  summary b255;
  [%expect {| 255 -> 258, head ff, tail ff 00 |}];
  (* the decoder also accepts the phantom-zero form of the same input *)
  let phantom =
    Bytes.of_string (String.concat ~sep:"" [ "\xFF"; Bytes.to_string b254; "\x01\x00" ])
  in
  Stdio.printf "%b\n" (Poly.equal (decode phantom) (Ok b254));
  [%expect {| true |}]
;;

let%expect_test "round trips" =
  [ Bytes.of_string ""
  ; Bytes.of_string "\x00"
  ; Bytes.of_string "\x00\x11\x00"
  ; Bytes.init 300 ~f:(fun i -> Char.of_int_exn (i % 256))
  ]
  |> List.iter ~f:(fun b -> Stdio.printf "%b\n" (Poly.equal (decode (encode b)) (Ok b)));
  [%expect {|
    true
    true
    true
    true
    |}]
;;

let%expect_test "the randomized properties" =
  let open QCheck in
  (* sizes up to 600 cross the 254-byte group boundary twice; one byte in four is a zero,
     thus the group structure is rich. The fixed seed keeps the run deterministic. *)
  let payload =
    let gen =
      let open Gen in
      let one_byte = oneof_weighted [ 1, return '\000'; 3, char ] in
      bytes_size ~gen:one_byte (int_range 0 600)
    in
    make ~print:hex gen
  in
  let check t = Test.check_exn ~rand:(Stdlib.Random.State.make [| 0xC0B5 |]) t in
  check
    (Test.make ~count:1000 ~name:"decode inverts encode" payload (fun b ->
       Poly.equal (decode (encode b)) (Ok b)));
  check
    (Test.make
       ~count:1000
       ~name:"the delimiter is the only zero, and the last byte"
       payload
       (fun b ->
          let e = encode b in
          Poly.equal (find_delimiter e ~from:0) (Some (Bytes.length e - 1))));
  check
    (Test.make ~count:1000 ~name:"the size bound holds" payload (fun b ->
       let n = Bytes.length b in
       let m = Bytes.length (encode b) in
       n + 2 <= m && m <= n + (n / 254) + 2));
  check
    (Test.make
       ~count:1000
       ~name:"a frame cut at any point does not decode"
       (pair payload (make Gen.(int_bound 10_000)))
       (fun (b, k) ->
         let e = encode b in
         Result.is_error (decode (Bytes.sub e ~pos:0 ~len:(k % Bytes.length e)))));
  check
    (Test.make
       ~count:1000
       ~name:"a byte after the delimiter does not decode"
       payload
       (fun b ->
          let extended = Bytes.of_string (Bytes.to_string (encode b) ^ "A") in
          Result.is_error (decode extended)));
  check
    (Test.make
       ~count:1000
       ~name:"decode does not raise on any input"
       (make Gen.(bytes_size (int_range 0 600)))
       (fun f ->
         match decode f with
         | Ok _ | Error _ -> true));
  [%expect {| |}]
;;

let%expect_test "frames that must not decode" =
  let show s =
    match decode (Bytes.of_string s) with
    | Ok _ -> Stdio.print_endline "decoded"
    | Error e -> Stdio.print_endline e
  in
  show "\x02\x81";
  [%expect {| the frame has no zero delimiter |}];
  show "\x00";
  [%expect {| the frame is too short |}];
  show "\x03\x11\x00\x33\x00";
  [%expect {| a zero byte is inside the frame |}];
  show "\x05\x11\x00";
  [%expect {| the frame is too short |}]
;;
