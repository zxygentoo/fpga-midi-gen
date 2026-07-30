let delimiter = '\000'

let encode src =
  let n = Bytes.length src in
  let out = Bytes.create (n + (n / 254) + 2) in
  let rec group i out_pos =
    match Bytes.index_from_opt src i delimiter with
    | Some z when z - i < 254 ->
      (* a short group, closed by a real zero at [z] *)
      Bytes.set_uint8 out out_pos (z - i + 1);
      Bytes.blit src i out (out_pos + 1) (z - i);
      group (z + 1) (out_pos + (z - i) + 1)
    | _ when n - i >= 254 ->
      (* a full group of 254 bytes, no zero *)
      Bytes.set_uint8 out out_pos 0xFF;
      Bytes.blit src i out (out_pos + 1) 254;
      let i = i + 254 in
      if i = n then finish (out_pos + 255) else group i (out_pos + 255)
    | _ ->
      (* the last group, closed by the end of the input *)
      let len = n - i in
      Bytes.set_uint8 out out_pos (len + 1);
      Bytes.blit src i out (out_pos + 1) len;
      finish (out_pos + len + 1)
  and finish out_pos =
    Bytes.set_uint8 out out_pos 0;
    Bytes.sub out 0 (out_pos + 1)
  in
  group 0 0
;;

let decode frame =
  let n = Bytes.length frame in
  if n < 2
  then Error "the frame is too short"
  else (
    match Bytes.index_from_opt frame 0 delimiter with
    | None -> Error "the frame has no zero delimiter"
    | Some z when z < n - 1 -> Error "a zero byte is inside the frame"
    | Some _ ->
      (* the body [0, n-1) is now known to hold no zero byte *)
      let body = n - 1 in
      let out = Bytes.create body in
      let rec group i out_pos =
        if i = body
        then Ok (Bytes.sub out 0 out_pos)
        else (
          let code = Bytes.get_uint8 frame i in
          if i + code > body
          then Error "the frame is too short"
          else (
            Bytes.blit frame (i + 1) out out_pos (code - 1);
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

let%expect_test "the examples from the paper" =
  let show s =
    encode (Bytes.of_string s)
    |> Bytes.to_seq
    |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
    |> List.of_seq
    |> String.concat " "
    |> print_endline
  in
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
  let b254 = Bytes.init 254 (fun i -> Char.chr (i + 1)) in
  let b255 = Bytes.init 255 (fun i -> Char.chr (i + 1)) in
  let summary b =
    let e = encode b in
    let n = Bytes.length e in
    Printf.printf
      "%d -> %d, head %02x, tail %02x %02x\n"
      (Bytes.length b)
      n
      (Bytes.get_uint8 e 0)
      (Bytes.get_uint8 e (n - 2))
      (Bytes.get_uint8 e (n - 1))
  in
  summary b254;
  [%expect {| 254 -> 256, head ff, tail fe 00 |}];
  summary b255;
  [%expect {| 255 -> 258, head ff, tail ff 00 |}];
  (* the decoder also accepts the phantom-zero form of the same input *)
  let phantom =
    Bytes.concat Bytes.empty [ Bytes.of_string "\xFF"; b254; Bytes.of_string "\x01\x00" ]
  in
  Printf.printf "%b\n" (decode phantom = Ok b254);
  [%expect {| true |}]
;;

let%expect_test "round trips" =
  [ Bytes.of_string ""
  ; Bytes.of_string "\x00"
  ; Bytes.of_string "\x00\x11\x00"
  ; Bytes.init 300 (fun i -> Char.chr (i mod 256))
  ]
  |> List.iter (fun b -> Printf.printf "%b\n" (decode (encode b) = Ok b));
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
      let byte = oneof_weighted [ 1, return '\000'; 3, char ] in
      bytes_size ~gen:byte (int_range 0 600)
    in
    let print b =
      Bytes.to_seq b
      |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
      |> List.of_seq
      |> String.concat " "
    in
    make ~print gen
  in
  let check t = Test.check_exn ~rand:(Random.State.make [| 0xC0B5 |]) t in
  check
    (Test.make ~count:1000 ~name:"decode inverts encode" payload (fun b ->
       decode (encode b) = Ok b));
  check
    (Test.make
       ~count:1000
       ~name:"the delimiter is the only zero, and the last byte"
       payload
       (fun b ->
          let e = encode b in
          Bytes.index_opt e delimiter = Some (Bytes.length e - 1)));
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
         Result.is_error (decode (Bytes.sub e 0 (k mod Bytes.length e)))));
  check
    (Test.make
       ~count:1000
       ~name:"a byte after the delimiter does not decode"
       payload
       (fun b -> Result.is_error (decode (Bytes.cat (encode b) (Bytes.make 1 'A')))));
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
    | Ok _ -> print_endline "decoded"
    | Error e -> print_endline e
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
