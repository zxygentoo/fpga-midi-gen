open Base
open Bytes_util
open Control_intf

type request =
  | Read of
      { addr : int
      ; len : int
      }
  | Write of
      { addr : int
      ; data : Bytes.t
      }

type response =
  { op : int
  ; status : Status.t
  ; data : Bytes.t
  }

let check_addr addr =
  if addr < 0 || addr > 0xFF
  then invalid_arg (Printf.sprintf "address %d is not an 8-bit value" addr)
;;

let check_len len =
  if len < 1 || len > Constants.max_data_len
  then invalid_arg (Printf.sprintf "length %d is out of range" len)
;;

let encode_request req =
  let payload =
    match req with
    | Read { addr; len } ->
      check_addr addr;
      check_len len;
      let b = Bytes.create Constants.request_header_bytes in
      set_byte b 0 Op.read;
      set_byte b 1 addr;
      set_byte b 2 len;
      b
    | Write { addr; data } ->
      check_addr addr;
      check_len (Bytes.length data);
      let n = Bytes.length data in
      let b = Bytes.create (Constants.request_header_bytes + n) in
      set_byte b 0 Op.write;
      set_byte b 1 addr;
      set_byte b 2 n;
      Bytes.blit
        ~src:data
        ~src_pos:0
        ~dst:b
        ~dst_pos:Constants.request_header_bytes
        ~len:n;
      b
  in
  Cobs.encode payload
;;

let encode_response { op; status; data } =
  let n = Bytes.length data in
  let b = Bytes.create (Constants.response_header_bytes + n) in
  set_byte b 0 (op lor Op.response_flag);
  set_byte b 1 (Status.to_code status);
  Bytes.blit ~src:data ~src_pos:0 ~dst:b ~dst_pos:Constants.response_header_bytes ~len:n;
  Cobs.encode b
;;

let decode_request frame =
  match Cobs.decode frame with
  | Error e -> Error e
  | Ok b ->
    let n = Bytes.length b in
    if n < Constants.request_header_bytes
    then Error "the request is too short"
    else (
      let op = byte b 0 in
      let addr = byte b 1 in
      let len = byte b 2 in
      if len < 1 || len > Constants.max_data_len
      then Error "the length is out of range"
      else if op = Op.read
      then
        if n = Constants.request_header_bytes
        then Ok (Read { addr; len })
        else Error "the read has extra bytes"
      else if op = Op.write
      then
        if n = Constants.request_header_bytes + len
        then
          Ok (Write { addr; data = Bytes.sub b ~pos:Constants.request_header_bytes ~len })
        else Error "the write length does not agree with the frame"
      else Error "the operation is not known")
;;

let decode_response frame =
  match Cobs.decode frame with
  | Error e -> Error e
  | Ok b ->
    let n = Bytes.length b in
    if n < Constants.response_header_bytes
    then Error "the response is too short"
    else (
      let op = byte b 0 in
      if op land Op.response_flag = 0
      then Error "the response flag is not set"
      else (
        match Status.of_code (byte b 1) with
        | None -> Error "the status is not known"
        | Some status ->
          Ok
            { op = op land lnot Op.response_flag
            ; status
            ; data =
                Bytes.sub
                  b
                  ~pos:Constants.response_header_bytes
                  ~len:(n - Constants.response_header_bytes)
            }))
;;

(* the board side of the codec: the drivers encode a request and decode a response, and
   the hardware does the other two. [Control_transport] fakes a board with this one. *)
module For_test = struct
  let encode_response = encode_response
end

let%expect_test "request round trips" =
  let check r =
    Stdio.printf "%b\n" (Poly.equal (decode_request (encode_request r)) (Ok r))
  in
  check (Read { addr = Reg.run; len = 1 });
  check (Read { addr = Reg.base; len = Reg.size });
  check (Write { addr = Reg.step_ms; data = Bytes.of_string "\xFA\x00" });
  check (Write { addr = Reg.seed; data = Bytes.of_string "\xEE\xFF\xC0\x00" });
  [%expect {|
    true
    true
    true
    true
    |}]
;;

let%expect_test "response round trips" =
  let check r =
    Stdio.printf "%b\n" (Poly.equal (decode_response (encode_response r)) (Ok r))
  in
  check { op = Op.read; status = Status.Ok; data = Bytes.of_string "\x01" };
  check { op = Op.write; status = Status.Ok; data = Bytes.create 0 };
  check { op = Op.read; status = Status.Bad_address; data = Bytes.create 0 };
  check { op = Op.write; status = Status.Bad_length; data = Bytes.create 0 };
  [%expect {|
    true
    true
    true
    true
    |}]
;;

let%expect_test "the one-shot doorbell frame on the wire" =
  (* Note On, channel 3, C4, velocity 100 *)
  let addr, data = build_doorbell [ 0x92; 0x3C; 0x64 ] in
  encode_request (Write { addr; data }) |> hex |> Stdio.print_endline;
  [%expect {| 02 02 07 05 92 3c 64 03 01 00 |}]
;;

let%expect_test "the encoder rejects values that do not fit" =
  let attempt f =
    match f () with
    | exception Invalid_argument m -> Stdio.print_endline m
    | _ -> Stdio.print_endline "accepted"
  in
  attempt (fun () -> encode_request (Read { addr = 0x10000; len = 1 }));
  [%expect {| address 65536 is not an 8-bit value |}];
  attempt (fun () -> encode_request (Read { addr = 0; len = 0 }));
  [%expect {| length 0 is out of range |}];
  attempt (fun () -> encode_request (Write { addr = 0; data = Bytes.create 33 }));
  [%expect {| length 33 is out of range |}]
;;

let%expect_test "a frame without the delimiter must not decode" =
  (match decode_response (Bytes.of_string "\x02\x81") with
   | Ok _ -> Stdio.print_endline "decoded"
   | Error e -> Stdio.print_endline e);
  [%expect {| the frame has no zero delimiter |}]
;;
