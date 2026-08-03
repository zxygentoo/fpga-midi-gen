open Base
open Bytes_util

module Constants = struct
  let request_header_bytes = 3 (* OP, ADDR, LEN *)
  let response_header_bytes = 2 (* OP, STATUS *)
  let max_data_len = 32
  let max_payload_bytes = request_header_bytes + max_data_len
end

module Op = struct
  let read = 0x01
  let write = 0x02
  let response_flag = 0x80
end

module Status = struct
  type t =
    | Ok
    | Bad_op
    | Bad_address
    | Bad_length

  let to_code = function
    | Ok -> 0x00
    | Bad_op -> 0x01
    | Bad_address -> 0x02
    | Bad_length -> 0x03
  ;;

  let of_code = function
    | 0x00 -> Some Ok
    | 0x01 -> Some Bad_op
    | 0x02 -> Some Bad_address
    | 0x03 -> Some Bad_length
    | _ -> None
  ;;

  let to_string = function
    | Ok -> "ok"
    | Bad_op -> "bad-op"
    | Bad_address -> "bad-address"
    | Bad_length -> "bad-length"
  ;;
end

module Reg = struct
  let base = 0x00
  let size = 16
  let run = 0x0F (* bit 0; the board button also toggles it *)
  let channel = 0x0E
  let step_ms = 0x0C (* 2 bytes *)
  let gate_ms = 0x0A (* 2 bytes *)
  let velocity = 0x09
  let seed = 0x05 (* 4 bytes *)
  let midi_go = 0x04 (* write: send; read: 1 while a message waits *)
  let midi_len = 0x03
  let midi_msg = 0x00 (* 3 bytes *)
end

module Default = struct
  (* MIDI channel 3, the S-1 factory default. *)
  let channel = 2
  let step_ms = 250
  let gate_ms = 125
  let velocity = 100
  let seed = 42 (* must not be 0 *)
end

(* The framing is COBS; see [Cobs] in [lib/cobs.ml]. The byte accessors and the
   little-endian codec are in [Bytes_util]: all values of more than one byte are
   little-endian, on the wire and in the cells. *)

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
  (* MSG, MSG_LEN and MSG_GO in one ascending write: Note On, channel 3, C4, velocity 100 *)
  encode_request
    (Write { addr = Reg.midi_msg; data = Bytes.of_string "\x92\x3C\x64\x03\x01" })
  |> hex
  |> Stdio.print_endline;
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
