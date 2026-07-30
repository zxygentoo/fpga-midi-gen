(** The ABI between the host drivers and the FPGA.

    This module is the single definition of the ABI constants and of the wire-frame codec.
    The drivers use it directly. The RTL reads the constants at elaboration time. The
    normative description is [docs/abi.md]. *)

module Limits = struct
  let max_data_len = 32
  let max_frame_wire_bytes = 64
  let response_timeout_ms = 100
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
end

module Reg = struct
  (* Control: 0xFFF0 to 0xFFFF, read and write — the full register file. Cells with a
     fixed nature sit at the top. MSG, the one payload with a variable nature, sits at the
     low edge so it can grow with no change to the other addresses. *)
  module Ctl = struct
    let base = 0xFFF0
    let size = 16
    let run = 0xFFFF (* bit 0; the board button also toggles it *)
    let channel = 0xFFFE
    let step_ms = 0xFFFC (* 2 bytes *)
    let gate_ms = 0xFFFA (* 2 bytes *)
    let velocity = 0xFFF9
    let seed = 0xFFF5 (* 4 bytes; the PRNG loads on a write to the last byte *)
    let msg_go = 0xFFF4 (* write: send; read: 1 while a message waits *)
    let msg_len = 0xFFF3
    let msg = 0xFFF0 (* 3 bytes, at the growth edge *)
  end

  (* The model window grows up from 0x0000: byte [i] of the blob is at address [i]. The
     wire protocol does not map it in this version. *)
  let window_base = 0x0000
end

module Default = struct
  (* MIDI channel 3, the S-1 factory default. *)
  let channel = 2
  let step_ms = 250
  let gate_ms = 125
  let velocity = 100
  let seed = 0x00C0FFEE (* must not be 0 *)
end

(* The framing is COBS; see [Cobs] in [lib/cobs.ml]. *)

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
  if addr < 0 || addr > 0xFFFF
  then invalid_arg (Printf.sprintf "address %d is not a 16-bit value" addr)
;;

let check_len len =
  if len < 1 || len > Limits.max_data_len
  then invalid_arg (Printf.sprintf "length %d is out of range" len)
;;

let encode_request req =
  let payload =
    match req with
    | Read { addr; len } ->
      check_addr addr;
      check_len len;
      let b = Bytes.create 4 in
      Bytes.set_uint8 b 0 Op.read;
      Bytes.set_uint16_le b 1 addr;
      Bytes.set_uint8 b 3 len;
      b
    | Write { addr; data } ->
      check_addr addr;
      check_len (Bytes.length data);
      let n = Bytes.length data in
      let b = Bytes.create (4 + n) in
      Bytes.set_uint8 b 0 Op.write;
      Bytes.set_uint16_le b 1 addr;
      Bytes.set_uint8 b 3 n;
      Bytes.blit data 0 b 4 n;
      b
  in
  Cobs.encode payload
;;

let encode_response { op; status; data } =
  let n = Bytes.length data in
  let b = Bytes.create (2 + n) in
  Bytes.set_uint8 b 0 (op lor Op.response_flag);
  Bytes.set_uint8 b 1 (Status.to_code status);
  Bytes.blit data 0 b 2 n;
  Cobs.encode b
;;

let decode_request frame =
  match Cobs.decode frame with
  | Error e -> Error e
  | Ok b ->
    let n = Bytes.length b in
    if n < 4
    then Error "the request is too short"
    else (
      let op = Bytes.get_uint8 b 0 in
      let addr = Bytes.get_uint16_le b 1 in
      let len = Bytes.get_uint8 b 3 in
      if len < 1 || len > Limits.max_data_len
      then Error "the length is out of range"
      else if op = Op.read
      then if n = 4 then Ok (Read { addr; len }) else Error "the read has extra bytes"
      else if op = Op.write
      then
        if n = 4 + len
        then Ok (Write { addr; data = Bytes.sub b 4 len })
        else Error "the write length does not agree with the frame"
      else Error "the operation is not known")
;;

let decode_response frame =
  match Cobs.decode frame with
  | Error e -> Error e
  | Ok b ->
    let n = Bytes.length b in
    if n < 2
    then Error "the response is too short"
    else (
      let op = Bytes.get_uint8 b 0 in
      if op land Op.response_flag = 0
      then Error "the response flag is not set"
      else (
        match Status.of_code (Bytes.get_uint8 b 1) with
        | None -> Error "the status is not known"
        | Some status ->
          Ok { op = op land lnot Op.response_flag; status; data = Bytes.sub b 2 (n - 2) }))
;;

let%expect_test "request round trips" =
  let check r = Printf.printf "%b\n" (decode_request (encode_request r) = Ok r) in
  check (Read { addr = Reg.Ctl.run; len = 1 });
  check (Read { addr = Reg.Ctl.base; len = Reg.Ctl.size });
  check (Write { addr = Reg.Ctl.step_ms; data = Bytes.of_string "\xFA\x00" });
  check (Write { addr = Reg.Ctl.seed; data = Bytes.of_string "\xEE\xFF\xC0\x00" });
  [%expect {|
    true
    true
    true
    true
    |}]
;;

let%expect_test "response round trips" =
  let check r = Printf.printf "%b\n" (decode_response (encode_response r) = Ok r) in
  check { op = Op.read; status = Status.Ok; data = Bytes.of_string "\x01" };
  check { op = Op.write; status = Status.Ok; data = Bytes.empty };
  check { op = Op.read; status = Status.Bad_address; data = Bytes.empty };
  check { op = Op.write; status = Status.Bad_length; data = Bytes.empty };
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
    (Write { addr = Reg.Ctl.msg; data = Bytes.of_string "\x92\x3C\x64\x03\x01" })
  |> Bytes.to_seq
  |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
  |> List.of_seq
  |> String.concat " "
  |> print_endline;
  [%expect {| 0a 02 f0 ff 05 92 3c 64 03 01 00 |}]
;;

let%expect_test "the encoder rejects values that do not fit" =
  let attempt f =
    match f () with
    | exception Invalid_argument m -> print_endline m
    | _ -> print_endline "accepted"
  in
  attempt (fun () -> encode_request (Read { addr = 0x10000; len = 1 }));
  [%expect {| address 65536 is not a 16-bit value |}];
  attempt (fun () -> encode_request (Read { addr = 0; len = 0 }));
  [%expect {| length 0 is out of range |}];
  attempt (fun () -> encode_request (Write { addr = 0; data = Bytes.create 33 }));
  [%expect {| length 33 is out of range |}]
;;

let%expect_test "a frame without the delimiter must not decode" =
  (match decode_response (Bytes.of_string "\x02\x81") with
   | Ok _ -> print_endline "decoded"
   | Error e -> print_endline e);
  [%expect {| the frame has no zero delimiter |}]
;;
