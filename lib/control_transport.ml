type t =
  { send : Bytes.t -> unit
  ; receive : unit -> char
  }

type error =
  | Garbled
  | Nak of Abi.Status.t

let resync t = t.send (Bytes.make 1 Cobs.delimiter)

(* one response frame, delimiter last *)
let collect_frame t =
  let buffer = Buffer.create 64 in
  let rec next () =
    let byte = t.receive () in
    Buffer.add_char buffer byte;
    if Char.equal byte Cobs.delimiter then Buffer.to_bytes buffer else next ()
  in
  next ()
;;

(* a response is usable when it decodes and echoes the op of this request *)
let usable_response ~op response_frame =
  let ( let* ) = Option.bind in
  let* response = Result.to_option (Abi.decode_response response_frame) in
  if response.Abi.op = op then Some response else None
;;

(* the three outcomes: the answer, a rejection, or garble — and each operation is
   idempotent, thus after a garble the caller can simply run it again *)
let transact t request ~op ~data_length =
  t.send (Abi.encode_request request);
  match usable_response ~op (collect_frame t) with
  | Some { status = Abi.Status.Ok; data; _ } when Bytes.length data = data_length ->
    Ok data
  | Some { status = Abi.Status.Ok; _ } (* the wrong shape *) | None -> Error Garbled
  | Some { status; _ } -> Error (Nak status)
;;

let read t ~address ~length =
  transact
    t
    (Abi.Read { addr = address; len = length })
    ~op:Abi.Op.read
    ~data_length:length
;;

let write t ~address ~data =
  transact t (Abi.Write { addr = address; data }) ~op:Abi.Op.write ~data_length:0
  |> Result.map (fun (_ : Bytes.t) -> ())
;;

(* The test transport: [pending] holds the scripted reply, and [sent] records the wire. *)

let fake () =
  let pending = Queue.create () in
  let sent = Buffer.create 16 in
  let transport =
    { send = (fun frame -> Buffer.add_bytes sent frame)
    ; receive =
        (fun () ->
          match Queue.take_opt pending with
          | Some byte -> byte
          | None -> failwith "the script is out of bytes")
    }
  in
  transport, pending, sent
;;

let reply pending frame = Bytes.iter (fun c -> Queue.add c pending) frame

let hex bytes =
  bytes
  |> Bytes.to_seq
  |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
  |> List.of_seq
  |> String.concat " "
;;

let show = function
  | Ok data -> Printf.printf "ok %s\n" (hex data)
  | Error Garbled -> print_endline "garbled"
  | Error (Nak status) ->
    Printf.printf
      "nak %s\n"
      (match status with
       | Abi.Status.Ok -> "ok"
       | Bad_op -> "bad-op"
       | Bad_address -> "bad-address"
       | Bad_length -> "bad-length")
;;

let%expect_test "a read, with the wire vectors of the hardware session" =
  let t, pending, sent = fake () in
  let response =
    Abi.encode_response
      { op = Abi.Op.read; status = Abi.Status.Ok; data = Bytes.of_string "\x64" }
  in
  Printf.printf "response frame %s\n" (hex response);
  reply pending response;
  show (read t ~address:Abi.Reg.Ctl.velocity ~length:1);
  Printf.printf "sent %s\n" (hex (Buffer.to_bytes sent));
  [%expect
    {|
    response frame 02 81 02 64 00
    ok 64
    sent 05 01 f9 ff 01 00
    |}]
;;

let%expect_test "a corrupt frame is Garbled" =
  let t, pending, _ = fake () in
  reply pending (Bytes.of_string "\xAA\xBB\x00");
  show (read t ~address:Abi.Reg.Ctl.velocity ~length:1);
  [%expect {| garbled |}]
;;

let%expect_test "the wrong shape is Garbled" =
  (* an Ok response with two data bytes answers a one-byte read; a wrong op echo is the
     same *)
  let t, pending, _ = fake () in
  reply
    pending
    (Abi.encode_response
       { op = Abi.Op.read; status = Abi.Status.Ok; data = Bytes.of_string "\x64\x64" });
  show (read t ~address:Abi.Reg.Ctl.velocity ~length:1);
  let t, pending, _ = fake () in
  reply
    pending
    (Abi.encode_response
       { op = Abi.Op.write; status = Abi.Status.Ok; data = Bytes.empty });
  show (read t ~address:Abi.Reg.Ctl.velocity ~length:1);
  [%expect {|
    garbled
    garbled
    |}]
;;

let%expect_test "a rejection is a rejection, not a garble" =
  let t, pending, _ = fake () in
  reply
    pending
    (Abi.encode_response
       { op = Abi.Op.read; status = Abi.Status.Bad_address; data = Bytes.empty });
  show (read t ~address:0x0000 ~length:1);
  [%expect {| nak bad-address |}]
;;

let%expect_test "resync sends one delimiter" =
  let t, _, sent = fake () in
  resync t;
  Printf.printf "sent %s\n" (hex (Buffer.to_bytes sent));
  [%expect {| sent 00 |}]
;;
