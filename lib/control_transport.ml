open Base

type t =
  { send : Bytes.t -> unit
  ; receive : unit -> char
  ; resync : unit -> unit
  }

type error =
  | Garbled
  | Nak of Control.Status.t

(* the transport over an open serial port: raw 8N1, a blocking read, and a resync of
   tcflush plus one delimiter. The caller opens the descriptor and owns its lifetime. *)
let serial ~baud fd =
  let module Terminal_io = Core_unix.Terminal_io in
  let tio = Terminal_io.tcgetattr fd in
  Terminal_io.tcsetattr
    { tio with
      c_ibaud = baud
    ; c_obaud = baud
    ; c_csize = 8
    ; c_cstopb = 1
    ; c_parenb = false
    ; c_cread = true
    ; c_clocal = true
    ; c_icanon = false
    ; c_isig = false
    ; c_echo = false
    ; c_echoe = false
    ; c_echok = false
    ; c_echonl = false
    ; c_ixon = false
    ; c_ixoff = false
    ; c_ignbrk = true
    ; c_brkint = false
    ; c_parmrk = false
    ; c_inpck = false
    ; c_istrip = false
    ; c_inlcr = false
    ; c_igncr = false
    ; c_icrnl = false
    ; c_opost = false
    ; c_vmin = 1
    ; c_vtime = 0
    }
    fd
    ~mode:TCSANOW;
  let send frame =
    if Core_unix.write fd ~buf:frame <> Bytes.length frame
    then failwith "short write to the serial port"
  in
  let receive () =
    let one = Bytes.create 1 in
    match Core_unix.read fd ~buf:one with
    | 1 -> Bytes.get one 0
    | _ -> failwith "the serial port closed"
  in
  let resync () =
    Terminal_io.tcflush fd ~mode:TCIOFLUSH;
    send (Bytes.make 1 Cobs.delimiter)
  in
  { send; receive; resync }
;;

let resync t = t.resync ()

(* one response frame, delimiter last *)
let collect_frame t =
  let buffer = Buffer.create 64 in
  let rec next () =
    let byte = t.receive () in
    Buffer.add_char buffer byte;
    if Char.equal byte Cobs.delimiter then Buffer.contents_bytes buffer else next ()
  in
  next ()
;;

(* a response is usable when it decodes and echoes the op of this request *)
let usable_response ~op response_frame =
  let ( let* ) x f = Option.bind x ~f in
  let* response = Result.ok (Control.decode_response response_frame) in
  if response.Control.op = op then Some response else None
;;

(* the three outcomes: the answer, a rejection, or garble — and each operation is
   idempotent, thus after a garble the caller can simply run it again *)
let transact t request ~op ~data_length =
  t.send (Control.encode_request request);
  match usable_response ~op (collect_frame t) with
  | Some { status = Control.Status.Ok; data; _ } when Bytes.length data = data_length ->
    Ok data
  | Some { status = Control.Status.Ok; _ } (* the wrong shape *) | None -> Error Garbled
  | Some { status; _ } -> Error (Nak status)
;;

let read t ~address ~length =
  transact
    t
    (Control.Read { addr = address; len = length })
    ~op:Control.Op.read
    ~data_length:length
;;

let write t ~address ~data =
  transact t (Control.Write { addr = address; data }) ~op:Control.Op.write ~data_length:0
  |> Result.map ~f:(fun (_ : Bytes.t) -> ())
;;

(* The test transport: [pending] holds the scripted reply, and [sent] records the wire. *)

let fake () =
  let pending = Queue.create () in
  let sent = Buffer.create 16 in
  let send frame = Buffer.add_bytes sent frame in
  let transport =
    { send
    ; receive =
        (fun () ->
          match Queue.dequeue pending with
          | Some byte -> byte
          | None -> failwith "the script is out of bytes")
    ; resync = (fun () -> send (Bytes.make 1 Cobs.delimiter))
    }
  in
  transport, pending, sent
;;

let reply pending frame = String.iter (Bytes.to_string frame) ~f:(Queue.enqueue pending)

let hex bytes =
  String.concat
    ~sep:" "
    (List.map (Bytes.to_list bytes) ~f:(fun c -> Printf.sprintf "%02x" (Char.to_int c)))
;;

let show = function
  | Ok data -> Stdio.printf "ok %s\n" (hex data)
  | Error Garbled -> Stdio.print_endline "garbled"
  | Error (Nak status) ->
    Stdio.printf
      "nak %s\n"
      (match status with
       | Control.Status.Ok -> "ok"
       | Bad_op -> "bad-op"
       | Bad_address -> "bad-address"
       | Bad_length -> "bad-length")
;;

let%expect_test "a read, with the wire vectors of the hardware session" =
  let t, pending, sent = fake () in
  let response =
    Control.encode_response
      { op = Control.Op.read; status = Control.Status.Ok; data = Bytes.of_string "\x64" }
  in
  Stdio.printf "response frame %s\n" (hex response);
  reply pending response;
  show (read t ~address:Control.Reg.Ctl.velocity ~length:1);
  Stdio.printf "sent %s\n" (hex (Buffer.contents_bytes sent));
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
  show (read t ~address:Control.Reg.Ctl.velocity ~length:1);
  [%expect {| garbled |}]
;;

let%expect_test "the wrong shape is Garbled" =
  (* an Ok response with two data bytes answers a one-byte read; a wrong op echo is the
     same *)
  let t, pending, _ = fake () in
  reply
    pending
    (Control.encode_response
       { op = Control.Op.read
       ; status = Control.Status.Ok
       ; data = Bytes.of_string "\x64\x64"
       });
  show (read t ~address:Control.Reg.Ctl.velocity ~length:1);
  let t, pending, _ = fake () in
  reply
    pending
    (Control.encode_response
       { op = Control.Op.write; status = Control.Status.Ok; data = Bytes.create 0 });
  show (read t ~address:Control.Reg.Ctl.velocity ~length:1);
  [%expect {|
    garbled
    garbled
    |}]
;;

let%expect_test "a rejection is a rejection, not a garble" =
  let t, pending, _ = fake () in
  reply
    pending
    (Control.encode_response
       { op = Control.Op.read
       ; status = Control.Status.Bad_address
       ; data = Bytes.create 0
       });
  show (read t ~address:0x0000 ~length:1);
  [%expect {| nak bad-address |}]
;;

let%expect_test "resync sends one delimiter" =
  let t, _, sent = fake () in
  t.resync ();
  Stdio.printf "sent %s\n" (hex (Buffer.contents_bytes sent));
  [%expect {| sent 00 |}]
;;
