(* mgt, the MIDI gen tool: reads and writes the control cells over the console UART.

   mgt [--device PATH] read ADDR LEN mgt [--device PATH] write ADDR BYTE.. mgt
   [--device PATH] doorbell BYTE.. mgt [--device PATH] dump

   ADDR and BYTE take the OCaml integer syntax: 0xFFF9 or 65529. The device default is
   /dev/ttyUSB1, the Nexys 4 console UART. *)

open Core
module Bytes_util = Mgen.Bytes_util
module Control = Mgen.Control
module Control_transport = Mgen.Control_transport

let default_device = "/dev/ttyUSB1"
let baud = 115200

let usage () : 'a =
  prerr_endline
    "usage: mgt [--device PATH] (read ADDR LEN | write ADDR BYTE.. | doorbell BYTE.. | \
     dump)";
  exit 2
;;

let int_arg s =
  match Option.try_with (fun () -> Int.of_string s) with
  | Some v -> v
  | None -> usage ()
;;

let serial_transport_exn device =
  let fd = Core_unix.openfile device ~mode:[ O_RDWR; O_NOCTTY ] in
  Control_transport.serial ~baud fd
;;

let serial_transport device =
  try serial_transport_exn device with
  | Core_unix.Unix_error (error, _, _) ->
    Printf.eprintf "cannot open %s: %s\n" device (Core_unix.Error.message error);
    exit 1
;;

let check = function
  | Ok v -> v
  | Error Control_transport.Garbled ->
    prerr_endline "garbled response: run the command again";
    exit 1
  | Error (Control_transport.Nak status) ->
    Printf.eprintf "rejected: %s\n" (Control.Status.to_string status);
    exit 1
;;

let dump t =
  let bytes =
    check
      (Control_transport.read
         t
         ~address:Control.Reg.Ctl.base
         ~length:Control.Reg.Ctl.size)
  in
  Printf.printf "%04x  %s\n" Control.Reg.Ctl.base (Bytes_util.hex bytes);
  let fields =
    List.sort
      ~compare:(fun (_, a, _) (_, b, _) -> Int.compare a b)
      [ "run", Control.Reg.Ctl.run, 1
      ; "channel", Control.Reg.Ctl.channel, 1
      ; "step_ms", Control.Reg.Ctl.step_ms, 2
      ; "gate_ms", Control.Reg.Ctl.gate_ms, 2
      ; "velocity", Control.Reg.Ctl.velocity, 1
      ; "seed", Control.Reg.Ctl.seed, 4
      ; "msg_go", Control.Reg.Ctl.msg_go, 1
      ; "msg_len", Control.Reg.Ctl.msg_len, 1
      ; "msg", Control.Reg.Ctl.msg, Mgen.Midi.max_message_bytes
      ]
  in
  List.iter
    ~f:(fun (name, address, width) ->
      let value = Bytes_util.uint_le bytes ~pos:(address - Control.Reg.Ctl.base) ~width in
      Printf.printf "%04x  %-8s  %d (0x%x)\n" address name value value)
    fields
;;

(* the doorbell: poll MSG_GO to 0, ring with one ascending burst, poll to 0 again as the
   confirmation that the send ran. The poll before the ring is the host-control rule; the
   poll after it bounds the exit at "the message went out". *)
let doorbell t bytes =
  let n = List.length bytes in
  if n < 1 || n > Mgen.Midi.max_message_bytes then usage ();
  let msg_go_clear () =
    let b = check (Control_transport.read t ~address:Control.Reg.Ctl.msg_go ~length:1) in
    Char.to_int (Bytes.get b 0) = 0
  in
  let wait_clear () =
    (* one poll is about 1 ms of wire time, and a message takes at most 1 ms *)
    let rec wait tries =
      if not (msg_go_clear ())
      then
        if tries = 0
        then (
          prerr_endline "a message waits and does not go out";
          exit 1)
        else wait (tries - 1)
    in
    wait 100
  in
  wait_clear ();
  let burst = Bytes.make (Control.Reg.Ctl.msg_go - Control.Reg.Ctl.msg + 1) '\x00' in
  List.iteri bytes ~f:(fun k b -> Bytes.set burst k (Char.of_int_exn b));
  Bytes.set burst (Control.Reg.Ctl.msg_len - Control.Reg.Ctl.msg) (Char.of_int_exn n);
  Bytes.set burst (Control.Reg.Ctl.msg_go - Control.Reg.Ctl.msg) '\x01';
  check (Control_transport.write t ~address:Control.Reg.Ctl.msg ~data:burst);
  wait_clear ()
;;

let () =
  let args = List.tl_exn (Array.to_list (Sys.get_argv ())) in
  let device, args =
    match args with
    | "--device" :: path :: rest -> path, rest
    | _ -> default_device, args
  in
  let t = serial_transport device in
  Control_transport.resync t;
  match args with
  | [ "read"; address; length ] ->
    let data =
      check (Control_transport.read t ~address:(int_arg address) ~length:(int_arg length))
    in
    print_endline (Bytes_util.hex data)
  | "write" :: address :: (_ :: _ as bytes) ->
    let data =
      Bytes.of_string
        (String.of_char_list
           (List.map bytes ~f:(fun b -> Char.of_int_exn (int_arg b land 0xff))))
    in
    check (Control_transport.write t ~address:(int_arg address) ~data)
  | "doorbell" :: (_ :: _ as bytes) ->
    doorbell t (List.map bytes ~f:(fun b -> int_arg b land 0xff))
  | [ "dump" ] -> dump t
  | _ -> usage ()
;;
