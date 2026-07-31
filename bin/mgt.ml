(* mgt, the MIDI gen tool: reads and writes the control cells over the console UART.

   mgt [--device PATH] read ADDR LEN mgt [--device PATH] write ADDR BYTE.. poke
   [--device PATH] dump

   ADDR and BYTE take the OCaml integer syntax: 0xFFF9 or 65529. The device default is
   /dev/ttyUSB1, the Nexys 4 console UART. *)

open Core
module Abi = Mgen.Abi
module Control_transport = Mgen.Control_transport

let default_device = "/dev/ttyUSB1"
let baud = 115200

let usage () : 'a =
  prerr_endline "usage: mgt [--device PATH] (read ADDR LEN | write ADDR BYTE.. | dump)";
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

let hex_line bytes =
  String.concat
    ~sep:" "
    (List.map (Bytes.to_list bytes) ~f:(fun c -> Printf.sprintf "%02x" (Char.to_int c)))
;;

let check = function
  | Ok v -> v
  | Error Control_transport.Garbled ->
    prerr_endline "garbled response: run the command again";
    exit 1
  | Error (Control_transport.Nak status) ->
    let name =
      match status with
      | Abi.Status.Ok -> "ok"
      | Bad_op -> "bad op"
      | Bad_address -> "bad address"
      | Bad_length -> "bad length"
    in
    Printf.eprintf "rejected: %s\n" name;
    exit 1
;;

let dump t =
  let bytes =
    check (Control_transport.read t ~address:Abi.Reg.Ctl.base ~length:Abi.Reg.Ctl.size)
  in
  Printf.printf "%04x  %s\n" Abi.Reg.Ctl.base (hex_line bytes);
  let fields =
    List.sort
      ~compare:(fun (_, a, _) (_, b, _) -> Int.compare a b)
      [ "run", Abi.Reg.Ctl.run, 1
      ; "channel", Abi.Reg.Ctl.channel, 1
      ; "step_ms", Abi.Reg.Ctl.step_ms, 2
      ; "gate_ms", Abi.Reg.Ctl.gate_ms, 2
      ; "velocity", Abi.Reg.Ctl.velocity, 1
      ; "seed", Abi.Reg.Ctl.seed, 4
      ; "msg_go", Abi.Reg.Ctl.msg_go, 1
      ; "msg_len", Abi.Reg.Ctl.msg_len, 1
      ; "msg", Abi.Reg.Ctl.msg, 3
      ]
  in
  List.iter
    ~f:(fun (name, address, width) ->
      (* little-endian, as the ABI stores multi-byte values *)
      let value =
        List.fold
          (List.rev (List.init width ~f:Fn.id))
          ~init:0
          ~f:(fun acc k ->
            (acc lsl 8) lor Char.to_int (Bytes.get bytes (address - Abi.Reg.Ctl.base + k)))
      in
      Printf.printf "%04x  %-8s  %d (0x%x)\n" address name value value)
    fields
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
    print_endline (hex_line data)
  | "write" :: address :: (_ :: _ as bytes) ->
    let data =
      Bytes.of_string
        (String.of_char_list
           (List.map bytes ~f:(fun b -> Char.of_int_exn (int_arg b land 0xff))))
    in
    check (Control_transport.write t ~address:(int_arg address) ~data)
  | [ "dump" ] -> dump t
  | _ -> usage ()
;;
