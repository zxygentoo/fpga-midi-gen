(* mgt, the MIDI gen tool: reads and writes the control cells over the console UART.

   mgt [--device PATH] read ADDR LEN mgt [--device PATH] write ADDR BYTE.. poke
   [--device PATH] dump

   ADDR and BYTE take the OCaml integer syntax: 0xFFF9 or 65529. The device default is
   /dev/ttyUSB1, the Nexys 4 console UART. *)

module Abi = Mgen.Abi
module Control_transport = Mgen.Control_transport

let default_device = "/dev/ttyUSB1"
let baud = 115200

let usage () : 'a =
  prerr_endline "usage: mgt [--device PATH] (read ADDR LEN | write ADDR BYTE.. | dump)";
  exit 2
;;

let int_arg s =
  match int_of_string_opt s with
  | Some v -> v
  | None -> usage ()
;;

(* raw 8N1: no echo, no signals, no flow control, no translation *)
let serial_transport_exn device =
  let fd = Unix.openfile device [ Unix.O_RDWR; Unix.O_NOCTTY ] 0 in
  let tio = Unix.tcgetattr fd in
  Unix.tcsetattr
    fd
    Unix.TCSANOW
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
    };
  Unix.tcflush fd Unix.TCIOFLUSH;
  let send frame =
    let n = Bytes.length frame in
    if Unix.write fd frame 0 n <> n then failwith "short write to the serial port"
  in
  let receive () =
    let one = Bytes.create 1 in
    match Unix.read fd one 0 1 with
    | 1 -> Bytes.get one 0
    | _ -> failwith "the serial port closed"
  in
  { Control_transport.send; receive }
;;

let serial_transport device =
  try serial_transport_exn device with
  | Unix.Unix_error (error, _, _) ->
    Printf.eprintf "cannot open %s: %s\n" device (Unix.error_message error);
    exit 1
;;

let hex_line bytes =
  bytes
  |> Bytes.to_seq
  |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
  |> List.of_seq
  |> String.concat " "
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
      (fun (_, a, _) (_, b, _) -> compare a b)
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
    (fun (name, address, width) ->
      (* little-endian, as the ABI stores multi-byte values *)
      let value =
        List.fold_left
          (fun acc k ->
            (acc lsl 8) lor Bytes.get_uint8 bytes (address - Abi.Reg.Ctl.base + k))
          0
          (List.rev (List.init width Fun.id))
      in
      Printf.printf "%04x  %-8s  %d (0x%x)\n" address name value value)
    fields
;;

let () =
  let args = List.tl (Array.to_list Sys.argv) in
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
      Bytes.of_seq
        (List.to_seq (List.map (fun b -> Char.chr (int_arg b land 0xff)) bytes))
    in
    check (Control_transport.write t ~address:(int_arg address) ~data)
  | [ "dump" ] -> dump t
  | _ -> usage ()
;;
