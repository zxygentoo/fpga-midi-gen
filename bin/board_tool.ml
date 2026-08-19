(* board_tool: reads and writes the control cells of the board over the console UART.

   Each subcommand takes -help. ADDR and BYTE take the OCaml integer syntax: 0x09 or 9.
   The device default is /dev/ttyUSB1, the Nexys 4 console UART. *)

open Core
module Bytes_util = Mgen_core.Bytes_util
module Control_intf = Mgen_core.Control_intf
module Control_transport = Mgen_board.Control_transport

let default_device = "/dev/ttyUSB1"
let baud = 115200

(* The tool rejects a value that the wire format cannot carry, and it does that here: the
   codec raises for a value outside its range, and an exception is not a diagnostic for a
   person at a command line. *)
let ensure name ~lower ~upper v =
  if v < lower || v > upper
  then (
    Printf.eprintf "%s must be %d to %d, not %d\n" name lower upper v;
    exit 2)
;;

let checked name ~lower ~upper v =
  ensure name ~lower ~upper v;
  v
;;

let address_arg =
  Command.Arg_type.create (fun s -> checked "ADDR" ~lower:0 ~upper:0xFF (Int.of_string s))
;;

let length_arg =
  Command.Arg_type.create (fun s ->
    checked "LEN" ~lower:1 ~upper:Control_intf.Constants.max_data_len (Int.of_string s))
;;

let byte_arg =
  Command.Arg_type.create (fun s -> checked "BYTE" ~lower:0 ~upper:0xFF (Int.of_string s))
;;

let device_param =
  let open Command.Param in
  flag
    "-device"
    (optional_with_default default_device string)
    ~doc:"PATH the console UART (default /dev/ttyUSB1)"
;;

let transport device =
  let fd =
    try Core_unix.openfile device ~mode:[ O_RDWR; O_NOCTTY ] with
    | Core_unix.Unix_error (error, _, _) ->
      Printf.eprintf "cannot open %s: %s\n" device (Core_unix.Error.message error);
      exit 1
  in
  let t = Control_transport.serial ~baud fd in
  Control_transport.resync t;
  t
;;

let check = function
  | Ok v -> v
  | Error Control_transport.Garbled ->
    prerr_endline "garbled response: run the command again";
    exit 1
  | Error (Control_transport.Nak status) ->
    Printf.eprintf "rejected: %s\n" (Control_intf.Status.to_string status);
    exit 1
;;

(* The board takes any value that fits a cell: a VELOCITY above 127 sets bit 7, which MIDI
   reads as a status and not as data, and a CHANNEL above 15 does not fit the field.
   [Control_intf.Reg] states the range of each cell that has one, thus the tool refuses
   the write here. SEED has no range: the slide switches can set 0 and the board accepts
   it, thus a table that refused 0 would only disagree with the panel.

   A write can cover a part of a cell, thus the check reads the cell and applies the
   pending bytes before it judges: a two-byte cell takes a write of one byte, and the byte
   that the write does not name keeps the value the board holds. *)
let refuse_out_of_range t ~address ~data =
  let n = Bytes.length data in
  List.iter Control_intf.Reg.fields ~f:(fun (f : Control_intf.Reg.field) ->
    match f.bounds with
    | None -> ()
    | Some { lower; upper } ->
      if address < f.address + f.width && f.address < address + n
      then (
        let current =
          check (Control_transport.read t ~address:f.address ~length:f.width)
        in
        let after =
          Bytes.init f.width ~f:(fun k ->
            let cell = f.address + k in
            if cell >= address && cell < address + n
            then Bytes.get data (cell - address)
            else Bytes.get current k)
        in
        let value = Bytes_util.uint_le after ~pos:0 ~width:f.width in
        if value < lower || value > upper
        then (
          Printf.eprintf "%s must be %d to %d, not %d\n" f.name lower upper value;
          exit 2)))
;;

let dump t =
  let bytes =
    check
      (Control_transport.read
         t
         ~address:Control_intf.Reg.base
         ~length:Control_intf.Reg.size)
  in
  Printf.printf "%04x  %s\n" Control_intf.Reg.base (Bytes_util.hex bytes);
  List.iter Control_intf.Reg.fields ~f:(fun (f : Control_intf.Reg.field) ->
    let value =
      Bytes_util.uint_le bytes ~pos:(f.address - Control_intf.Reg.base) ~width:f.width
    in
    Printf.printf "%04x  %-8s  %d (0x%x)\n" f.address f.name value value)
;;

let read_command =
  Command.basic
    ~summary:"read LEN cells at ADDR"
    (let%map_open.Command device = device_param
     and address = anon ("ADDR" %: address_arg)
     and length = anon ("LEN" %: length_arg) in
     fun () ->
       let data = check (Control_transport.read (transport device) ~address ~length) in
       print_endline (Bytes_util.hex data))
;;

let write_command =
  Command.basic
    ~summary:"write the BYTEs to the cells at ADDR"
    (let%map_open.Command device = device_param
     and address = anon ("ADDR" %: address_arg)
     and bytes = anon (sequence ("BYTE" %: byte_arg)) in
     fun () ->
       ensure
         "the number of BYTEs"
         ~lower:1
         ~upper:Control_intf.Constants.max_data_len
         (List.length bytes);
       let data = Bytes.of_char_list (List.map bytes ~f:Char.of_int_exn) in
       let t = transport device in
       refuse_out_of_range t ~address ~data;
       check (Control_transport.write t ~address ~data))
;;

let dump_command =
  Command.basic
    ~summary:"read every cell and name each register"
    (let%map_open.Command device = device_param in
     fun () -> dump (transport device))
;;

let command =
  Command.group
    ~summary:"the control cells of the FPGA, over the console UART"
    [ "read", read_command; "write", write_command; "dump", dump_command ]
;;

let () = Command_unix.run command
