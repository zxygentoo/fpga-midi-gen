(* mgt, the MIDI gen tool: reads and writes the control cells over the console UART.

   Each subcommand takes -help. ADDR and BYTE take the OCaml integer syntax: 0x09 or 9.
   The device default is /dev/ttyUSB1, the Nexys 4 console UART. *)

open Core
module Bytes_util = Mgen.Bytes_util
module Control_intf = Mgen.Control_intf
module Control_transport = Mgen.Control_transport
module Midi = Mgen.Midi

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

(* The board takes any value that fits a cell: a SEED of 0 holds the PRNG state at 0 and
   every voice on its root, and a VELOCITY above 127 sets bit 7, which MIDI reads as a
   status and not as data. [Control_intf.Reg] states the range of each cell that has one,
   thus the tool refuses the write here.

   A write can cover a part of a cell, thus the check reads the cell and applies the
   pending bytes before it judges: the default seed is 2a 00 00 00, and `write 0x05 0`
   alone would make it 0. *)
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

(* the doorbell: poll MIDI_GO to 0, ring with one ascending burst, poll to 0 again as the
   confirmation that the send ran. The poll before the ring is the host-control rule; the
   poll after it bounds the exit at "the message went out". *)
let doorbell t message =
  let midi_go_clear () =
    let b =
      check (Control_transport.read t ~address:Control_intf.Reg.midi_go ~length:1)
    in
    Bytes_util.byte b 0 = 0
  in
  let wait_clear () =
    (* one poll is about 1 ms of wire time, and a message takes at most 1 ms *)
    let rec wait tries =
      if not (midi_go_clear ())
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
  let address, data = Control_intf.build_doorbell message in
  check (Control_transport.write t ~address ~data);
  wait_clear ()
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

let doorbell_command =
  Command.basic
    ~summary:"send the BYTEs to the MIDI output as one test message"
    (let%map_open.Command device = device_param
     and bytes = anon (sequence ("BYTE" %: byte_arg)) in
     fun () ->
       ensure
         "the number of BYTEs"
         ~lower:1
         ~upper:Midi.max_message_bytes
         (List.length bytes);
       doorbell (transport device) bytes)
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
    [ "read", read_command
    ; "write", write_command
    ; "doorbell", doorbell_command
    ; "dump", dump_command
    ]
;;

let () = Command_unix.run command
