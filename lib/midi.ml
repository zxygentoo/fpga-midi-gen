open Hardcaml

let max_message_bytes = 3
let note_on = 0x90
let note_off = 0x80
let release_velocity = 0x40

module Message = struct
  type 'a t =
    { data : 'a [@bits max_message_bytes * 8]
    ; len : 'a [@bits 8]
    ; valid : 'a
    }
  [@@deriving hardcaml]
end

(* the host side: [note_on] and [note_off] in the RHS below are the status constants of
   this module, not recursion *)
module Host = struct
  open Core

  let open_device path = Core_unix.openfile path ~mode:[ O_WRONLY ]

  let send fd bytes =
    let buf = Bytes.of_char_list (List.map bytes ~f:Char.of_int_exn) in
    ignore (Core_unix.write fd ~buf : int)
  ;;

  let note_on fd ~channel ~note ~velocity =
    send fd [ note_on lor channel; note; velocity ]
  ;;

  let note_off fd ~channel ~note =
    send fd [ note_off lor channel; note; release_velocity ]
  ;;
end
