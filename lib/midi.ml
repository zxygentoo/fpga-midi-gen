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
