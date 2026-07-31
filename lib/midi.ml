open Hardcaml

let max_message_bytes = 3

module Message = struct
  type 'a t =
    { data : 'a [@bits max_message_bytes * 8]
    ; len : 'a [@bits 8]
    ; valid : 'a
    }
  [@@deriving hardcaml]
end
