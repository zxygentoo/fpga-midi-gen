open Core
open Hardcaml

let max_message_bytes = 3
let note_on = 0x90
let note_off = 0x80
let release_velocity = 0x40

(* the bytes of a channel voice message: the status low nibble carries the channel *)
let note_on_bytes ~channel ~note ~velocity = [ note_on lor channel; note; velocity ]
let note_off_bytes ~channel ~note = [ note_off lor channel; note; release_velocity ]
let open_device path = Core_unix.openfile path ~mode:[ O_WRONLY ]

let send fd bytes =
  let buf = Bytes.of_char_list (List.map bytes ~f:Char.of_int_exn) in
  ignore (Core_unix.write fd ~buf : int)
;;

let send_note_on fd ~channel ~note ~velocity =
  send fd (note_on_bytes ~channel ~note ~velocity)
;;

let send_note_off fd ~channel ~note = send fd (note_off_bytes ~channel ~note)

module Rtl = struct
  open Signal

  module Message = struct
    type 'a t =
      { data : 'a [@bits max_message_bytes * 8]
      ; len : 'a [@bits 8]
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  (* the same layout as [note_on_bytes] and [note_off_bytes], in the order of
     [Message.data]: the first byte is in the low 8 bits *)
  let channel_voice_data ~status ~channel ~data1 ~data2 =
    concat_lsb
      [ concat_msb [ of_unsigned_int ~width:4 (status lsr 4); channel ]; data1; data2 ]
  ;;

  let note_on_data ~channel ~pitch ~velocity =
    channel_voice_data ~status:note_on ~channel ~data1:pitch ~data2:velocity
  ;;

  let note_off_data ~channel ~pitch =
    channel_voice_data
      ~status:note_off
      ~channel
      ~data1:pitch
      ~data2:(of_unsigned_int ~width:8 release_velocity)
  ;;
end
