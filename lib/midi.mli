(** MIDI: the message that a source gives to the transmitter, and the senders of the host.

    A source holds [data] and [len] while [valid] is 1. The transfer is the one cycle in
    which [valid] and the [ready] of the sink are both 1. After that cycle the source is
    free. When [valid] is 0, [data] and [len] have no meaning.

    The interface carries a complete message, and not a byte stream. Therefore a source
    cannot put its bytes between the bytes of another source, and the merge at message
    boundaries is the shape of the interface.

    The senders write to the USB MIDI device of the synthesizer. The bytes go directly to
    the rawmidi device; there is no driver stack and no timing. *)

open Hardcaml

(** The maximum bytes in one message. A channel voice message is at most 3 bytes, and a
    real-time message is 1 byte. A System Exclusive message does not fit. *)
val max_message_bytes : int

(** The Note On status. The high nibble is the status, and the low nibble carries the
    channel. *)
val note_on : int

val note_off : int

(** The velocity byte of each Note Off that this design sends. *)
val release_velocity : int

(** [open_device path] opens the device for writing. It raises on a system error. *)
val open_device : string -> Core_unix.File_descr.t

val send_note_on
  :  Core_unix.File_descr.t
  -> channel:int
  -> note:int
  -> velocity:int
  -> unit

val send_note_off : Core_unix.File_descr.t -> channel:int -> note:int -> unit

(** The hardware side: the interface that a message source gives to a sink. *)
module Rtl : sig
  module Message : sig
    type 'a t =
      { data : 'a (** the message bytes; the first byte is in the low 8 bits *)
      ; len : 'a (** the number of bytes, 1 to [max_message_bytes] *)
      ; valid : 'a (** the sink takes the message when its [ready] is also 1 *)
      }
    [@@deriving hardcaml]
  end
end
