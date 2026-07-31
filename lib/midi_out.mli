(** The MIDI transmitter: one message to the line.

    The block takes one message at a time. It latches the message at the transfer, gives
    each byte to the serial transmitter, and holds [ready] at 0 until the last stop bit is
    on the line. Therefore no other source can put a byte between the bytes of the message
    that goes out.

    [Midi.Message.len] must be 1 to [Midi.max_message_bytes]. The walk also ends at the
    last byte, thus a length outside the range cannot hold the block. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; message : 'a Midi.Message.t (** the block takes it when [ready] is 1 *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { serial : 'a (** the MIDI line; it idles at 1 *)
    ; ready : 'a (** 1 when the block can take a message *)
    }
  [@@deriving hardcaml]
end

(** [clocks_per_bit] selects the baud rate at elaboration time. *)
val create : clocks_per_bit:int -> Signal.t I.t -> Signal.t O.t
