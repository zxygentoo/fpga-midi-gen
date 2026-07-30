(** The streaming COBS deframer: the hardware mirror of [Cobs.decode].

    The block consumes the raw byte stream and emits the decoded body bytes. At each
    delimiter it strobes [frame_end] for a well-formed frame, or [abort] for a frame that
    ends inside a group. The block holds no buffer: the consumer stores the bytes. *)

open Hardcaml

type t =
  { data : Signal.t
  ; valid : Signal.t (** a decoded body byte, one strobe for each byte *)
  ; frame_end : Signal.t
  ; abort : Signal.t
  }

val create : clock:Signal.t -> clear:Signal.t -> data:Signal.t -> valid:Signal.t -> t
