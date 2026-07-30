(** Serial receiver, 8N1, LSB first, with a two-flop synchronizer on the input. *)

open Hardcaml

type t =
  { data : Signal.t
  ; valid : Signal.t (** high for one cycle when a byte is complete *)
  }

(** [clocks_per_bit] selects the baud rate at elaboration time. The block samples each bit
    at its center, verifies the start bit, and discards a frame with a bad stop bit. *)
val create : clocks_per_bit:int -> clock:Signal.t -> clear:Signal.t -> rxd:Signal.t -> t
