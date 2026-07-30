(** Serial transmitter, 8N1, LSB first. *)

open Hardcaml

type t =
  { txd : Signal.t
  ; busy : Signal.t
  }

(** [clocks_per_bit] selects the baud rate at elaboration time. The block takes [data]
    when [valid] is 1 and [busy] is 0. The line idles at 1. *)
val create
  :  clocks_per_bit:int
  -> clock:Signal.t
  -> clear:Signal.t
  -> data:Signal.t
  -> valid:Signal.t
  -> t
