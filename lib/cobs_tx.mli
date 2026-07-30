(** The streaming COBS framer: the hardware mirror of [Cobs.encode].

    The block reads the payload by index: it presents [rd_addr], and [rd_data] must hold
    that byte on the subsequent cycle (a registered read). Thus the payload can live in a
    memory, a register file, or a function of the index; the producer decides.

    A [start] strobe with [length] begins a frame. The block emits the encoded bytes and
    the final delimiter on [tx_data]/[tx_valid], and the transmitter takes each byte when
    [tx_busy] is 0. [busy] stays high until the delimiter is out. *)

open Hardcaml

type t =
  { rd_addr : Signal.t
  ; tx_data : Signal.t
  ; tx_valid : Signal.t
  ; busy : Signal.t
  }

val create
  :  clock:Signal.t
  -> clear:Signal.t
  -> start:Signal.t
  -> length:Signal.t
  -> rd_data:Signal.t
  -> tx_busy:Signal.t
  -> t
