(** The memory bridge: the wire-protocol engine and the control register file.

    The bridge consumes the byte stream from the host UART, decodes COBS frames, executes
    read and write requests against the register file, and produces the response byte
    stream for the transmitter. The behavior is the one of [docs/abi.md]: a frame that
    does not decode gets no response, a write applies its bytes in the sequence of
    increasing addresses, and a rejected access changes no cell. *)

open Hardcaml

type t =
  { tx_data : Signal.t
  ; tx_valid : Signal.t (** the transmitter takes the byte when it is not busy *)
  ; cells : Signal.t
  (** the register file as one vector; the cell at [Abi.Reg.Ctl.base] is the low byte *)
  }

val create
  :  clock:Signal.t
  -> clear:Signal.t
  -> rx_data:Signal.t
  -> rx_valid:Signal.t
  -> tx_busy:Signal.t
  -> t
