(** The control port: the wire-protocol engine and the control register file.

    The behavior is the one of [docs/abi.md]: a frame that does not decode gets no
    response, a write applies its bytes in the sequence of increasing addresses, and a
    rejected access changes no cell.

    After power-on and after clear, the port writes the control defaults into the register
    file (state [init]); then it serves one transaction at a time.

    A write applies one byte each cycle: a multi-byte value is torn between these cycles.
    The wire protocol cannot observe this, because the response comes after the last byte.
    A hardware block that consumes a multi-byte cell must take the value on the write of
    its last byte, as the SEED rule of the ABI does, or sample the cells only when [state]
    is [State.ready]. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a (** the byte stream from the host UART *)
    ; in_valid : 'a (** a strobe: [in_data] holds one stream byte *)
    ; hold : 'a (** from the consumer: 1 stalls the stream *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { out_data : 'a (** the response byte stream: COBS frames with their delimiters *)
    ; out_valid : 'a (** the transmitter takes [out_data] when [hold] is 0 *)
    ; state : 'a (** [State.init], [State.ready] or [State.busy] *)
    }
  [@@deriving hardcaml]
end

(** The values of [O.state]. *)
module State : sig
  (** the port loads the control defaults; the cells are not valid *)
  val init : int

  (** idle; the cells are valid and stable *)
  val ready : int

  (** a transaction is in progress; a write can tear a multi-byte cell *)
  val busy : int
end

val create : Signal.t I.t -> Signal.t O.t
