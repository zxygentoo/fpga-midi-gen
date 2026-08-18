(** The control port: the wire-protocol engine.

    The behavior is the one of [docs/host_control.md]: a frame that does not decode gets
    no response, a write applies its bytes in the sequence of increasing addresses, and a
    rejected access changes no cell.

    The block serves one transaction at a time. While it serves a transaction, it ignores
    the input stream: a frame in that interval gets no response, and the sender must
    repeat it.

    The block holds no cell. It fills the shadow copy of [Control_regs] one byte in each
    cycle, and it strobes [commit] at the end of the burst; thus the whole write applies
    at one time and no value of more than one byte tears. It reads a cell for the response
    with [read_address] and [read_data].

    The port needs no start-up time. The cells carry their power-on value in the
    bitstream, therefore the first request gets a correct answer. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a (** the byte stream from the host UART *)
    ; in_valid : 'a (** a strobe: [in_data] holds one stream byte *)
    ; hold : 'a (** from the host transmitter: 1 stalls the response stream *)
    ; read_data : 'a (** from [Control_regs]: the byte at [read_address] *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { out_data : 'a (** the response byte stream: COBS frames with their delimiters *)
    ; out_valid : 'a (** the transmitter takes [out_data] when [hold] is 0 *)
    ; write_enable : 'a (** writes one byte into the shadow copy *)
    ; write_address : 'a (** the cell index to write *)
    ; write_data : 'a (** the byte to write *)
    ; commit : 'a (** a strobe at the end of the burst *)
    ; read_address : 'a (** the cell index that the response reads *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
