(** The streaming COBS framer: the hardware mirror of [Cobs.encode].

    The block reads the payload by index; thus the payload can live in a memory, a
    register file, or a function of the index. The producer decides. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a (** a strobe; begins a frame of [length] payload bytes *)
    ; length : 'a (** the payload length; the block takes it with [start] *)
    ; rd_data : 'a (** the payload byte at the [rd_addr] of the previous cycle *)
    ; tx_busy : 'a (** from the transmitter: 1 while it sends *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { rd_addr : 'a (** the payload index to read; a registered read *)
    ; tx_data : 'a (** an encoded frame byte; the delimiter is the last *)
    ; tx_valid : 'a (** the transmitter takes [tx_data] when [tx_busy] is 0 *)
    ; busy : 'a (** high from [start] until the delimiter is out *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
