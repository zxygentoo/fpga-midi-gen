(** The streaming COBS encoder: the hardware mirror of [Cobs.encode].

    The block reads the payload by index; thus the payload can live in a memory, a
    register file, or a function of the index. The producer decides.

    The payload is at most 127 bytes. Thus each group code is at most 0x80, and the block
    does not make the 254-byte groups of full COBS. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; frame_start : 'a
    (** a strobe; begins a frame of [payload_length] bytes. The block ignores it while
        [busy] is 1 *)
    ; payload_length : 'a
    (** the length of the payload in bytes; the block takes it with [frame_start]. Zero is
        legal: the frame is then the empty group and the delimiter, 01 00 *)
    ; read_data : 'a (** the payload byte at the [address] of the previous cycle *)
    ; hold : 'a (** from the consumer: 1 stalls the stream *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { address : 'a (** the payload index to read; a registered read *)
    ; data : 'a (** an encoded frame byte; the delimiter is the last *)
    ; valid : 'a (** the consumer takes [data] when [hold] is 0 *)
    ; busy : 'a (** high from [frame_start] until the delimiter is out *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
