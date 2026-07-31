(** Serial transmitter, 8N1, LSB first. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; data : 'a (** the byte to send *)
    ; valid : 'a (** the block takes [data] when [valid] is 1 and [busy] is 0 *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { serial : 'a (** the serial line; it idles at 1 *)
    ; busy : 'a (** high while a frame is on the line *)
    }
  [@@deriving hardcaml]
end

(** [clocks_per_bit] selects the baud rate at elaboration time. *)
val create : clocks_per_bit:int -> Signal.t I.t -> Signal.t O.t
