(** The streaming COBS deframer: the hardware mirror of [Cobs.decode].

    The block holds no buffer: the consumer stores the decoded bytes. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; data : 'a (** the raw byte stream *)
    ; valid : 'a (** a strobe: [data] holds one stream byte *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { data : 'a (** the decoded body byte; read it when [valid] is 1 *)
    ; valid : 'a (** a strobe, one for each decoded byte *)
    ; frame_end : 'a (** a strobe at the delimiter of a well-formed frame *)
    ; abort : 'a (** a strobe at the delimiter of a frame that ends inside a group *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
