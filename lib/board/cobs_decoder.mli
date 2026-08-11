(** The streaming COBS decoder: the hardware mirror of [Cobs.decode].

    The block holds no buffer: the consumer stores the decoded bytes. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a (** the raw byte stream *)
    ; in_valid : 'a (** a strobe: [in_data] holds one stream byte *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { out_data : 'a (** the decoded body byte; read it when [out_valid] is 1 *)
    ; out_valid : 'a (** a strobe, one for each decoded byte *)
    ; frame_end : 'a (** a strobe at the delimiter of a well-formed frame *)
    ; abort : 'a (** a strobe at the delimiter of a frame that ends inside a group *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
