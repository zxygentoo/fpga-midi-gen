(** The PRNG: Marsaglia xorshift32, in software and as a circuit.

    One step is three shift-and-XOR layers. The software gives the reference, the circuit
    computes the same recurrence combinationally, and the vector test in this module
    drives the two side by side. Thus the definition of the walk has one home.

    The state walks all 32-bit values except 0. *)

open Hardcaml

type t

(** [create ~seed] raises [Invalid_argument] when [seed] does not fit 32 bits or is 0, the
    rule of the SEED cell. *)
val create : seed:int -> t

(** [next t] is the new state and one draw: the low 8 bits of the new state. *)
val next : t -> t * int

(** The circuit. The state register takes the new value at a [step] strobe. The clear puts
    1 into the state: the state has no use before the first [load], and 1 keeps the
    no-zero rule. [load] wins over [step] in the same cycle. *)
module Rtl : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; load : 'a (** a strobe: the state takes [seed] *)
      ; seed : 'a (** 32 bits *)
      ; step : 'a (** a strobe: the state advances one time *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { value : 'a (** the 32-bit state; a draw is the low 8 bits *) }
    [@@deriving hardcaml]
  end

  val create : Signal.t I.t -> Signal.t O.t
end
