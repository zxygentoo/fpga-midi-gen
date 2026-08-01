(** The PRNG: Marsaglia xorshift32 as a circuit.

    One step is three shift-and-XOR layers, combinational; the state register takes the
    new value at a [step] strobe. The reference is [Pink.Prng], and the vector test drives
    the two side by side.

    The state walks all 32-bit values except 0. The clear puts 1 into the state: the state
    has no use before the first [load], and 1 keeps the no-zero rule. [load] wins over
    [step] in the same cycle. *)

open Hardcaml

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
