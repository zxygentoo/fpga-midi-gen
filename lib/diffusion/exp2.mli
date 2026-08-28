(** The exp2 lookup, ONE MAGNITUDE A CYCLE: era six's fork of [Mgen_nn.Exp2].

    THE FORK EXISTS BECAUSE THE SHARED UNIT NEEDS ITS MAGNITUDE HELD. That unit registers
    the table entry but reads the shift and the zero test from [nn] as it stands, thus a
    magnitude a cycle meets one class's entry under another class's shift — silently wrong
    music. The draw walks 48 classes and wants one a cycle, thus all three derive from one
    registered [nn] here. The same register keeps the caller's magnitude cone off the
    table's address pins, which is what the timing round of ring 3 cost.

    A CALLER THAT HOLDS ITS MAGNITUDE READS THE SAME WEIGHT EITHER WAY, thus the backport
    to the two shipped eras is open and the gate below is its evidence.

    The table is the shared unit's, unmoved: bits 11 to 4 index its 256 entries, bits 15
    to 12 shift the entry right, and a magnitude of 16 or above is zero.
    [Mgen_nn.Quantized.exp2_of_magnitude] is the reference. *)

open Hardcaml

(** cycles from a magnitude at [nn] to its weight at [e]: two, and a new magnitude may
    stand at [nn] every cycle *)
val latency : int

(** the magnitude port: its width, and the Q it reads at. A caller saturates into the one
    and shifts into the other, thus it states neither number of its own. *)
val magnitude_bits : int

val input_q : int

module I : sig
  type 'a t =
    { clock : 'a
    ; nn : 'a
    (** the magnitude of the power: bits 11 to 0 the fraction, bits 21 to 12 the integer
        part *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { e : 'a (** 2 ** -[nn], in Q15; 16 bits, unsigned *) }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
