(** The sigmoid table: one of the two nonlinearities era five adds.

    SiLU is [v * sigmoid(v)], and it runs twice in a layer — over the convolution output
    and over the gate — thus one table and one multiply carry both. The table is 256 rows
    of 16 bits: the sigmoid in Q15 over a signed Q12 input, which is |v| < 8 exactly, and
    an entry is the sigmoid at the CENTRE of its row. The centres are symmetric about
    zero, thus a value and its negative weigh 2^15 together and the identity
    [sigmoid(-v) = 1 - sigmoid(v)] survives the quantization.

    [Quantized.Constants] holds the entries and the row rule; this is the ROM around them.

    The contract, which is [Exp2]'s:

    - There is no [start] and no [busy]. The read registers, thus [v] must stand for two
      cycles and [s] is whole on the second. The cycle after a change carries the old row.
    - The low eight bits of [v] fall away. That truncation is a rule of the arithmetic and
      the reference holds it too. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; v : 'a (** 16 bits, signed Q12 *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { s : 'a (** 16 bits, unsigned Q15 *) } [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
