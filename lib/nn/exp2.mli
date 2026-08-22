(** The exp2 lookup: 2 raised to a nonpositive power, in Q15.

    The unit takes the magnitude [nn] of the power, thus an [nn] of 0 gives 1.0 — 32768 in
    Q15 — and a larger [nn] gives a smaller weight. The power splits at the point: the
    fraction reads a table of 256 entries, and the integer part shifts the entry right.

    [nn] is Q12. Bits 11 to 4 read the table, bits 15 to 12 shift the entry, and bits 21
    to 16 force 0: a magnitude of 16 or above gives 0, because an entry shifted that far
    is 0. The table therefore steps at 2^-8 of the power, and the low four fraction bits
    fall away. This truncation is a rule of the circuit, and the reference
    [Quantized.Engine.exp2_q] holds the same one.

    The unit has no start and no busy: it is a table and a shift. The table read stands in
    a register, thus [nn] must hold for two cycles and [e] is whole on the second. That
    register has no clear; a stale entry can reach nothing, because the caller waits. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; nn : 'a
    (** the magnitude of the power, 22 bits: Q12, thus bits 11 to 0 are the fraction and
        bits 21 to 12 the integer part *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { e : 'a (** 2 ** -[nn], in Q15; 16 bits, unsigned *) }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
