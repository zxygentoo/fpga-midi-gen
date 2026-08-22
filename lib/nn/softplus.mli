(** The softplus correction table: the second nonlinearity era five adds.

    [dt] is the one place softplus runs — [heads] values in a layer — and it is split so
    that a table carries only what a table has to:

    {v
      softplus(v) = relu(v) + ln(1 + exp(-|v|))
    v}

    The ramp is exact and it carries the whole of a large input; the correction falls from
    ln 2 at zero to one unit of Q12 at |v| = 8, which is the largest magnitude a signed
    Q12 int16 takes. **This unit gives the correction alone.** The caller adds the ramp,
    which costs a mux, and clamps the sum.

    The table is 256 rows of 16 bits over the magnitude, and an entry is the value at the
    CENTRE of its row. [Quantized.Constants] holds the entries and the row rule.

    The contract, which is [Exp2]'s:

    - There is no [start] and no [busy]. The read registers, thus [v] must stand for two
      cycles and [c] is whole on the second. The cycle after a change carries the old row.
    - The sign of [v] and its low seven bits fall away. Both are rules of the arithmetic
      and the reference holds them too. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; v : 'a (** 16 bits, signed Q12 *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { c : 'a (** 16 bits, unsigned Q12: ln(1 + exp(-|v|)) *) }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
