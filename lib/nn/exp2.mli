(** The exp2 lookup: 2 raised to a nonpositive power, in Q15 — ONE MAGNITUDE A CYCLE.

    The unit takes the magnitude [nn] of the power, thus an [nn] of 0 gives 1.0 — 32768 in
    Q15 — and a larger [nn] gives a smaller weight. The power splits at the point: the
    fraction reads a table of 256 entries, and the integer part shifts the entry right.

    [nn] is Q[input_q]. Bits 11 to 4 read the table, bits 15 to 12 shift the entry, and
    bits 21 to 16 force 0: a magnitude of 16 or above gives 0, because an entry shifted
    that far is 0. The table therefore steps at 2^-8 of the power, and the low four
    fraction bits fall away. This truncation is a rule of the circuit, and
    [Quantized.For_test.exp2_q] — with the twins that read it through [jax/fixed.py] —
    holds the same one.

    EVERY PART OF THE ANSWER DERIVES FROM ONE REGISTERED MAGNITUDE: the entry, the shift
    and the zero test alike. That is what lets a caller present a magnitude every cycle —
    without it, a magnitude a cycle meets one class's entry under another class's shift,
    which is silently wrong music. It also keeps the caller's magnitude cone off the
    table's address pins, which is what era six's ring-3 timing round cost. The unit has
    no start and no busy: it is a register, a table and a shift.

    A CALLER READS [e] [latency] CYCLES AFTER ITS MAGNITUDE STANDS AT [nn], and it may
    present a new one meanwhile. A caller that holds instead reads the same weight; the
    registers cost it the extra cycle and nothing else. *)

open Hardcaml

(** cycles from a magnitude standing at [nn] to its weight at [e]: two, and a new
    magnitude may stand at [nn] every cycle. A caller's own tick chain counts from here
    rather than stating a number of its own. *)
val latency : int

(** the magnitude port: its width, and the Q it reads at. A caller saturates into the one
    and shifts into the other, thus it states neither number of its own. *)
val magnitude_bits : int

val input_q : int

module I : sig
  type 'a t =
    { clock : 'a
    ; nn : 'a
    (** the magnitude of the power, [magnitude_bits] wide: bits 11 to 0 the fraction, bits
        21 to 12 the integer part *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { e : 'a (** 2 ** -[nn], in Q15; 16 bits, unsigned *) }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
