(** The exp2 lookup, ONE MAGNITUDE A CYCLE: era six's fork of [Mgen_nn.Exp2].

    The shared unit registers its table entry but takes the shift and the zero test from
    [nn] AS IT STANDS, thus its contract is that [nn] holds for two cycles and [e] is
    whole on the second. Eras four and five hold it — their schedules dwell on an op for
    many cycles — and era six does not: the draw walks 48 classes and would like one a
    cycle. Under the shared unit a magnitude a cycle reads one class's table entry under
    another class's shift, which is silently wrong music.

    **THE FORK REGISTERS THE SHIFT AND THE ZERO TEST BESIDE THE ENTRY**, thus a magnitude
    may change every cycle and [e] stands behind it. That is two flip-flops.

    **THE FORK ALSO REGISTERS THE ADDRESS BEFORE THE MEMORY** — era four's rule, which the
    first fork broke: ring 3 of the machine round read the caller's whole magnitude cone
    on the table's address pins, the worst path of the build. [nn] is taken whole into one
    register, thus the entry, the shift and the zero test derive from one value and the
    caller's cone ends on a flip-flop and not on a memory.

    **THE CHANGE DOES NOT MOVE WHAT A HOLDING CALLER READS.** Where [nn] holds across the
    cycles of [latency], the entry and the shift name one magnitude either way, thus era
    four and era five would read the same weight at their held reads. The gate below
    states that against the shared unit rather than arguing it.

    It stands here and not in [Mgen_nn] because a unit two shipped eras carry does not
    move for a round that has not shipped. WHETHER TO BACKPORT IS A DECISION FOR WHEN ERA
    SIX SETTLES, and the gate is the evidence for it.

    The rules of the table are the shared unit's, unmoved: [nn] is Q12, bits 11 to 4 read
    the table of 256 entries, bits 15 to 12 shift the entry right, and bits 21 to 16 force
    zero — a magnitude of 16 or above states nothing. The reference is
    [Mgen_nn.Quantized.exp2_of_magnitude], which holds the same truncation. *)

open Hardcaml

(** cycles from a magnitude at [nn] to its weight at [e]. Two — the address register and
    the entry register — and a new magnitude may stand at [nn] every cycle. *)
val latency : int

(** the width of the magnitude port, and the Q it reads at. A caller saturates into the
    width and shifts into the Q, thus the two stand here and a caller states neither
    number of its own. *)
val magnitude_bits : int

val input_q : int

module I : sig
  type 'a t =
    { clock : 'a
    ; nn : 'a
    (** the magnitude of the power, [magnitude_bits] wide at [input_q]: bits 11 to 0 are
        the fraction and bits 21 to 12 the integer part *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { e : 'a (** 2 ** -[nn], in Q15; 16 bits, unsigned *) }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
