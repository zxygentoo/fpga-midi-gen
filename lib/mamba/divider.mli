(** The restoring divider: the one division of the circuit, toward zero.

    The magnitude walks bit by bit — one quotient bit a cycle, the numerator's magnitude
    shifted into a remainder that the denominator subtracts when it fits — and the sign
    lands at the output. Therefore the quotient truncates toward zero at every sign, as
    the reference division does.

    This unit is era five's own, and it differs from era four's in one design choice: the
    start cycle latches the raw signed numerator, and the FIRST BUSY CYCLE takes the
    magnitude, register to register. Era four's unit negates in the start cycle, which
    puts a 40-bit carry chain between the caller's operand mux and the first register.
    That closed with one writer of the numerator; era five has two — the norm and the
    attention head — and the program counter's mux in front of that carry chain is the
    critical path of the whole design. The magnitude stage cuts it, and costs one cycle
    for each divide. The two units unify when the common parts of the sources are
    extracted; until that round, this copy stands.

    The contract:

    - [start] loads the operands and begins the walk. It wins over a walk in flight: a
      [start] in a busy cycle discards that walk and begins a new one.
    - [busy] reads 1 in the cycle after [start], and it reads 0 again [busy_cycles] later
      — one cycle for the magnitude, then one for each bit of the quotient. [quotient] is
      whole in the cycle [busy] reads 0, and it stands until the next [start]. Therefore
      the caller waits on [busy] and reads the result in the cycle the wait releases.
    - [numerator] is signed and [denominator] is unsigned.
    - The unit does not test the denominator against 0. A walk over 0 gives the largest
      magnitude the quotient holds, thus the caller must keep a zero denominator away.
    - The operands are read in the [start] cycle only, and the caller may move them after. *)

open Hardcaml

(** the length of one walk, in cycles: the caller's cost model reads it here rather than
    restate it *)
val busy_cycles : int

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a (** a strobe: load the operands and begin the walk *)
    ; numerator : 'a (** 40 bits, signed *)
    ; denominator : 'a (** 24 bits, unsigned *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { quotient : 'a (** 40 bits, signed; whole in the cycle [busy] reads 0 *)
    ; busy : 'a (** a walk is in flight *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
