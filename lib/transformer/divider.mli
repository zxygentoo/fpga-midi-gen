(** The restoring divider: the one division of the circuit, toward zero.

    The magnitude walks bit by bit — one quotient bit a cycle, the numerator's magnitude
    shifted into a remainder that the denominator subtracts when it fits — and the sign
    lands at the output. Therefore the quotient truncates toward zero at every sign, as
    the reference division does.

    The contract:

    - [start] loads the operands and begins the walk. It wins over a walk in flight: a
      [start] in a busy cycle discards that walk and begins a new one.
    - [busy] reads 1 in the cycle after [start], and it reads 0 again 40 cycles later —
      one cycle for each bit of the quotient. [quotient] is whole in the cycle [busy]
      reads 0, and it stands until the next [start]. Therefore the caller waits on [busy]
      and reads the result in the cycle the wait releases.
    - [numerator] is signed and [denominator] is unsigned.
    - The unit does not test the denominator against 0. A walk over 0 gives the largest
      magnitude the quotient holds, thus the caller must keep a zero denominator away.
    - The operands are read in the [start] cycle only, and the caller may move them after. *)

open Hardcaml

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
