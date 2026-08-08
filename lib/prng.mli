(** The PRNG: Marsaglia xorshift32, in software and as a circuit.

    One step is three shift-and-XOR layers. The software gives the reference, the circuit
    computes the same recurrence combinationally, and the vector test in this module
    drives the two side by side. Thus the definition of the walk has one home.

    The state walks all 32-bit values except 0.

    Every draw is pure: it takes a state and gives the next state beside the value. The
    caller carries that new state forward, because a second call on the same state repeats
    the same draw. Therefore one seed names one sequence, in the software, in the
    simulation and on the board.

    A draw comes out as a plain float or float array. This module owns the randomness
    alone; a caller that wants a tensor gives the array a shape with [Nx.create]. *)

open Hardcaml

(** the state of the walk: 32 bits, and never 0 *)
type t

(** [create ~seed] is the walk that starts at [seed]. It raises [Invalid_argument] when
    [seed] is 0 or does not fit 32 bits — the rule of the SEED cell of the board. *)
val create : seed:int -> t

(** [create_folded ~seed] is the walk that [seed] names, for any integer. The fold
    squeezes the seed into 32 bits, and 0 — no state of the walk — takes the top state. A
    seed already inside the range names itself, thus [create_folded ~seed:7] is the walk
    of the board's seed 7. Two seeds can fold together; this is harmless, because a seed
    only names a walk.

    Take [create] where the seed must obey the rule of the SEED cell, and this where the
    seed comes from a flag or from a stream that obeys no such rule. *)
val create_folded : seed:int -> t

(** [next t] is the state after one step, and the draw of that step: the low 8 bits of the
    new state, 0 to 255. This is the draw the circuit gives, thus a walk of [next] and a
    walk of [Rtl] agree byte for byte. *)
val next : t -> t * int

(** [uniform t] is the state after three steps, and one draw of the uniform distribution
    over \[0, 1). The three bytes make a grid of 2 ** -24, which is fine enough to hold
    the tail of a Box-Muller draw that one byte would cut at 3.3 sigma.

    The grid holds 0. Therefore a caller that takes the logarithm of a draw must hold that
    case. *)
val uniform : t -> t * float

(** [uniforms t ~count] is the state after [count] draws of [uniform], and the draws in
    the order of the walk. A count of 0 gives the state and an empty array. *)
val uniforms : t -> count:int -> t * float array

(** [normals t ~count ~scale] is the state after the draws, and [count] draws of the
    normal distribution whose mean is 0 and whose standard deviation is [scale].
    Box-Muller makes them, thus one draw takes two uniforms and six steps of the walk. *)
val normals : t -> count:int -> scale:float -> t * float array

(** [bernoullis t ~count ~probability] is the state after the draws, and [count] draws of
    the Bernoulli distribution: 1.0 at [probability], and 0.0 else. One draw takes one
    uniform. The hit weighs 1.0; a caller that wants another weight multiplies the array. *)
val bernoullis : t -> count:int -> probability:float -> t * float array

(** The circuit: the same recurrence, one step in one cycle. *)
module Rtl : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a (** the state takes 1 *)
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

  (** [create i] is the block. The state register takes the new value at a [step] strobe
      and holds between the strobes. The clear puts 1 into the state: the state has no use
      before the first [load], and 1 keeps the no-zero rule. [load] wins over [step] in
      the same cycle. *)
  val create : Signal.t I.t -> Signal.t O.t
end
