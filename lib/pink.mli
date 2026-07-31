(** The pink-noise model: the reference implementation.

    The model makes a note sequence from 1/f noise with the Voss-McCartney algorithm, in
    integer arithmetic. The circuit implements the same arithmetic, therefore the same
    seed and the same parameters give the same note sequence here and in the RTL.

    The algorithm keeps [rows] byte values. At step [i] the model re-rolls rows 0 to
    [ctz i], in ascending order, with one PRNG draw for each row. Thus row [r] re-rolls
    every [2**r] steps: the fast rows give the local movement, and the slow rows give the
    long-term shape. The sum of the rows is the pitch signal, and the mapping quantizes it
    to a degree of the scale. *)

open Base

(** Marsaglia xorshift32. One step is three shift-and-XOR layers, thus the circuit can
    compute it in one cycle. The state walks all 32-bit values except 0. *)
module Prng : sig
  type t

  (** [create ~seed] raises [Invalid_argument] when [seed] does not fit 32 bits or is 0,
      the rule of the SEED cell. *)
  val create : seed:int -> t

  (** [next t] is the new state and one draw: the low 8 bits of the new state. *)
  val next : t -> t * int

  (** [state t] is the 32-bit state, for the comparison against the RTL. *)
  val state : t -> int
end

module Params : sig
  type t =
    { rows : int (** the number of rows; row [r] re-rolls every [2**r] steps *)
    ; root : int (** the MIDI note of degree 0 *)
    ; degrees : int (** the number of scale degrees in the melody range *)
    ; scale : int list (** the semitone above the octave start, for each scale step *)
    ; stretch : int
    (** the mapping window. The sum of [rows] uniform bytes concentrates in the middle of
        its range, thus a map of the full range uses the outer degrees rarely. 1 maps the
        full range onto the degrees; [n] maps the centered [1/n] of the range and clips
        outside it. *)
    }

  (** the C major pentatonic scale *)
  val pentatonic : int list

  (** 8 rows, root C4, 15 degrees of the pentatonic scale, stretch 2 *)
  val default : t
end

type t

(** [create params ~seed] is the model in its power-on state. It raises [Invalid_argument]
    when a parameter is out of its range: the seed rule is the one of [Prng.create],
    [scale] must not be empty, [rows], [degrees] and [stretch] must be at least 1, the
    stretch window must not be empty, and each degree must give a note in 0 to 127. *)
val create : Params.t -> seed:int -> t

(** [next t] is the model after one step and the MIDI note of that step. *)
val next : t -> t * int

(** [notes params ~seed] is the note sequence: pure, with no end, and equal for equal
    arguments. *)
val notes : Params.t -> seed:int -> int Sequence.t
