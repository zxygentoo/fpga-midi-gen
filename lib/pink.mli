(** The pink model: the reference implementation.

    The model makes music from 1/f noise with the Voss-McCartney algorithm, in integer
    arithmetic. The row values re-roll on the binary schedule: at step [i], rows 0 to
    [ctz i], in ascending order, with one PRNG draw for each row. Thus row [r] re-rolls
    every [2**r] steps: the fast rows give the local movement, and the slow rows give the
    long-term shape.

    The rows split into voices — the register decomposition. Each voice takes a group of
    rows and maps the group's sum onto its own register, and a voice re-articulates
    exactly when the walk re-rolls one of its rows. Thus the partition is the rhythm: a
    group that starts at row [r] re-articulates every [2**r] steps, and the note-rate
    hierarchy is the 1/f structure made audible, with no rhythm generator.

    The registers of [default_voices] are disjoint — no two voices can ever hold one
    pitch. A Note Off on the one MIDI channel releases a voice by pitch, thus a shared
    pitch lets one voice's off silence another.

    The one-voice model — every row in one group — is the model of the shipped circuit,
    and [notes] gives it: the reference of the RTL stream comparisons. The sequence is a
    pure function of the seed in every configuration. *)

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
    { rows : int (** the number of rows of the voice's group *)
    ; root : int (** the MIDI note of degree 0 *)
    ; degrees : int (** the number of scale degrees in the register *)
    ; scale : int list (** the semitone above the octave start, for each scale step *)
    ; stretch : int
    (** the mapping window. The sum of [rows] uniform bytes concentrates in the middle of
        its range, thus a map of the full range uses the outer degrees rarely. 1 maps the
        full range onto the degrees; [n] maps the centered [1/n] of the range and clips
        outside it. *)
    }

  (** the C major pentatonic scale *)
  val pentatonic : int list

  (** the rotation of the pentatonic that starts on A: the same pitch classes, for an
      A-rooted register *)
  val a_pentatonic : int list

  (** 8 rows, root C4, 15 degrees of the pentatonic scale, stretch 2: the one-voice model
      of the shipped circuit *)
  val default : t
end

(** [degree_offsets params] is the semitone offset above the root of each degree, from
    degree 0 upward. The RTL elaboration reads this table, thus one definition serves the
    reference and the circuit. It raises [Invalid_argument] when the scale is empty. *)
val degree_offsets : Params.t -> int list

(** [reroll prng values ~count] re-rolls the first [count] values, with one draw for each,
    in ascending order — the walk of the model. The circuit does the same walk, thus the
    draw order is part of the contract. *)
val reroll : Prng.t -> int list -> count:int -> Prng.t * int list

(** [mapper params] is the map from a row sum to a MIDI note: the centered [1/stretch]
    window of the sum range, quantized to the degrees, on the scale from the root. The
    degree table is computed one time, at the partial application. It raises
    [Invalid_argument] when the window is empty. *)
val mapper : Params.t -> int -> int

module Voice : sig
  type t =
    { params : Params.t
    ; restrike : bool
    (** re-strike at every due step, or only at a pitch change — the low voices of
        [default_voices] speak only when they move *)
    }
end

(** The four voices, from the fastest rows to the slowest: soprano (A4 to A6), alto (C4 to
    G4), tenor (C3 to A3) and bass (A1 to A2) — the partition 2+2+2+2, thus the periods 1,
    4, 16 and 64 steps. The A-rooted registers take [Params.a_pentatonic], thus every
    voice stays on the pitch classes of C major pentatonic. *)
val default_voices : Voice.t list

(** The pitch and the due flag of one voice at one step. [due] is 1 at the steps where the
    voice's group re-rolled, and at step 1 for every voice — the piece begins with all of
    them. Between the due steps the pitch does not change. *)
type state =
  { note : int
  ; due : bool
  }

type t

(** [create ~voices ~seed] is the model at its origin. It raises [Invalid_argument] when
    [voices] is empty or a voice's parameters are out of range — the scale must not be
    empty, rows, degrees and stretch must be at least 1, the stretch window must not be
    empty, and each degree must give a note in 0 to 127. The seed rule is the one of
    [Prng.create]. *)
val create : voices:Voice.t list -> seed:int -> t

(** [next_step t] is the model after one step and the voice states, from the slowest voice
    upward — the strike order of the wire. *)
val next_step : t -> t * state list

(** [notes params ~seed] is the note sequence of the one-voice model: pure, with no end,
    and equal for equal arguments. This is the reference of the shipped circuit. *)
val notes : Params.t -> seed:int -> int Sequence.t
