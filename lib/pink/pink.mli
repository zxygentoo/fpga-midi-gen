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

    The registers of the voices of [default] are disjoint — no two voices can ever hold
    one pitch. A Note Off on the one MIDI channel releases a voice by pitch, thus a shared
    pitch lets one voice's off silence another.

    [Player] is the interface of the model: it makes the note events that the drivers
    send. The sequence is a pure function of the seed, thus the same seed gives the same
    piece in the reference, in the simulation and on the board. *)

open Base

module Params : sig
  type t =
    { rows : int (** the number of rows of the voice's group *)
    ; root : int
    (** the MIDI note of degree 0; its pitch class must be a degree of the scale *)
    ; degrees : int (** the number of scale degrees in the register *)
    ; stretch : int
    (** the mapping window. The sum of [rows] uniform bytes concentrates in the middle of
        its range, thus a map of the full range uses the outer degrees rarely. 1 maps the
        full range onto the degrees; [n] maps the centered [1/n] of the range and clips
        outside it. *)
    }
end

module Voice : sig
  type t =
    { params : Params.t
    ; restrike : bool
    (** re-strike at every due step, or only at a pitch change — the low voices of
        [default] speak only when they move *)
    }
end

(** The model: one scale, and the voices that live on it. A voice takes its offsets from
    the scale rotated to its own root, thus every voice holds the pitch classes of the one
    scale and a root outside the scale is an error. *)
type t =
  { scale : int list (** the semitone of each degree above the octave start *)
  ; voices : Voice.t list
  }

(** C major pentatonic, and four voices from the fastest rows to the slowest: soprano (A4
    to A6), alto (C4 to G4), tenor (C3 to A3) and bass (A1 to A2) — the partition 2+2+2+2,
    thus the periods 1, 4, 16 and 64 steps. *)
val default : t

(** [degree_offsets ~scale params] is the semitone offset above the root of each degree,
    from degree 0 upward, on the scale rotated to the root. The RTL elaboration reads this
    table, thus one definition serves the reference and the circuit. It raises
    [Invalid_argument] when the scale is empty, or when the root is not a degree of it. *)
val degree_offsets : scale:int list -> Params.t -> int list

(** [window params] is the low bound and the size of the mapping window, in that order.
    The RTL elaboration reads them, thus the reference and the circuit clamp the same sum
    to the same range. It raises [Invalid_argument] when the window is empty. *)
val window : Params.t -> int * int

(** [total_rows voices] is the number of rows that the voices share. It sets the size of
    the row set, thus the RTL elaboration reads it. *)
val total_rows : Voice.t list -> int

(** The state of one run: the row values, the PRNG and the step count. [Player] drives it,
    and the RTL tests compare [Source] against it. *)
type walk

(** The pitch and the due flag of one voice at one step. [due] is true at the steps where
    the voice's group re-rolled, and at step 1 for every voice — the piece begins with all
    of them. Between the due steps the pitch does not change. *)
type state =
  { note : int
  ; due : bool
  }

(** [create ~model ~seed] is the model at its origin. It raises [Invalid_argument] when
    the model has no scale or no voice, or a voice's parameters are out of range — rows,
    degrees and stretch must be at least 1, the stretch window must not be empty, the root
    must be a degree of the scale, and each degree must give a note in 0 to 127. The seed
    must fit 32 bits and must not be 0, the rule of the SEED cell. *)
val create : model:t -> seed:int -> walk

(** [next_step w] is the model after one step and the voice states, in the order of the
    voices of the model — the highest voice first, the strike order of the wire. *)
val next_step : walk -> walk * state list
