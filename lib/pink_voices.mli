(** The register decomposition: the rows of the pink model become voices.

    An experiment. The shipped model sums all eight rows into one pitch; this model splits
    them into three groups and maps each group's sum onto its own register:

    - the treble, rows 0 to 2: it re-articulates at every step
    - the middle, rows 3 to 5: at every step with [ctz] at least 3 — every 8th
    - the bass, rows 6 and 7: at every step with [ctz] at least 6 — every 64th

    A group re-articulates exactly when the walk re-rolls one of its rows, thus the
    note-rate hierarchy is the 1/f structure made audible: no rhythm generator exists. The
    draw stream is [Pink.Prng] with the ascending row order of the shipped model, and the
    sequence is a pure function of the seed.

    Three voices at one time fit the four-voice buffer of the S-1 with no steal. *)

open Base

(** The pitch and the due flag of each voice at one step, from the bass upward — the
    strike order of the wire. [due] is 1 at the steps where the voice's group re-rolled,
    and at step 1 for every voice — the piece begins with all of them. Between the due
    steps the pitch does not change. *)
type state =
  { note : int
  ; due : bool
  }

(** the number of voices — the length of every [next_step] state list *)
val voices : int

type t

(** [create ~seed] is the model at its origin; the seed rule is the one of
    [Pink.Prng.create]. *)
val create : seed:int -> t

(** [next_step t] is the model after one step and the three voice states, bass first. *)
val next_step : t -> t * state list
