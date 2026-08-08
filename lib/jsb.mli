(** The JSB chorale corpus: the reader and the walk to sentences.

    The file is [Jsb16thSeparated.json], the Boulanger-Lewandowski split on the sixteenth
    grid with the voices apart: each step holds four cells indexed by voice, the soprano
    first, and a cell holds the pitch its voice sings, or -1 for a rest. The reader
    derives the flat sounding sets of the walk, and the voices give the transposition
    policy. The dataset does not mark a repeated note: a pitch in two neighbour steps is
    one held note. A unison doubling is one sounding pitch, as it is on the wire. *)

type chorale =
  { steps : int list array
  (** the sounding set of each sixteenth step: ascending, unisons merged *)
  ; legal_shifts : int list
  (** the legal transpositions, ascending: every shift that keeps each voice inside the
      observed range of its voice in the corpus. Zero is always a member. *)
  }

type t =
  { train : chorale list
  ; valid : chorale list
  ; test : chorale list
  }

(** the committed place of the corpus in this repository *)
val default_path : string

(** the steps of one bar on the sixteenth grid *)
val bar_steps : int

val load : path:string -> t

(** [transpose ~by chorale] moves each pitch [by] semitones and the legal shifts with it.
    It raises when a pitch leaves 1 to 126 — the reserved codes bound the field. A member
    of [legal_shifts] never raises. *)
val transpose : by:int -> chorale -> chorale

(** [encode chorale] is the walk of the design document: two parallel arrays with one
    element per token.

    [codes] holds each token as its code, per [Token.to_code] — the stream the model
    trains on. The walk opens with [Start]; then each step gives the OFF events ascending,
    the ON events ascending, then [End].

    [phases] holds the bar phase (0 to bar − 1) of the step of each token, and zero is the
    downbeat. [Start] takes phase zero: the entry draw does not see a bar position. The
    bar length (16 or 12 steps) and the pickup come from the cadential holds; a piece with
    too few holds keeps the plain sixteen-step grid. The model reads the phase as the
    index into its bar-phase table. *)
val encode : chorale -> codes:int array * phases:int array
