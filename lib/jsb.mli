(** The JSB chorale corpus: the reader and the walk to sentences.

    The file is [jsb-chorales-16th.json], the Boulanger-Lewandowski split on the sixteenth
    grid. One step is the list of the sounding pitches, the soprano first. The dataset
    does not mark a repeated note: a pitch in two neighbour steps is one held note. A
    unison doubling is one sounding pitch, as it is on the wire. *)

type chorale = int list array

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

(** [transpose ~by chorale] moves each pitch [by] semitones. It raises when a pitch leaves
    1 to 127. *)
val transpose : by:int -> chorale -> chorale

(** [pitch_range pool] is the lowest and the highest pitch that sound anywhere in [pool].
    It raises when no pitch sounds. *)
val pitch_range : chorale list -> int * int

(** [legal_shifts ~within chorale] is the transpositions, ascending, that keep every pitch
    of [chorale] inside [within] — the range-limited policy of the trainer and the corpus
    export. When [within] is the [pitch_range] of a pool that holds the chorale, zero is a
    member. A silent chorale takes the single shift zero. *)
val legal_shifts : within:int * int -> chorale -> int list

(** [encode ~lead_bars chorale] is the walk of the design document: two parallel arrays
    with one element per token.

    [codes] holds each token as its byte, per [Token.to_byte] — the stream the model
    trains on. Each step gives the OFF events ascending, then the ON events ascending,
    then [End].

    [phases] holds the bar phase (0 to bar − 1) of the step of each token, and zero is the
    downbeat. The bar length (16 or 12 steps) and the pickup come from the cadential
    holds; a piece with too few holds keeps the plain sixteen-step grid. The model reads
    the phase as the index into its bar-phase table.

    The walk begins with [lead_bars] bars of silence — bare [End] sentences — thus a model
    learns how a piece starts, and the cleared context of the sampler boots inside the
    training distribution. A trainer that varies [lead_bars] teaches an exit at every
    downbeat of silence, not at one count alone. *)
val encode : lead_bars:int -> chorale -> codes:int array * phases:int array
