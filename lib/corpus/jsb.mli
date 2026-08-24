(** The JSB chorale corpus: the reader and the walk to step frames.

    The file is [Jsb16thSeparated.json], the Boulanger-Lewandowski split on the sixteenth
    grid with the voices apart: each step holds four cells indexed by voice, the soprano
    first, and a cell holds the pitch its voice sings, or -1 for a rest. The reader keeps
    that shape — the step frame of docs/transformer.md is the same four voices, and voice
    leading is the craft of a chorale. The dataset does not mark a repeated note: a pitch
    in two neighbour steps is one held note. A unison doubling is two voices on one pitch,
    and the decode of the sequencer sends one Note On for it. *)

type chorale =
  { cells : int list array
  (** the four cells of each sixteenth step, indexed by voice and the soprano first: the
      pitch that the voice sings, or -1 for a rest. This is the file as it stands; the
      reader escapes no pitch, because the frame reserves no code. *)
  ; legal_shifts : int list
  (** the legal transpositions, ascending: every shift that keeps each voice inside the
      observed range of its voice in the corpus. Zero is always a member. *)
  }

type t =
  { train : chorale list
  ; valid : chorale list
  ; test : chorale list
  }

(** one packed stream: piece, seam, piece, seam, with one element of [frames] and one of
    [positions] per step *)
type stream =
  { frames : int array
  (** the step frame of docs/transformer.md: four voice codes in one word, seat 0 in the
      low byte. A voice code holds the MIDI pitch in bits 6:0 and the sounding flag in bit
      7, thus a silent voice is [0x00] and a silent step is the word zero. Seat 0 is the
      bass and seat 3 is the soprano — the file gives the soprano first, thus the packer
      turns the order around.

      A frame is a wire word and not a class index. The vocabulary of the model is sized
      to the corpus and the two are different questions; the model maps one to the other. *)
  ; positions : int array
  (** the rolling coordinate of each step: the step count of the stream modulo
      [window_steps]. The bar phase is the low four bits and the frame is the high four,
      thus the two 16-row tables of the model read one number. *)
  }

(** the committed place of the corpus in this repository *)
val default_path : string

(** the voices of one step of this corpus: the rows of the range table. It equals
    [Frame.voices] because a four-part chorale fills the four seats of the synthesizer,
    and the two are different facts that happen to agree. *)
val voices : int

(** the steps of one bar on the sixteenth grid: the rows of the bar-phase table of the
    model, and the one clock the packed stream carries *)
val bar_steps : int

(** the period of the rolling coordinate, in steps: [bar_steps] bars, which is the memory
    window of the model *)
val window_steps : int

val load : path:string -> t

(** [transpose ~by chorale] moves each sounding cell [by] semitones and the legal shifts
    with it. A rest stays a rest. It raises when a pitch leaves 0 to 127 — the pitch field
    of a voice code bounds it — and a member of [legal_shifts] never raises. *)
val transpose : by:int -> chorale -> chorale

(** [on_grid ~every chorale] is the same piece on a coarser grid: every [every]-th step of
    the sixteenth grid, and nothing else. A grid of 1 is the identity, thus the sixteenth
    grid of the corpus — the grid of docs/coconet.md — costs the caller no special case.

    A grid of 2 is the eighth grid, which halves a canvas. It loses the onset that stands
    on an odd sixteenth: the even step before it holds its pitch through. That is 1.4
    percent of the onsets of this corpus, the ornamental passing tones.

    The legal shifts are computed again over the steps that remain. A pitch that only a
    dropped step sang no longer bounds the piece, thus a coarse-grid piece can hold a
    shift that the sixteenth-grid piece cannot.

    It raises [Invalid_argument] when [every] is 0 or less. *)
val on_grid : every:int -> chorale -> chorale

(** [pack chorales] is the packed stream of docs/transformer.md: piece, seam, piece, seam.
    The caller states the order of the pieces and the transposition each one takes, thus
    one call is one stream and a split needs more than one to carry its transpositions.

    Each step gives one frame, thus one step is one position of the model and the work of
    a step is constant. The walk opens with no code of its own: there is no start and
    there are no pieces to the model, only one walk.

    A seam is a count of silent frames, and the packer alone makes it. It needs no
    release: a piece ends with its four voices still sounding, and the decode of the
    sequencer reads the silent frame after it and sends the Note Offs. The count is the
    smallest that puts the downbeats of the next piece on the clock, and it is never zero,
    because that release needs one step. Every piece of this corpus is a whole number of
    quarter notes and every rotation is one, thus a seam is 4, 8, 12 or 16 steps: never
    shorter than a quarter note and never longer than a bar. The stream closes with a seam
    of its own, thus it leaves no chord sounding and it ends on a bar boundary. *)
val pack : chorale list -> stream

(** [streams chorales ~count ~random_state] is the streams of one split, [count] of them.

    The first is the canonical stream: every piece at shift zero, in the order given. A
    referee reads it alone, thus a measurement over it stays deterministic.

    Each of the others takes a uniform permutation of the pieces and a uniform draw from
    the legal shifts of each. One stream holds one draw, thus the count decides how many
    transpositions of a piece the trainer sees: a piece of this corpus has 7.4 legal
    shifts at the mean. *)
val streams : chorale list -> count:int -> random_state:Core.Random.State.t -> stream list

(** [windows stream ~context] cuts [stream] into the fixed windows of a referee: whole
    windows of [context + 1] steps, at stride [context], from the start. The tail that
    cannot fill a window is dropped, thus every window carries the same count of steps and
    a short one never reaches a mean.

    A window holds one step more than the context because its last frame is a label alone:
    [context] inputs state [context] labels. Therefore two windows in sequence share one
    step — the label of the first is the first input of the second.

    Give it the canonical stream, which [streams] gives first. A referee reads that stream
    alone, thus its measurement is deterministic and two referees that read one checkpoint
    must agree; the twin of this cut is in [jax/data.py], and the two state the same
    windows. A caller that wants fewer windows takes a prefix of the list.

    The training draw is not here. A trainer takes a uniform stream and then a uniform
    window of it, and the trainer of this project is on the JAX side.

    It raises [Invalid_argument] when [context] is 0 or less. *)
val windows : stream -> context:int -> stream list
