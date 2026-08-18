(** The JSB chorale corpus: the reader and the walk to step frames.

    The file is [Jsb16thSeparated.json], the Boulanger-Lewandowski split on the sixteenth
    grid with the voices apart: each step holds four cells indexed by voice, the soprano
    first, and a cell holds the pitch its voice sings, or -1 for a rest. The reader keeps
    that shape — the step frame of docs/transformer_model.md is the same four voices, and
    voice leading is the craft of a chorale. The dataset does not mark a repeated note: a
    pitch in two neighbour steps is one held note. A unison doubling is two voices on one
    pitch, and the decode of the sequencer sends one Note On for it. *)

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
  (** the step frame of docs/transformer_model.md: four voice codes in one word, seat 0 in
      the low byte. A voice code holds the MIDI pitch in bits 6:0 and the sounding flag in
      bit 7, thus a silent voice is [0x00] and a silent step is the word zero. Seat 0 is
      the bass and seat 3 is the soprano — the file gives the soprano first, thus the
      packer turns the order around.

      A frame is a wire word and not a class index. The vocabulary of the model is sized
      to the corpus and the two are different questions; the model maps one to the other. *)
  ; positions : int array
  (** the rolling coordinate of each step: the step count of the stream modulo
      [window_steps]. The bar phase is the low four bits and the frame is the high four,
      thus the two 16-row tables of the model read one number. *)
  }

(** the committed place of the corpus in this repository *)
val default_path : string

(** the voices of one step, which is the rows of the range table and the seats of a frame *)
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

(** [pack chorales] is the packed stream of docs/transformer_model.md: piece, seam, piece,
    seam. The caller states the order of the pieces and the transposition each one takes,
    thus one call is one stream and a split needs more than one to carry its
    transpositions.

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
