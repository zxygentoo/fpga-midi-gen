(** The step frame: the wire word of one step of music, and what it means.

    One step of music is one frame, and one frame is four voice codes in one 32-bit word.
    A voice code is one byte: bit 7 says the voice sounds and bits 6:0 hold the MIDI
    pitch. Seat 0 takes the low byte, seat 0 is the lowest voice and seat 3 is the
    highest. A cleared word is silence, thus a cleared context memory reads as silence and
    a seam costs no code.

    The pitch field holds all of 0 to 127. No code is reserved, thus a reader escapes no
    note and the MIDI range is the MIDI range. The codes [0x01] to [0x7F] have the flag
    clear: they are silence with a pitch that no writer sets, and they are 127 spare codes
    that this design does not use.

    **The wire is general and the model is not.** A frame states any MIDI pitch, and the
    vocabulary a model draws over is sized to the corpus it learns. The two are different
    questions: the class index lives with the corpus that sizes it — [Vocab] — and this
    module never sees one.

    The design is [docs/transformer_model.md]. *)

(** What a step does to the synthesizer: the messages the sequencer sends.

    An event is not a token. It names a pitch and whether the pitch starts or stops, and
    nothing else can happen — there is no code that opens a walk and none that closes a
    sentence. *)
module Event : sig
  type t =
    | On of int (** the MIDI note *)
    | Off of int (** the MIDI note *)
  [@@deriving sexp_of]
end

(** the seats of a frame: 4, because the synthesizer has four voices. This is a fact of
    the hardware and not a parameter of a model. *)
val voices : int

(** the bits of one voice code: 8. The frame is [code_bits * voices] wide on the socket,
    and a circuit that states a code sizes on this and not on a literal. *)
val code_bits : int

(** the frame of a step in which no voice sounds: the word zero *)
val silent : int

(** the voice code of a silent voice: [0x00] *)
val silent_code : int

(** [code_of_pitch pitch] is the voice code of [pitch], and [silent_code] for a negative
    pitch — the rest of a corpus cell. A pitch outside 0 to 127 raises [Invalid_argument]. *)
val code_of_pitch : int -> int

(** [pitch_of_code code] is the pitch a voice code sounds, or [None] when the flag is
    clear *)
val pitch_of_code : int -> int option

(** [of_codes codes] packs the voice codes into one frame, **seat 0 first**. A list that
    is not [voices] long raises [Invalid_argument].

    A corpus that names its voices in another order turns the order around before it calls
    this: the JSB file gives the soprano first, thus its bass reaches seat 0. *)
val of_codes : int list -> int

(** [codes frame] is the four voice codes of [frame], seat 0 first *)
val codes : int -> int list

(** [pitches frame] is the pitches the frame asks to sound, ascending and each one time. A
    unison is two seats on one pitch and one pitch on the wire, thus this is shorter than
    [codes] whenever two seats agree. *)
val pitches : int -> int list

(** [events_of_frames frames] is the walk a stream of frames states: the events of each
    step, released before struck.

    The rule is over sets and not over seats. The sequencer holds the set of pitches that
    sound; a frame states the set that must sound. The releases are the first set less the
    second, the strikes are the second less the first, and every release goes before every
    strike.

    A seat walk breaks on two cases of a chorale. Two voices that exchange pitches would
    send the Note On of a pitch before its Note Off, and the synthesizer would stop the
    new note, because the four voices share one MIDI channel and a Note Off releases a
    note by pitch. Two voices on one pitch would send two of each, and the second of each
    does the wrong thing.

    The rule gives three safety properties for nothing, and a legality mask is therefore
    not necessary anywhere: no strike names a pitch that sounds, no release names a pitch
    that does not, and four notes sound at the most — the releases only make the set
    smaller and the strikes then fill it to four or less. *)
val events_of_frames : int array -> Event.t list list
