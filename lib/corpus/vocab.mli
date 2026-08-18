(** The vocabulary of the corpus: the dense class window that the tables of a model read.

    A voice code is the wire byte of [Frame], and it states any MIDI pitch. A class index
    is a row of a table, and the rows exist only where the corpus sings. Class 0 is
    silence, class [1 + i] is the pitch [pitch_low + i], and the map between the two is
    one subtraction.

    **The corpus sets the two numbers, and not the model.** The corpus states the pitches
    36 to 81, and the transposition policy of [Jsb] holds each voice inside the observed
    range of that voice, thus 47 classes cover the music and the 48th is spare. The device
    ratified that count and did not choose it: four tables of 129 rows — one row for each
    MIDI pitch — put the six-layer model at 99 percent of the block RAM of the device,
    where it does not fit, and four tables of 48 rows put it at 93 percent.

    A model states how many tables read this window. The window is the same for all of
    them, thus a second model needs no vocabulary of its own.

    The numbers are stated here and derived from nothing. A checkpoint holds one row for
    each class, thus a window that moved with the corpus would silently make every trained
    model wrong. The expect test of [Jsb] holds the corpus inside the window, and a corpus
    with a wider range takes a wider window, wider tables and a new bitstream.

    The design is [docs/transformer_model.md]. *)

(** the rows of one table of a model: 48 *)
val classes : int

(** the class of a voice that does not sound *)
val silence : int

(** the pitch of class 1, thus class [1 + i] is the pitch [pitch_low + i] *)
val pitch_low : int

(** [class_of_code code] is the class index of a voice code of [Frame].

    A code with the flag clear is [silence], which is what the wire says: the codes [0x01]
    to [0x7F] are the spare codes of the frame and no writer makes one.

    A code that sounds a pitch outside the window raises [Invalid_argument]. No table
    holds a row for it, thus such a code is a fault of the corpus and not a case that a
    reader handles at run time. *)
val class_of_code : int -> int

(** [code_of_class index] is the voice code of a class index. An index outside 0 up to
    [classes - 1] raises [Invalid_argument].

    The spare class gives the pitch above the corpus, and the frame carries it like any
    other: a model that draws the spare states music that the corpus does not, and that is
    not an error. *)
val code_of_class : int -> int

(** [classes_of_frame frame] is the class of each seat, seat 0 first: the rows that the
    tables of a model read for one step *)
val classes_of_frame : int -> int list

(** [frame_of_classes indices] is the frame of one drawn step, and the inverse of
    [classes_of_frame]. A list that is not [Frame.voices] long raises [Invalid_argument]. *)
val frame_of_classes : int list -> int

(** The circuit's half of the map: the class the chain draws becomes the voice code the
    frame carries.

    It states one rule with the software above it, and the expect test of the module holds
    the two over every class of the vocabulary. The circuit needs no table for it — the
    silent class gives the silent code, and any other gives its pitch with the sounding
    flag set, which is one add and one bit. *)
module Rtl : sig
  (** [Make] over [Hardcaml.Bits] evaluates the map, and over [Hardcaml.Signal] it
      elaborates; the module below is the second one. *)
  module Make (Comb : Hardcaml.Comb.S) : sig
    (** [code_of_class index] is the voice code of a drawn class, [Frame.code_bits] wide *)
    val code_of_class : Comb.t -> Comb.t
  end

  include module type of Make (Hardcaml.Signal)
end
