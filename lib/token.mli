(** The token of the transformer model, per [docs/transformer_model.md].

    One token is one byte. Bit 7 is the type: 1 is Note On and 0 is Note Off. Bits 6:0 are
    the MIDI pitch. Two codes are reserved. The code 0xFF is START: it opens the walk of a
    piece and the model never draws it, thus [On 127] does not exist and the corpus reader
    moves pitch 127 down one semitone. The code 0x00 is END: it closes the sentence of a
    step, thus [Off 0] does not exist and the corpus reader moves pitch 0 up one semitone. *)

type t =
  | Start
  | On of int
  | Off of int
  | End
[@@deriving sexp_of]

(** the size of the vocabulary: one byte *)
val vocab : int

(** the seats of the sequencer: at most four pitches sound *)
val seats : int

(** [to_byte t] is the byte code of [t]. It raises for a pitch outside its range: 0 to 126
    for [On], 1 to 127 for [Off]. *)
val to_byte : t -> int

(** [of_byte b] decodes a byte: 0xFF is [Start], 0x80 to 0xFE is [On], 0x01 to 0x7F is
    [Off], and 0 is [End]. *)
val of_byte : int -> t
