(** The token of the transformer model of era three. Its design is the 2026-08-14 entry of
    [build-log.md]; [docs/transformer_model.md] holds era four, where a step is one frame
    and there is no token.

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

(** the size of the vocabulary: every code fits one byte *)
val vocab : int

(** the seats of the sequencer: at most four pitches sound *)
val seats : int

(** [to_code t] is the code of [t]. It raises [Invalid_argument] for a pitch outside its
    range: 0 to 126 for [On], and 1 to 127 for [Off]. *)
val to_code : t -> int

(** [of_code code] is the token of [code]: 0xFF is [Start], 0x80 to 0xFE is [On], 0x01 to
    0x7F is [Off], and 0 is [End]. A code outside the range 0 to 255 raises
    [Invalid_argument]. *)
val of_code : int -> t
