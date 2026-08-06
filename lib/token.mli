(** The token of the transformer model, per [docs/transformer_model.md].

    One token is one byte. Bit 7 is the type: 1 is Note On and 0 is Note Off. Bits 6:0 are
    the MIDI pitch. The code 0x00 is END: it closes the sentence of a step. Therefore
    [Off 0] does not exist, and the corpus reader moves pitch 0 up one semitone. *)

type t =
  | End
  | Off of int
  | On of int
[@@deriving sexp_of]

(** the size of the vocabulary: one byte *)
val vocab : int

(** the seats of the sequencer: at most four pitches sound *)
val seats : int

(** [to_byte t] is the byte code of [t]. It raises for a pitch outside its range: 1 to 127
    for [Off], 0 to 127 for [On]. *)
val to_byte : t -> int

(** [of_byte b] decodes a byte: 0 is [End], 0x01 to 0x7F is [Off], the rest is [On]. *)
val of_byte : int -> t
