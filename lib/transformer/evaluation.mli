(** The fixed evaluation protocol: the windows of a split, the masks of their codes, and
    the mean loss. This is the referee of the checkpoint election — the trainer, the
    checkpoint tool and the parity gates of the JAX seam all read the same rows, thus
    their numbers compare. *)

(** one evaluation row: [context + 1] codes — the inputs and the shifted labels — with the
    bar phases, the piece positions and the legality masks of the [context] input
    positions *)
type row =
  { codes : int array
  ; phases : int array
  ; progress : int array
  ; masks : bool array array
  }

(** the rows of one batch, each field stacked over the rows *)
type batch =
  { codes : int array array
  ; phases : int array array
  ; progress : int array array
  ; masks : bool array array array
  }

(** [masks_after codes] is the mask walk of one encoded piece: element [i] is the legal
    set after code [i], thus it guards the draw of code [i + 1]. *)
val masks_after : int array -> bool array array

(** the int32 words of one packed mask, and thus the row width of the [masks] tensor of a
    seam file. A writer sizes and strides its tensor with this and states no count of its
    own. *)
val words_per_mask : int

(** [mask_words mask] is the wire form of one mask in the seam files: [words_per_mask]
    int32 words, bit [k] of word [j] flags code [32 * j + k]. The bytes are little endian
    on the wire, thus the Python side views the words as bytes and unpacks with bitorder
    little. *)
val mask_words : bool array -> int array

(** [batch_of_rows rows] stacks the rows into the parallel batch arrays. *)
val batch_of_rows : row list -> batch

(** [rows chorales ~context ~limit] is the fixed windows of a split: each chorale encodes
    with no transposition, and its windows cut at stride [context]. The same rows serve
    every evaluation point, thus the curve of one run and the curves of a sweep compare. A
    chorale below one window is skipped. *)
val rows : Jsb.chorale list -> context:int -> limit:int -> row list

(** [loss config params rows ~batch] is the mean loss over the rows, weighted by rows. No
    gradient runs here. *)
val loss : Transformer.Config.t -> Transformer.Params.t -> row list -> batch:int -> float
