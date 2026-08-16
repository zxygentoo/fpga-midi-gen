(** The fixed evaluation protocol: the windows of a split, the masks of their codes, and
    the mean loss. This is the referee of the checkpoint election — the trainer, the
    checkpoint tool and the parity gates of the JAX seam all read the same rows, thus
    their numbers compare. *)

(** one evaluation row: [context + 1] codes — the inputs and the shifted labels — with the
    bar phases, the frames and the legality masks of the [context] input positions. The
    phase and the frame are the two halves of the rolling coordinate of the stream. *)
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

(** [masks_after codes] is the mask walk of a whole stream: element [i] is the legal set
    after code [i], thus it guards the draw of code [i + 1]. The walk opens at silence, so
    [codes] must open the stream. A window of a stream takes [row] instead: the masks of a
    whole stream do not fit in memory. *)
val masks_after : int array -> bool array array

(** [row stream ~start ~context] is the window of [stream] at [start]. It walks
    [Sounding_state] from the anchor at or below [start] to build the masks, thus its cost
    grows with the piece the window sits in and not with the stream. *)
val row : Jsb.stream -> start:int -> context:int -> row

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

(** [rows stream ~context ~limit] is the fixed windows of a split: the canonical stream —
    every piece at shift zero, in the order of the corpus — cut at stride [context] from
    its start. The same rows serve every evaluation point, thus the curve of one run and
    the curves of a sweep compare. [limit] bounds the work and not only the result. *)
val rows : Jsb.stream -> context:int -> limit:int -> row list

(** [loss config params rows ~batch] is the mean loss over the rows, weighted by rows. No
    gradient runs here. *)
val loss : Transformer.Config.t -> Transformer.Params.t -> row list -> batch:int -> float
