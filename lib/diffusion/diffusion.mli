(** The masked canvas on the host: the float reference of era six.

    One canvas is T steps of four voices over the 48-class vocabulary of [Vocab], beside a
    mask over its cells. The model reads the canvas with the masked cells hidden and
    states a distribution over the classes of every cell at once; the Gibbs walk hides and
    restores cells in turn until a piece stands. Nothing is causal and nothing is written
    left to right. The design of the model is [docs/diffusion.md], and the design of this
    layer and its gates is [docs/diffusion_rtl.md].

    This module is the reference and not the trainer. The trainer is
    [jax/diffusion/train.py], and what stands here is what the gates of the project need:
    the same loss on the same canvases (Gate A), the same walk from the same seed (Gate
    C), and the float model that the integer twin of [Quantized] must follow.

    Every pass here is an inference pass: the batch norm reads the population statistics
    of the checkpoint and never a batch. The reference has no training mode, thus a canvas
    answers alone and a referee reproduces a number.

    Every draw of the walk comes from [Prng], the xorshift32 of the circuit, and THE
    CONSUMPTION ORDER IS THE CONTRACT of [docs/diffusion_rtl.md]: the opening draws one
    uniform for each cell, each pass draws one for each cell (the masks) and then one for
    each hidden cell (the classes), and the cells walk in step-major, seat-minor order
    everywhere. Every cell of a canvas is free — nothing is given to a walk of this era.
    One seed names one canvas — in JAX, here, in the twin and on the board — and the same
    seed gives the same piece. *)

(** every tensor of the float model is float32 *)
type tensor = (float, Nx.float32_elt) Nx.t

(** the rows of the roll: the classes of [Vocab], row 0 silence *)
val rows : int

(** the seats of a step: the voices of [Frame] *)
val voices : int

(** the reach of one convolution: three by three over time and pitch *)
val kernel : int

(** the batch_norm_variance_epsilon of the trainer *)
val norm_epsilon : float

(** The register of each seat: the lowest and the highest pitch it sings anywhere in this
    corpus, seat 0 the bass — [Jsb.voice_ranges] turned around, thus the corpus library's
    own test pins it. The seeded opening of the walk draws inside these, thus a cell the
    first Bernoulli leaves standing states a note a chorale could hold. They are the
    RANGES of [jax/measure.py], stated as pitches. *)
val seat_ranges : (int * int) array

(** [classes_of_chorale chorale] is the canvas of one piece of [Jsb]: [steps] rows of
    [voices] class indices, seat 0 the bass — the file gives the soprano first, thus the
    step turns around, as every reader of the corpus turns it. A rest is the silence
    class. A pitch outside the vocabulary raises [Invalid_argument], because no table
    holds a row for it. *)
val classes_of_chorale : Jsb.chorale -> int array array

(** The shape of the model: the two numbers that size every tensor. The reference takes
    any shape a checkpoint states; the elected shape of the era is L 48 by H 20. *)
module Config : sig
  type t =
    { layers : int (** L: the stem, the residual pairs, and the head *)
    ; width : int (** H, the channels of the trunk *)
    }

  (** [of_checkpoint path] reads the shape from the tensor shapes alone: five tensors make
      a layer, thus the count states L, and the stem kernel states H. A player states no
      flag, and two sides cannot drift apart in one.

      It raises [Invalid_argument] when the file is not a canvas model — the tensor count
      does not divide by five, fewer than three layers stand, or a kernel shape does not
      agree with its place — and the message names the tensor that refused. *)
  val of_checkpoint : string -> t
end

(** The weights of the float model: for each layer the kernel, the two norm terms, and the
    two population statistics. The statistics are not parameters — no gradient ever
    reached them — but the model cannot state a probability without them, thus they travel
    inside the checkpoint and inside this structure. *)
module Params : sig
  type t

  (** the shapes of the tensors in the flat order of the checkpoint, from the
      configuration: the quantization of [Quantized] reads the same table, thus the two
      models cannot disagree about what a checkpoint holds *)
  val shapes : Config.t -> int array list

  (** [init config ~seed] draws initial parameters through [Prng], thus a test needs no
      checkpoint file and the same seed gives the same weights. The draw follows the SHAPE
      of [jax/diffusion/train.py] and not its values: He normal kernels, the norm scale at
      the trainer's tenth, the shift at zero, and the population at mean 0 and variance 1.
      The two generators differ, thus only a trained checkpoint crosses the seam; what
      must agree is the distribution, because a drift report over weights of the wrong
      scale would measure an arithmetic this era does not run.

      [norm_scale] overrides the trainer's opening tenth, and the drift gates take 1.0: at
      the tenth an L-layer DRAWN trunk decays its activations tenfold at every layer — a
      trained norm grows out of that opening, an untrained one never leaves it — and by
      the third layer a drift report reads the resolution floor of the format and not the
      arithmetic. At 1.0 the drawn trunk holds the O(1) activations a trained model holds,
      which is the regime the twin must answer for. *)
  val init : ?norm_scale:float -> Config.t -> seed:int -> t

  (** the tensors in the flat order of the checkpoint: for each layer the kernel, the
      scale, the shift, the mean and the variance *)
  val to_list : t -> tensor list

  (** [load config ~path] is the parameters of the checkpoint at [path] — the safetensors
      file the JAX trainer writes, tensors named "0" upward in the order of [to_list]. A
      shape that does not agree with [shapes config] raises [Invalid_argument]. *)
  val load : Config.t -> path:string -> t
end

(** [logits params ~classes ~hidden] is one forward pass over one canvas: the raw logits
    of every cell, as a [steps; rows; voices] tensor. [classes] is the canvas — [steps]
    rows of [voices] class indices — and [hidden] the mask over its cells, true where a
    cell is hidden. A hidden cell shows zero in the roll and one in its mask plane; the
    softmax runs over the rows axis, and the model states a class for every cell, hidden
    or not. The parameters carry the whole shape, thus no configuration enters.

    The arithmetic is the trainer's, expression for expression — the zero-padded
    convolution, the norm in the trainer's order, the residual pairs, the head with its
    norm and no activation — because Gate A holds the two forwards to a tolerance and
    every rearrangement spends some of it. *)
val logits : Params.t -> classes:int array array -> hidden:bool array array -> tensor

(** [masked_nll params ~classes ~hidden] is the orderless-NADE loss of one canvas: the
    negative log-likelihood of the hidden cells under the softmax over the rows, in nats
    for each hidden cell. It is the number of Gate A, and it does not compare with the
    paper's Table 1 — that referee is Algorithm 1, and it lives in JAX.

    It raises [Invalid_argument] when no cell is hidden. *)
val masked_nll : Params.t -> classes:int array array -> hidden:bool array array -> float

(** [anneal_threshold ~step ~walk] is the masking threshold of pass [step] of [walk], on
    the 24-bit grid of the generator: [floor (alpha * 2^24)] of the paper's annealed
    probability [alpha = max (0.1, 0.9 - 0.8 step / (0.7 walk))]. A cell hides exactly
    when its uniform times 2^24 falls under this integer. The float sides and the twin
    compare against the same number, and it is the entry the circuit's table will hold. *)
val anneal_threshold : step:int -> walk:int -> float

(** [opening_canvas state ~steps] is the seeded opening of the walk: one uniform for each
    cell in the cell order, the class [low + floor (u * width)] inside the register of the
    cell's own seat. [gibbs] opens here and the engine of [Quantized] opens here, thus the
    two openings are one rule and one consumption. *)
val opening_canvas : Prng.state -> steps:int -> Prng.state * int array array

(** [hidden_cells state ~steps ~threshold] is the mask of one pass: one uniform for each
    cell in the cell order, hidden exactly when [u * 2^24] falls under [threshold]. The
    same sharing argument as [opening_canvas]. *)
val hidden_cells
  :  Prng.state
  -> steps:int
  -> threshold:float
  -> Prng.state * bool array array

(** [gibbs params ~steps ~walk ~temperature ~seed] draws one canvas: [steps] steps of four
    voices, by [walk] passes of independent blocked Gibbs under the annealed schedule.

    The walk opens on notes — each cell a uniform draw inside the register of its own seat
    — thus every step of the walk is the same step: masks, one forward pass, redraws. The
    draws go through [Policy.draw_class] with no min-p floor, thus this walk and the walks
    of the other eras are comparable pick for pick.

    The seed folds ([Prng.create_folded]), thus any integer names a walk; the integer twin
    and the board take the seed as it stands, under the rule of the SEED cell. The same
    seed gives the same music here and in [jax/diffusion/infer.py], which is Gate C.

    It raises [Invalid_argument] when [temperature] is 0 or less, through
    [Policy.check_policy]. *)
val gibbs
  :  Params.t
  -> steps:int
  -> walk:int
  -> temperature:float
  -> seed:int
  -> int array array

(** [frames_of_canvas canvas] is the frame of each step: the classes of a step become the
    voice codes of one word, seat 0 in the low byte. [Frame.events_of_frames] then states
    the events of the wire, and the step lines of the players print them — the decode is
    the rule of the frame, and it is the same rule [jax/data.py] states, thus a walk here
    and a walk there compare as text. *)
val frames_of_canvas : int array array -> int array

(** [gate_canvases chorales ~steps] is the deterministic set of Gate A: the first [steps]
    steps of every chorale that holds them, in the order given. No draw enters: the same
    corpus states the same canvases on both sides of the seam. [check_diffusion loss]
    states the number over these, and [jax/tests/test_parity.py] demands it of the JAX
    forward. *)
val gate_canvases : Jsb.chorale list -> steps:int -> int array array array

(** [gate_mask ~index ~steps] is the mask of gate canvas [index]: a Bernoulli half from
    [Prng.create ~seed:(index + 1)], one uniform for each cell in the pinned order, hidden
    exactly when [u * 2^24 < 2^23]. The JAX side draws the same masks from the batched
    twin of the generator. *)
val gate_mask : index:int -> steps:int -> bool array array

(** The checkpoint seam, for the gates that cross it. *)
module For_test : sig
  (** [with_checkpoint tensors ~f]: [Checkpoint.with_checkpoint], the one writer of the
      file the JAX trainer writes *)
  val with_checkpoint : tensor list -> f:(string -> 'a) -> 'a

  (** [refusal ~path f] is the message of the [Invalid_argument] that [f] raises, with
      [path] scrubbed, and ["no raise"] when [f] raises nothing *)
  val refusal : path:string -> (unit -> unit) -> string
end
