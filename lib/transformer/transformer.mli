(** The transformer model on the host: the network, the loss and the sampler.

    The network follows [docs/transformer_model.md]: decoder-only, no bias terms, RMSNorm
    with no scale, ALiBi for the position, a bar-phase table, and the embedding table tied
    with the output head. The host model is float32; the integer ladder comes after the
    audition. Raven carries the work: Nx computes, Rune differentiates, and Kaun gives
    AdamW, the loss and the checkpoint format.

    Every draw — the parameters, the dropout masks and the sampler — comes from [Prng],
    the xorshift32 of the circuit. Thus one seed names one walk in the software, in the
    simulation and on the board. A seed here is any integer: it folds into the 32 bits of
    the state, and a seed already inside that range names itself. *)

(** every tensor of the host model is float32 *)
type tensor = (float, Nx.float32_elt) Nx.t

(** the rows of the bar-phase table, the rows of the piece-position table, and the steps
    of one draw bucket; [docs/transformer_model.md] holds the reasoning *)
val phase_buckets : int

val progress_buckets : int
val progress_stride : int

module Config : sig
  type t =
    { d : int (** the width of the residual stream *)
    ; layers : int
    ; heads : int (** they split the width at run time, thus [d] divides by them *)
    ; context : int (** the attention window, in tokens *)
    ; slope_span : int
    (** The ALiBi exponent span: the slope of head k is 2 ** -(span (k+1) / heads), and a
        head subtracts its slope times the distance from its logits. The span sets how far
        the gentlest head sees: the paper's 8 leaves it at -4 logits by 1024 tokens and -8
        by 2048, blind to a phrase. A wider span reaches further and stays a power of two,
        thus a shift in the circuit. *)
    }

  (** the baseline of the design document: d 64, layers 2, heads 4, context 256 *)
  val baseline : t

  (** [of_checkpoint path ~heads ~context ~slope_span] reads the width and the layer count
      from the tensor shapes of the checkpoint, thus a player states neither: the file
      holds three tables and six tensors for each layer. A checkpoint from before the
      piece-position table holds two tables and does not load; `_train/archive` keeps a
      player that reads those. The heads and the context are not in the file: no tensor
      shape holds them, because the heads only split the width at run time and ALiBi holds
      no position table. The context is a choice of the draw in any case — a model trained
      long can sample short. *)
  val of_checkpoint : string -> heads:int -> context:int -> slope_span:int -> t
end

(** The parameter structure of the model, over any tensor type: the three tables — the
    tied embedding, the bar phase and the piece position — and six tensors for each layer.
    One definition holds the structure and the flat order of the checkpoint: [Params]
    instantiates it with the float tensor and keeps its own type opaque, and [Fixed.Model]
    instantiates it with the quantized tensor. The module holds the shape alone — the
    tensor shapes, the draw and the checkpoint live with the models. *)
module Params_data : sig
  type 'a t =
    { embed : 'a (** the tied token table; the head reads it backward *)
    ; phase : 'a (** the bar-phase table *)
    ; progress : 'a (** the piece-position table *)
    ; layers : 'a layer array
    }

  and 'a layer =
    { wq : 'a
    ; wk : 'a
    ; wv : 'a
    ; wo : 'a
    ; w1 : 'a
    ; w2 : 'a
    }

  (** the flat order of the tensors — the order of the checkpoint; [of_list] reads the
      same order *)
  val to_list : 'a t -> 'a list

  (** [of_list ~layers items] reads the order of [to_list]. It raises [Invalid_argument]
      when the count of items is not three tables and six for each of [layers]. *)
  val of_list : layers:int -> 'a list -> 'a t
end

module Params : sig
  (** the weights of the float model: [Params_data] over [tensor] *)
  type t

  (** the shapes of the tensors in the flat order, from the configuration; the
      quantization of [Fixed] reads the same table *)
  val shapes : Config.t -> int array list

  (** [init config ~seed] draws the initial parameters: normal, scale 0.02. [Prng] and
      Box-Muller make the draw, thus the seed is an input and the same seed gives the same
      start. The tensors come in the order of [to_list]. *)
  val init : Config.t -> seed:int -> t

  (** [save t ~path] writes the checkpoint: the tensors in the flat order, and nothing
      else. The width and the layer count come back from the shapes with
      [Config.of_checkpoint], but the heads, the context and the ALiBi span do not — no
      tensor shape holds them, thus [load] takes the config of the training run. *)
  val save : t -> path:string -> unit

  (** [load config ~path] is the parameters of the checkpoint at [path]. [config] states
      the shapes, and a shape of the file that does not agree with them raises
      [Invalid_argument]. *)
  val load : Config.t -> path:string -> t

  (** the flat order of the tensors; [of_list] reads the same order *)
  val to_list : t -> tensor list

  (** [of_list config tensors] reads the order of [to_list]. It raises [Invalid_argument]
      when the count of tensors does not fit [config]. *)
  val of_list : Config.t -> tensor list -> t

  (** The flat order as a [Ptree] list, for an optimizer that walks a tree. The checkpoint
      goes through [save] and [load]; take these only to reach a library that wants the
      tree itself. *)
  val to_ptree : t -> Kaun.Ptree.t

  val of_ptree : Config.t -> Kaun.Ptree.t -> t
end

(** The instrumentation of one sampling run: how often the mask fires. [refused] is the
    mean legal probability mass that the min-p filter removed each draw. [illegal_mass] is
    the mean raw mass the model put outside the legal set each draw — the rate at which a
    sampler with no mask would emit an illegal token. [illegal_top] is the share of draws
    whose raw argmax was itself illegal, and [draws] is how many draws the run took — the
    count the three means divide by. *)
type sample_stats =
  { refused : float
  ; illegal_mass : float
  ; illegal_top : float
  ; draws : int
  }

(** The dropout of one training step. The JAX sweep of 2026-08-07 found the rate scales
    with the model: 0.1 at d 64, 0.2 at d 128 and at the long context. The blocks that
    drop are the embedding sum and the two branches of each layer, and each of them draws
    from a walk of its own, thus the step is a function of the seed alone and the same
    seed gives the same step. *)
module Dropout : sig
  type t

  (** the identity: every mask is one, and the forward pass is the inference pass *)
  val none : t

  (** [create ~rate ~seed] drops [rate] of each block, and scales the survivors by 1 /
      (1 - rate) so that the inference pass rescales nothing. The mask takes the shape of
      the tensor it drops, thus the batch and the context are not inputs here. A rate of
      zero or less is [none]; a rate of 1 or more raises, because the scale of the
      survivors divides by zero. *)
  val create : rate:float -> seed:int -> t
end

(** [logits config params ~codes ~phases ~progress ~dropout] is the raw next-code logits
    of each input position, shape \[batch; length; vocab\] — the forward pass that the
    loss and the sampler share. The calibration of the integer twin compares against it. *)
val logits
  :  Config.t
  -> Params.t
  -> codes:int array array
  -> phases:int array array
  -> progress:int array array
  -> dropout:Dropout.t
  -> tensor

(** [loss config params ~codes ~phases ~progress ~masks ~weights ~dropout] is the cross
    entropy of the next code, a scalar. [progress] holds the piece-position bucket of each
    input position. A row of [codes] holds [length + 1] codes: the inputs and the shifted
    labels. [masks] holds the legal set of each label, from the walk of the whole chorale,
    and it sits inside the softmax: the model spends no mass on a code that the sampler
    would refuse. Therefore its raw mass outside the legal set stays untrained, and
    [sample] must carry the same mask at every draw. [weights] holds one weight for each
    label position; zero drops the position from the mean, which is how the padding of a
    short piece stays out of the loss — a padded label would teach the walk to hold the
    last chord and emit END for ever. *)
val loss
  :  Config.t
  -> Params.t
  -> codes:int array array
  -> phases:int array array
  -> progress:int array array
  -> masks:bool array array array
  -> weights:float array array
  -> dropout:Dropout.t
  -> tensor

(** [sample config params ~seed ~steps ~temperature ~min_p] draws [steps] steps from the
    boot: an empty context, then [Start] at phase zero — power on, music on. The
    piece-position bucket of step [i] is [i / 16 mod 16]: sixteen bars to the arc, about
    the length of a chorale, and the arc repeats for as long as the draw runs. Therefore a
    short draw is a prefix of a long one, and a draw of any length stays inside the walk
    the corpus taught — where dividing by the length of the draw would stretch one arc
    over a span no piece ever had, and would ask the board, which plays for ever, for a
    length it does not know. One element of [music] is one drawn step: the events of its
    sentence, without the [End]. The mask of the design document guards every draw, thus
    each sentence is valid. [min_p] removes each legal token whose tempered probability is
    below [min_p] of the peak's; the peak always stays, thus a draw always exists, and
    zero turns the filter off. The same seed gives the same music.

    It raises [Invalid_argument] when [temperature] is 0 or less, or when [min_p] falls
    outside 0 up to 1. *)
val sample
  :  Config.t
  -> Params.t
  -> seed:int
  -> steps:int
  -> temperature:float
  -> min_p:float
  -> music:Token.t list list * stats:sample_stats
