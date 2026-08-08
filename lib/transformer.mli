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

type tensor = (float, Nx.float32_elt) Nx.t

module Config : sig
  type t =
    { d : int (** the width of the residual stream *)
    ; layers : int
    ; heads : int
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

  (** [of_checkpoint path ~heads ~context] reads the width and the layer count from the
      tensor shapes of the checkpoint, thus a player states neither. The heads and the
      context are not in the file: no tensor shape holds them, because the heads only
      split the width at run time and ALiBi holds no position table. The context is a
      choice of the draw in any case — a model trained long can sample short. *)
  val of_checkpoint : string -> heads:int -> context:int -> slope_span:int -> t
end

module Params : sig
  type t

  (** [draw config ~seed] draws the initial parameters: normal, scale 0.02. [Prng] and
      Box-Muller make the draw, thus the seed is an input and the same seed gives the same
      start. The tensors come in the order of [to_list]. *)
  val draw : Config.t -> seed:int -> t

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
    whose raw argmax was itself illegal. *)
module Sample_stats : sig
  type t =
    { refused : float
    ; illegal_mass : float
    ; illegal_top : float
    ; draws : int
    }
end

(** The dropout of one training step. The JAX sweep of 2026-08-07 found the rate scales
    with the model: 0.1 at d 64, 0.2 at d 128 and at the long context. The masks are drawn
    before the gradient runs and passed in, thus the step stays pure and the seed
    reproduces it. *)
module Dropout : sig
  type t

  (** the identity: every mask is one, and the forward pass is the inference pass *)
  val none : t

  (** [draw config ~rate ~batch ~length ~seed] draws the masks of one step: the embedding
      sum and the two branches of each layer. A rate of zero is [none], and a rate of 1 or
      more raises: it drops every unit, and the scale of the survivors divides by zero. *)
  val draw : Config.t -> rate:float -> batch:int -> length:int -> seed:int -> t
end

(** [loss config params ~codes ~phases ~masks ~weights ~dropout] is the cross entropy of
    the next code, a scalar. A row of [codes] holds [length + 1] codes: the inputs and the
    shifted labels. [masks] holds the legal set of each label, from the walk of the whole
    chorale, and it sits inside the softmax: the model spends no mass on a code that the
    sampler would refuse. Therefore its raw mass outside the legal set stays untrained,
    and [sample] must carry the same mask at every draw. [weights] holds one weight for
    each label position; zero drops the position from the mean, which is how the padding
    of a short piece stays out of the loss — a padded label would teach the walk to hold
    the last chord and emit END for ever. *)
val loss
  :  Config.t
  -> Params.t
  -> codes:int array array
  -> phases:int array array
  -> masks:bool array array array
  -> weights:float array array
  -> dropout:Dropout.t
  -> tensor

(** [sample config params ~seed ~steps ~temperature ~min_p] draws [steps] steps from the
    boot: an empty context, then [Start] at phase zero — power on, music on. One element
    of the first result is one drawn step: the events of its sentence, without the [End].
    The mask of the design document guards every draw, thus each sentence is valid.
    [min_p] removes each legal token whose tempered probability is below [min_p] of the
    peak's; the peak always stays, thus a draw always exists, and zero turns the filter
    off. The same seed gives the same music. *)
val sample
  :  Config.t
  -> Params.t
  -> seed:int
  -> steps:int
  -> temperature:float
  -> min_p:float
  -> Token.t list list * Sample_stats.t
