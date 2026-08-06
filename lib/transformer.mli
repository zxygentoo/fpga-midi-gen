(** The transformer model on the host: the network, the loss and the sampler.

    The network follows [docs/transformer_model.md]: decoder-only, no bias terms, RMSNorm
    with no scale, ALiBi for the position, a bar-phase table, and the embedding table tied
    with the output head. The host model is float32; the integer ladder comes after the
    audition. Raven carries the work: Nx computes, Rune differentiates, and Kaun gives
    AdamW, the loss and the checkpoint format. *)

type tensor = (float, Nx.float32_elt) Nx.t

module Config : sig
  type t =
    { d : int (** the width of the residual stream *)
    ; layers : int
    ; heads : int
    ; context : int (** the attention window, in tokens *)
    }

  (** the baseline of the design document: d 64, layers 2, heads 4, context 256 *)
  val baseline : t
end

module Params : sig
  type t

  (** [draw config ~seed] draws the initial parameters: normal, scale 0.02. The OCaml PRNG
      makes the draw, thus the seed is an input and the same seed gives the same start. *)
  val draw : Config.t -> seed:int -> t

  (** the flat order of the tensors; [of_list] reads the same order *)
  val to_list : t -> tensor list

  val of_list : Config.t -> tensor list -> t

  (** the tree for the optimizer and the checkpoint: the flat order as a [Ptree] list *)
  val to_ptree : t -> Kaun.Ptree.t

  val of_ptree : Config.t -> Kaun.Ptree.t -> t
end

(** The instrumentation of one sampling run: how often the guards fire. [refused] is the
    mean legal probability mass that the min-p filter removed each draw. [illegal_mass] is
    the mean raw mass the model put on masked codes each draw — the rate at which an
    unmasked sampler would emit an illegal token. [illegal_top] is the share of draws
    whose raw argmax was itself illegal. *)
module Sample_stats : sig
  type t =
    { refused : float
    ; illegal_mass : float
    ; illegal_top : float
    ; draws : int
    }
end

(** [loss config params ~codes ~phases] is the cross entropy of the next code, a scalar,
    over the whole vocabulary. A row of [codes] holds [length + 1] codes: the inputs and
    the shifted labels. No mask sits in the loss: the model learns the instrument from the
    data, and the guard of the sampler holds the line at the draw. *)
val loss
  :  Config.t
  -> Params.t
  -> codes:int array array
  -> phases:int array array
  -> tensor

(** [masked_loss config params ~codes ~phases ~masks] is the loss of the mask era, kept
    for controls: the grammar mask sits inside the softmax, thus the model never learns
    the instrument. [masks] holds the legal set of each label, from the walk of the whole
    chorale. Its numbers live on the masked scale, not the raw scale of [loss]. *)
val masked_loss
  :  Config.t
  -> Params.t
  -> codes:int array array
  -> phases:int array array
  -> masks:bool array array array
  -> tensor

(** The guard of the sampler. [Grammar] is the full mask of the corpus encoding — a model
    whose training loss carried the mask needs it, because its raw mass outside is
    untrained. [Hazards] is the safety floor alone, for a model that learned the grammar
    from data. *)
module Guard : sig
  type t =
    | Grammar
    | Hazards
end

(** [sample config params ~seed ~steps ~temperature ~min_p] draws [steps] steps from
    silence. One element of the first result is one drawn step: the events of its
    sentence, without the [End]. The mask of the design document guards every draw, thus
    each sentence is valid. [min_p] removes each legal token whose tempered probability is
    below [min_p] of the peak's; the peak always stays, thus a draw always exists, and
    zero turns the filter off. The same seed gives the same music. *)
val sample
  :  Config.t
  -> Params.t
  -> seed:int
  -> steps:int
  -> temperature:float
  -> min_p:float
  -> guard:Guard.t
  -> Token.t list list * Sample_stats.t
