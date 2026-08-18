(** The step-frame model on the host: the network, the loss and the sampler.

    One step of music is one position and one frame — four voice codes in one word — thus
    the work of a step is constant: one pass, always. The four classes of a step enter
    through four tables that sum, and they leave through the same four tables in a chain
    from the soprano down. The network under that head is a decoder with no bias terms,
    RMSNorm before each sublayer, ALiBi for the position, and [d_ff = 4 d]. The design is
    [docs/transformer_model.md].

    This module is the reference and not the trainer. The trainer is
    [jax/transformer/train.py], and what stands here is what the gates of the project
    need: the same loss on the same windows, the same walk from the same seed, and the
    float model that the integer twin of the circuit must follow.

    The vocabulary is [Vocab], in the corpus library, because the corpus sizes it. The
    tables of this model are the only thing here that counts them.

    Every draw — the initial parameters and the sampler — comes from [Prng], the
    xorshift32 of the circuit. Thus one seed names one walk in the software, in the
    simulation and on the board. A seed is any integer: it folds into the 32 bits of the
    state, and a seed already inside that range names itself. *)

(** The shape of the model: the numbers that size every tensor here and every register in
    the circuit. *)
module Config : sig
  type t =
    { d : int (** the width of the residual stream *)
    ; layers : int
    ; heads : int (** they split the width at run time, thus [d] divides by them *)
    ; context : int (** the attention window, in steps *)
    ; slope_span : int
    (** The ALiBi exponent span: the slope of head k is 2 ** -(span (k+1) / heads), and a
        head subtracts its slope times the distance from its logits. The span sets how far
        the gentlest head sees. Four is elected, and it is the most robust number of the
        era: the means of 4 and 8 are a dead heat, and the spread over six seeds is 5 to 7
        times tighter at 4, replicated at two step budgets. Every head is then local, and
        a seed cannot latch onto whatever distant structure its initial values favour. *)
    }

  (** the model the ear elected on 2026-08-18: d 64, 6 layers, 4 heads, context 256, span
      4 *)
  val baseline : t

  (** [of_checkpoint path ~heads ~context ~slope_span] reads the width and the layer count
      from the tensor shapes of the checkpoint, thus a player states neither: the file
      holds two tables and six tensors for each layer, and the seat table states the seats
      and the classes, which are checked. The heads and the context are not in the file —
      the heads only split the width at run time, ALiBi holds no position table, and the
      context is a choice of the draw in any case, because a model trained long can sample
      short. *)
  val of_checkpoint : string -> heads:int -> context:int -> slope_span:int -> t
end

(** The weights of the float model: the two tables — the four tied seat tables in one
    tensor, and the bar phase — and six tensors for each layer. *)
module Params : sig
  type t

  (** [load config ~path] is the parameters of the checkpoint at [path], which is the
      safetensors file that the JAX trainer writes: the tensors named "0" upward, in the
      order that [Config.of_checkpoint] reads. [config] states the shapes, and a shape of
      the file that does not agree with them raises [Invalid_argument]. *)
  val load : Config.t -> path:string -> t
end

(** [loss config params ~windows] is the cross entropy of the frames of [windows],
    in **nats for each step** — the sum over the four seats, and the mean over the steps.

    Give it the windows of [Jsb.windows], which every referee cuts the same way. A window
    holds [context + 1] steps: the first [context] are the inputs and the last [context]
    are the labels, thus every position of every window weighs one. There is no mask and
    no weight: no frame is illegal, and a window of a packed stream is always full.

    **The unit is the measure.** A mean over the predictions would divide by four here and
    by 2.673 in the encoding of era three, thus it compares nothing across the two. Even
    nats for each step do not compare them — the frame states which voice holds which
    pitch, and a sentence stated only the set — so rank inside one encoding with this
    number, and across them with the ear.

    A second number belongs beside it and does not belong here: 77.91 percent of the voice
    slots repeat the step before, thus a model that holds its chord for ever scores well
    on this mean and plays a drone. The trainer reports the loss over the steps where two
    or more voices move; a reference measures one thing.

    It raises [Invalid_argument] when [windows] is empty. *)
val loss : Config.t -> Params.t -> windows:Jsb.stream list -> float

(** [sample config params ~seed ~steps ~temperature ~min_p] draws one endless walk of
    [steps] steps and gives the frame of each one. The walk knows no piece and takes no
    boundary: [Frame.events_of_frames] turns it into the events of the wire.

    The boot is a lead-in of silence: the walk opens with one bar of silent frames, and
    the model opens the music itself. Attention needs one position, and the packed corpus
    holds a run of silent frames at every seam, thus this is a condition the model trained
    on. The lead-in counts inside [steps] and stands at the head of the result, because it
    is silence the walk really plays. It is measured and settled: over twelve seeds the
    first note fell inside one bar of the end of the lead-in, always on a multiple of four
    steps, and half of them on the downbeat itself.

    One step is one pass of the network and then a chain of four draws on the host. No
    mask stands before a draw, because no frame is illegal. [min_p] removes each class
    whose tempered weight is below [min_p] of the peak's; the peak always stays, thus a
    draw always exists, and zero turns the filter off. Temperature 1.0 with min-p 0.05 is
    elected by ear over a sweep of 0.7 to 1.3 against 0.0039 to 0.15.

    The same seed gives the same music. It raises [Invalid_argument] when [temperature] is
    0 or less, or when [min_p] falls outside 0 up to 1. *)
val sample
  :  Config.t
  -> Params.t
  -> seed:int
  -> steps:int
  -> temperature:float
  -> min_p:float
  -> int array
