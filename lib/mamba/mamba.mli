(** The state-space model on the host: the recurrence, the loss and the sampler.

    One step of music is one step of the recurrence and one frame — four voice codes in
    one word — thus the work of a step is constant and it does not know how long the walk
    has run. The four classes of a step enter through four tables that sum, and they leave
    through the same four tables in a chain from the soprano down: that head is era
    four's, unchanged. What changed is the trunk. A Mamba-2 block holds a fixed state
    where era four held a window of keys and values, thus the model has no context length
    at inference and the training window is a training choice alone. The design is
    [docs/mamba.md].

    This module is the reference and not the trainer. The trainer is [jax/mamba/train.py],
    and what stands here is what the gates of the project need: the same loss on the same
    windows, the same walk from the same seed, and the float model that the integer twin
    of the circuit must follow.

    The vocabulary is [Vocab], in the corpus library, because the corpus sizes it. The
    tables of this model are the only thing here that counts them.

    Every draw — the initial parameters and the sampler — comes from [Prng], the
    xorshift32 of the circuit. Thus one seed names one walk in the software, in the
    simulation and on the board. A seed is any integer: it folds into the 32 bits of the
    state, and a seed already inside that range names itself.

    **0 is the one seed that does not cross to the board.** This module folds, thus 0
    takes a live walk here: a tool that draws its seed from a flag, a counter or a stream
    must not get a dead one. The board and [Quantized.Engine] take the seed as it stands,
    and 0 is the fixed point of the recurrence, thus their walk at 0 stands still and is
    not this module's. [Prng.create_folded] states the rule. *)

(** every tensor of the float model is float32 *)
type tensor = (float, Nx.float32_elt) Nx.t

(** [numel shape] is the element count of a shape. It stands beside [Params.shapes],
    because the quantization of [Quantized] sizes its flat tensors from that table and the
    product is the step between the two. *)
val numel : int array -> int

(** the taps of the depthwise convolution: four, the Mamba default. It sizes the tap ring
    of the circuit as well, thus it is stated one time. *)
val conv_taps : int

(** The shape of the model: the numbers that size every tensor here and every register in
    the circuit. There is no context and no window — the recurrence has neither. *)
module Config : sig
  type t =
    { d : int (** the width of the residual stream *)
    ; d_in : int (** the inner width of a block; the expansion of two puts it at [2 d] *)
    ; heads : int (** they split [d_in], thus [d_in] divides by them *)
    ; state : int (** N, the state width of one head *)
    ; layers : int
    }

  (** the shape of [docs/mamba.md]: d 64, d_in 128, 4 heads, state 16, 6 layers *)
  val baseline : t

  (** [head t] is P, the head width: the state is [heads] blocks of [head] by [state]. It
      is a power of four at the baseline, thus the shift rules of the machine hold. *)
  val head : t -> int

  (** [channels t] is the channels the convolution walks: [d_in] of x, then B and C *)
  val channels : t -> int

  (** [projection t] is the width of the input projection: the gate, the convolution input
      and the raw dt, in that order *)
  val projection : t -> int

  (** [of_checkpoint path] reads every width from the tensor shapes, thus a player states
      none of them: the seat table gives [d], the output projection gives [d_in], the dt
      bias gives the heads, the input projection gives the state width, and the tensor
      count gives the layers. Era four had to be told its heads, its context and its span;
      no number of this model stands outside its file.

      It raises [Invalid_argument] when the file is not a checkpoint of this model, and
      the message names which shape refused. *)
  val of_checkpoint : string -> t
end

(** The parameter structure over any tensor type, and the flat order of the checkpoint
    with it: the two tables, then six tensors for each layer. One definition holds the
    shape and the order — [Params] instantiates it with the float tensor and keeps its own
    type opaque, and [Quantized.Model] instantiates it with the integer one. *)
module Params_data : sig
  type 'a t =
    { seats : 'a (** the four tied tables in one tensor, seat 0 first *)
    ; phase : 'a (** the bar-phase table *)
    ; layers : 'a layer array
    }

  and 'a layer =
    { w_in : 'a (** [d] by [projection]: the gate, the convolution input, the raw dt *)
    ; conv : 'a (** [channels] by [conv_taps]: tap k of a channel reads k steps back *)
    ; dt_bias : 'a (** one for each head *)
    ; a_log : 'a (** the log of the decay rate of each head *)
    ; d_skip : 'a (** the skip of each head, around the state *)
    ; w_out : 'a (** [d_in] by [d] *)
    }

  (** the flat order of the tensors — the order of the checkpoint; [of_list] reads the
      same order *)
  val to_list : 'a t -> 'a list

  (** [of_list ~layers items] reads the order of [to_list]. It raises [Invalid_argument]
      when the count of items is not two tables and six for each of [layers]. *)
  val of_list : layers:int -> 'a list -> 'a t
end

(** The weights of the float model: the two tables — the four tied seat tables in one
    tensor, and the bar phase — and six tensors for each layer. *)
module Params : sig
  type t

  (** the shapes of the tensors in the flat order of [Params_data.to_list], from the
      configuration. The quantization of [Quantized] reads the same table, thus the two
      models cannot disagree about what a checkpoint holds. *)
  val shapes : Config.t -> int array list

  (** [init config ~seed] draws the initial parameters through [Prng], thus a test needs
      no checkpoint file and the same seed gives the same weights.

      The draw follows the SHAPE of [jax/mamba/train.py] and not its values: the matrices
      are normal at scale 0.02, the decay rate is uniform in \[1, 16\], the step is
      uniform in \[0.001, 0.1\] through its inverse softplus, and the skip opens at one.
      The two generators are different, thus only a trained checkpoint crosses that seam;
      what must agree is the distribution, because a drift report over weights that put
      every decay near one would measure a model this era does not train. *)
  val init : Config.t -> seed:int -> t

  (** the tensors in the flat order of [Params_data.to_list] *)
  val to_list : t -> tensor list

  (** [load config ~path] is the parameters of the checkpoint at [path], which is the
      safetensors file that the JAX trainer writes: the tensors named "0" upward, in the
      order that [Config.of_checkpoint] reads. [config] states the shapes, and a shape of
      the file that does not agree with them raises [Invalid_argument]. *)
  val load : Config.t -> path:string -> t
end

(** The memory of the walk, and the only thing that crosses from one step to the next: for
    each layer a state and the convolution taps behind the step.

    It is a value and not a buffer. A walk that wants to branch keeps two of them, and
    nothing in this module writes one in place. *)
module Memory : sig
  type t

  (** [origin config ~batch] is the memory at the head of a walk: a zero state and empty
      taps, for [batch] independent walks at once.

      A training window opens here and the boot of the sampler opens here, thus the seam
      condition of the packed corpus is the condition the model trains on. *)
  val origin : Config.t -> batch:int -> t
end

(** [loss config params ~windows] is the cross entropy of the frames of [windows],
    in **nats for each step** — the sum over the four seats, and the mean over the steps.

    Give it the windows of [Jsb.windows], which every referee cuts the same way. A window
    holds [context + 1] steps: the first [context] are the inputs and the last [context]
    are the labels, thus every position of every window weighs one. There is no mask and
    no weight: no frame is illegal, and a window of a packed stream is always full.

    Each window opens on a zero state. A window is therefore not a slice of a longer walk,
    and it needs no lead-in: the recurrence begins where the boot of the sampler begins.

    **The unit is the measure**, and era four speaks it on these same windows: the two
    eras compare by this number for the first time, and the elected model of era four
    stands at 1.6282. Across an encoding they still do not compare — the frame states
    which voice holds which pitch and a sentence stated only the set — so rank inside one
    encoding with this number, and across them with the ear.

    A second number belongs beside it and does not belong here: 77.91 percent of the voice
    slots repeat the step before, thus a recurrence that decays too fast scores well on
    this mean and plays a drone. The trainer reports the loss over the steps where two or
    more voices move; a reference measures one thing.

    It raises [Invalid_argument] when [windows] is empty. *)
val loss : Config.t -> Params.t -> windows:Jsb.stream list -> float

(** [sample config params ~seed ~steps ~temperature ~min_p] draws one endless walk of
    [steps] steps and gives the frame of each one. The walk knows no piece and takes no
    boundary: [Frame.events_of_frames] turns it into the events of the wire.

    The boot is a lead-in of silence: the walk opens with one bar of silent frames, and
    the model opens the music itself. The state opens at zero, which is where a training
    window opens, thus this is a condition the model trained on. The lead-in counts inside
    [steps] and stands at the head of the result, because it is silence the walk really
    plays.

    One step is one step of the recurrence and then a chain of four draws on the host. No
    mask stands before a draw, because no frame is illegal. [min_p] removes each class
    whose tempered weight is below [min_p] of the peak's; the peak always stays, thus a
    draw always exists, and zero turns the filter off.

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

(** [forward config params memory ~frame ~phase] takes one step of the recurrence: the
    memory after it, and the residual stream the head of the NEXT step reads.

    This is the whole trunk as one function, and it is what makes a long comparison
    affordable: the drift report of [Quantized] walks the float model beside the integer
    one step for step, where era four had to re-run a whole window at every step. A caller
    holds the memory and nothing else. *)
val forward
  :  Config.t
  -> Params.t
  -> Memory.t
  -> frame:int
  -> phase:int
  -> Memory.t * float array

(** [logits config params ~stream ~drawn] is the raw logits of every seat over the
    residual stream of one step, indexed by seat and each one over the classes of [Vocab].

    [drawn] holds the classes the chain conditions on — seats 3, 2 and 1 are read, as the
    chain reads them. The drift report of [Quantized] walks the quantized engine and reads
    this at each of its four draws, thus the two models are compared over one history and
    one chain, and what stands between them is the quantization alone. *)
val logits
  :  Config.t
  -> Params.t
  -> stream:float array
  -> drawn:int array
  -> float array array

(** [draw_class raw ~temperature ~min_p ~uniform] is the draw of one seat as one function:
    the tempered weights over the classes, the min-p floor, and the class whose running
    total passes [uniform] times the total. [sample] draws through it, and the drift walk
    of [Quantized] draws through it with the same uniform as the quantized engine, thus
    the two pipelines are comparable pick for pick.

    The rules of [sample] on [temperature] and [min_p] hold here; this function does not
    check them. *)
val draw_class : float array -> temperature:float -> min_p:float -> uniform:float -> int

(** [check_policy ~temperature ~min_p] raises [Invalid_argument] when [temperature] is 0
    or less, or when [min_p] falls outside 0 up to 1. [sample] and the quantized twin both
    state these bounds, thus one module owns them and one message answers each. *)
val check_policy : temperature:float -> min_p:float -> unit

(** The draw of era four, carried over: temperature 1.0 and min-p 0.05.

    Every player takes its flag default from here, and [Quantized.Model] takes the policy
    it bakes into the bitstream from here, thus the numbers stand one time and the ear
    moves them in one place. This era re-elects them with the whole chain in view; until
    it does, the two eras are auditioned on one policy. The audition tool
    [jax/mamba/infer.py] states them again, because no constant crosses the language seam. *)
val elected_temperature : float

val elected_min_p : float

(** The checkpoint seam, for the gates that cross it. It stands here because the naming
    rule of the file is this module's: the tensors are named "0" upward, in the flat order
    of [Params_data.to_list]. *)
module For_test : sig
  (** [with_checkpoint tensors ~f] writes [tensors] to a temporary safetensors file under
      that rule — the file the JAX trainer writes — and gives [f] its path. The file goes
      when [f] gives and when [f] raises.

      [Config.of_checkpoint], [Params.load] and [Quantized.Model.of_checkpoint] are the
      three readers of the seam, thus one writer serves the gates of both modules and no
      gate reads a file that git ignores. *)
  val with_checkpoint : tensor list -> f:(string -> 'a) -> 'a

  (** [refusal ~path f] is the message of the [Invalid_argument] that [f] raises, with
      [path] taken out of it, and ["no raise"] when [f] raises nothing. A reader of the
      checkpoint names the file in its refusal, and the file of a gate is a temporary one,
      thus the name must leave the message before an expected block holds it. *)
  val refusal : path:string -> (unit -> unit) -> string
end
