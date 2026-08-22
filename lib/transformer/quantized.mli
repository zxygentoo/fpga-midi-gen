(** The integer twin of the step-frame model: the reference the circuit must equal.

    The float model of [Transformer] is what the ear elected. This module is the same
    network in the arithmetic the board can hold — int8 weights with a power-of-two
    exponent for each tensor, int8 KV rings, and the sampler folded into integers — and
    the circuit of era four must equal it operation for operation, not approximately.

    Nothing here approximates on purpose. Every shift, every floor and every table is a
    rule that the RTL reads from this module rather than restating, thus a change of the
    arithmetic changes both sides at one time.

    What the quantization costs is a measurement and not a promise: [Drift] states it, on
    the walk the board really takes.

    The design is [docs/transformer.md]. *)

(** The fixed-point formats of the machine, and the constants that cross between the
    reference and the circuit. A Q number holds [value * 2^-q]. *)
module Constants : sig
  (** the residual stream: Q16 in int32 *)
  val h_q : int

  (** the normed vector, and the score of attention: Q12 in int16 *)
  val y_q : int

  (** the query, the keys, the values and the context: Q12 in int16. It is a name of its
      own because the rings store these rows and the ring is where the format is a design
      choice, not an accident of the datapath. *)
  val kv_q : int

  (** the feed-forward hidden vector: Q10 in int16 *)
  val hid_q : int

  (** the rms epsilon of the float model, in the Q of the squared stream *)
  val eps_q : int

  (** A fixed-point multiplier: the value stands for [q_value * 2^-q]. The Q travels with
      the value because the two are one fact — a multiply that takes the wrong shift is
      silently wrong, and both the reference and the circuit apply these scales. *)
  type scale =
    { q_value : int
    ; q : int
    }

  (** [apply s v] scales [v] by [s], toward negative infinity — an arithmetic shift, as
      the circuit's. *)
  val apply : scale -> int -> int

  (** log2(e): the exp2 form of the softmax exponent *)
  val log2e : scale

  (** the exp2 table of the softmax and the sampler as the circuit's ROM: 256 entries of
      16 bits, exp2 of -j/256 in Q15 *)
  val exp2_bits : Hardcaml.Bits.t array

  (** [score_shift ~head_d] carries a score walk's sum from Q(2 [kv_q]) to Q[y_q] and
      applies the 1/sqrt([head_d]) of the reference in the same shift, thus [head_d] is a
      power of four. *)
  val score_shift : head_d:int -> int

  (** [slope_exponent ~span ~heads ~head] is the ALiBi exponent of one head: its slope is
      2^-(this), thus the penalty of an age is a shift and never a multiply. *)
  val slope_exponent : span:int -> heads:int -> head:int -> int
end

(** A vector of the integer model, and the two measures that compare one against the float
    vector of the same place. *)
module Tensor : sig
  type t = int array
  type floats = float array

  (** [same_peak q f] is true when the two vectors elect the same index — the top-1
      agreement of the drift report. A tie keeps the first index on both sides. *)
  val same_peak : t -> floats -> bool

  (** [cosine q f] is the cosine between the two vectors. *)
  val cosine : t -> floats -> float
end

(** The weights in the form the bitstream carries, and the sampling policy in integers. *)
module Model : sig
  (** one quantized weight tensor: the int8 values, flat in the row-major order of the
      float checkpoint, and the exponent [e] that reads them — the value of an element is
      [q * 2^-e] *)
  type quantized =
    { q : Tensor.t
    ; e : int
    }

  (** the structure of [Transformer.Params_data] over the quantized tensor: one definition
      of the shape and of the checkpoint order serves the float model and this one *)
  type params = quantized Transformer.Params_data.t

  type layer = quantized Transformer.Params_data.layer

  type t =
    { config : Transformer.Config.t
    ; params : params
    ; temper : Constants.scale
    (** the sampling temper, log2(e) / T — folded with the exp2 form *)
    ; min_weight : int (** the min-p share of the peak weight 2^15 *)
    }

  (** [check_shape t] raises when the model breaks a rule that its consumers assume: [d]
      and the context are powers of two, the head width is a power of four, the layer
      count agrees with the tensors, the seat table holds one row for each seat and class,
      and the seat and phase tables share one exponent. The record is open, thus a model
      that no constructor here made can break a rule; the circuit calls this at
      elaboration, where a bad shape must fail loudly. *)
  val check_shape : t -> unit

  (** the ROM image of the circuit: every tensor in the checkpoint order, one byte for
      each weight, two's complement *)
  val rom_bits : t -> Hardcaml.Bits.t array

  (** the base of each tensor inside the ROM, in the shape of the parameters — the address
      the circuit adds its own offsets to. The four seat tables stand inside one tensor,
      thus seat [s] begins at [seats + s * classes * d], which is a shift and an add. *)
  val rom_bases : t -> int Transformer.Params_data.t

  (** [of_checkpoint config path] loads the float checkpoint and quantizes it. The
      temperature and min-p default to [Transformer.elected_temperature] and
      [Transformer.elected_min_p], and they are part of the model here, because the
      bitstream carries them: the board commits to the numbers its elaboration was given.

      It raises [Invalid_argument] when the file holds no tensor of a name the order
      wants, or when a tensor holds a count of values that the configuration does not fit. *)
  val of_checkpoint
    :  ?temperature:float
    -> ?min_p:float
    -> Transformer.Config.t
    -> string
    -> t

  module For_test : sig
    (** the shape of a test model: small enough to run in a simulation, and the same
        structure as the model of the era. The gates of [Source] take it, thus the circuit
        and the reference are compared at a shape a test can afford. *)
    val config : Transformer.Config.t

    (** [init config ~seed] is a model of drawn weights in the shapes of [config],
        quantized under the draw of the era: the elaboration of a circuit and the engine
        both take one, thus a test reads no checkpoint and no file that git ignores. The
        weights are not the weights the trainer draws from the same number — only a
        trained checkpoint crosses that seam. *)
    val init : Transformer.Config.t -> seed:int -> t
  end
end

(** One running inference, as a value: an operation gives the engine after it. *)
module Engine : sig
  type t

  (** one draw of the chain: the seat it drew, the logits it read, the uniform it took,
      and the class it chose. The drift report reads all four; the walk itself needs only
      the class. *)
  type draw =
    { seat : int
    ; logits : Tensor.t
    ; uniform : float
    ; drawn : int
    }

  (** what one step of the walk states: the frame of the step, and the draws the chain
      took to reach it. The draws come in the order they happened — the soprano first —
      and they are empty through the silent lead-in, which draws nothing. *)
  type step =
    { frame : int
    ; draws : draw list
    }

  (** [init model ~seed] is the engine at its origin: an empty ring, no residual, and the
      PRNG at [seed] — [Prng.create] and not the fold, thus a seed of the board names the
      walk of the board. The seed is any 32-bit value.

      0 is the standing walk. The slide switches can state it, because all of them down is
      the rest position of the panel, thus the circuit plays it and this reference plays
      it: the generator never moves, every uniform is 0, and each seat takes the first
      class that survives min-p.

      The lead-in is not here. It is the first steps of the walk itself, thus a caller
      that counts steps counts the steps [Transformer.sample] counts, and the two walks
      compare index for index. *)
  val init : Model.t -> seed:int -> t

  (** [next_step t] takes one step: the engine after it, and what the step states.

      Through the lead-in of one bar the step is a silent frame and no draw at all — the
      generator does not move, exactly as it does not move in the float sampler. After it,
      one pass of the network gives the stream, and the chain draws the four seats from
      the soprano down, each one reading the stream the seats above it have written.

      No mask stands before a draw, because no frame is illegal. The pick needs no
      fallback either: the threshold is a floor of [u * total] over 2^24 with [u] below
      2^24, thus it is below the total, thus some running total passes it and the class it
      names always holds the weight the floor left standing. *)
  val next_step : t -> t * step
end

(** What the quantization costs, measured on the walk the board takes. *)
module Drift : sig
  type stats =
    { steps : int (** the steps of the walk, the silent lead-in inside *)
    ; draws : int (** four for each drawn step: one for each seat of the chain *)
    ; same_peak : int (** the draws where both models elect the same class *)
    ; same_draw : int (** the draws where both models pick the same class *)
    ; mean_cosine : float
    }

  (** [walk config params ~steps ~seed] draws the quantized walk and scores the float
      model against it, draw for draw.

      The walk quantizes [params] itself, under the draw of the era, thus the pair cannot
      slip: one weights source and one policy. The float pass is teacher-forced on the
      quantized history and on the quantized chain — it reads the frames the engine drew
      and conditions each seat on the classes the engine chose — thus what the report
      measures is the quantization and never a walk that parted for another reason.

      The same-draw share reads the float draw on the very uniform the engine took, thus a
      difference there is the arithmetic and not the generator. *)
  val walk
    :  Transformer.Config.t
    -> Transformer.Params.t
    -> steps:int
    -> seed:int
    -> stats
end
