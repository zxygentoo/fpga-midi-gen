(** The integer twin of the state-space model: the reference the circuit must equal.

    The float model of [Mamba] is what the trainer produced. This module is the same
    recurrence in the arithmetic the board can hold — int8 weights with a power-of-two
    exponent for each tensor, an int16 state, and the sampler folded into integers — and
    the circuit of era five must equal it operation for operation, not approximately.

    Nothing here approximates on purpose. Every shift, every floor and every table is a
    rule that the RTL reads from this module rather than restating, thus a change of the
    arithmetic changes both sides at one time.

    **The state is the tensor this era must not coarsen.** Era four's ring took the top
    byte of each row because 48 tiles of block RAM forced it; a state-space model has no
    such pressure and it has the opposite exposure — an error in the state carries
    forward, where a ring error died with its window. The state is int16 here and the
    drift walk runs long, past many decay lifetimes, because that is the only way to see a
    cumulative error.

    **The key and value ring of the Zamba head keeps era four's coarse byte**, and that is
    the same argument read the other way: a ring error still dies with its window. The
    block RAM is there to widen it, and [test/test_mamba_drift.ml] records what widening
    it would buy.

    What the quantization costs is a measurement and not a promise: [Drift] states it, on
    the walk the board really takes.

    The design is [docs/mamba.md] and [docs/mamba_rtl.md]. *)

(** The fixed-point formats of the machine, and the constants that cross between the
    reference and the circuit. A Q number holds [value * 2^-q]. *)
module Constants : sig
  (** the residual stream: Q16 in int32 *)
  val h_q : int

  (** the normed vector: Q12 in int16 *)
  val y_q : int

  (** The working class of the datapath: Q12 in int16. The projection, the convolution
      output and x, z, B and C after SiLU all take it. It is a name of its own because it
      is a different design choice from [y_q] and only happens to agree with it. *)
  val v_q : int

  (** the state: Q12 in int16, clamped. The sensitive tensor of the era. *)
  val s_q : int

  (** the decay: Q15, unsigned. The exp2 output, whose peak at dt 0 is 2^15. *)
  val alpha_q : int

  (** the gate product, in an int32: two Q12 values multiply and the norm reads them whole *)
  val gate_q : int

  (** the feed-forward hidden after its ReLU: era four's Q10, unchanged *)
  val hid_q : int

  (** [beta = dt * B], the state-inject operand: Q15 in int16. It bounds the injection at
      one, and the drift report prints the share that rode that clamp. *)
  val beta_q : int

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

  (** log2(e): the exp2 form of an exponential *)
  val log2e : scale

  (** era four's exp2 table, read back through its public image: exp2 of -j/256 in Q15.
      One table serves the softmax of that circuit and the decay of this one, and [Exp2]
      is the unit both drive. *)
  val exp2_table : int array

  (** the sigmoid ROM of the circuit: 256 entries of 16 bits, Q15, over a signed Q12
      input. An entry is the sigmoid at the centre of its bucket, thus a value and its
      negative weigh 2^15 together. *)
  val sigmoid_bits : Hardcaml.Bits.t array

  (** the softplus correction ROM: 256 entries of 16 bits, ln(1 + exp(-|v|)) in Q12 over a
      Q12 magnitude. The ramp relu(v) is exact and carries the rest. *)
  val softplus_bits : Hardcaml.Bits.t array

  (** [sigmoid_index v] is the row a Q12 int16 reads: its top eight bits with the sign bit
      flipped, which is no arithmetic at all in the circuit. *)
  val sigmoid_index : int -> int

  (** [softplus_index v] is the row a Q12 int16 reads through its magnitude. The clamp
      catches the one value -32768, whose magnitude does not fit the table. *)
  val softplus_index : int -> int

  (** [decay_scale ~a] is [a * log2(e)] as the constant the Decay op carries: the run time
      multiplies [dt] by it and the exp2 unit reads the product. The constant rides the
      25-bit operand port and [dt] the 18-bit one. *)
  val decay_scale : a:float -> scale

  (** [score_shift ~head_d] brings a product of two Q[v_q] rows to Q[y_q] and applies the
      1/sqrt(head_d) of the reference in the same move, thus the scale costs no multiply.
      [head_d] is a power of two, thus its half-log is exact. It is era four's rule. *)
  val score_shift : head_d:int -> int

  (** [slope_exponent ~span ~heads ~head] is the ALiBi slope of a head as an exponent: the
      slope is [2 ** -(this)], thus the penalty of an age is a shift of the age. *)
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

  (** The structure of the ROM image: the two tables, then the matrices of each layer.

      It is NOT the structure of the checkpoint. A checkpoint BLOCK holds six tensors and
      three of them never reach the image — [a_log], [dt_bias] and [d_skip] are [heads]
      values that quantize at elaboration into the constants the ops carry — thus the two
      orders are two structures and neither is implied by the other. An attention layer
      and a feed-forward layer carry every tensor they hold. *)
  module Rom_data : sig
    type 'a t =
      { seats : 'a
      ; phase : 'a
      ; layers : 'a layer array
      }

    and 'a layer =
      | Block of 'a block
      | Attention of 'a attention
      | Feed_forward of 'a feed_forward

    and 'a block =
      { w_in : 'a
      ; conv : 'a
      ; w_out : 'a
      }

    and 'a attention =
      { wq : 'a
      ; wk : 'a
      ; wv : 'a
      ; wo : 'a
      }

    and 'a feed_forward =
      { w1 : 'a
      ; w2 : 'a
      }

    (** the order of the image; [of_list] reads the same order *)
    val to_list : 'a t -> 'a list

    val of_list : plan:Mamba.Kind.t array -> 'a list -> 'a t
  end

  (** The weights of one block as the machine holds them.

      An int8 tensor cannot hold the three per-head numbers. The bias enters a softplus: a
      step of one part in 127 of its range moves [dt] by more than a small [dt] is, and
      the decay would follow it. They quantize at elaboration instead — [a * log2(e)]
      folds into one Q constant for each head, exactly as era four folded log2(e) into the
      temper. *)
  type block =
    { w_in : quantized
    ; conv : quantized
    ; w_out : quantized
    ; decay : Constants.scale array (** [a * log2(e)], one for each head *)
    ; dt_bias : Tensor.t (** Q12 *)
    ; d_skip : Tensor.t (** Q12 *)
    }

  (** the four matrices of the Zamba head: [wq] and [wk] are [2 d] by [d], because the
      query and the key read the normed stream beside the normed embedding *)
  type attention =
    { wq : quantized
    ; wk : quantized
    ; wv : quantized
    ; wo : quantized
    }

  type feed_forward =
    { w1 : quantized
    ; w2 : quantized
    }

  type layer =
    | Block of block
    | Attention of attention
    | Feed_forward of feed_forward

  type t =
    { config : Mamba.Config.t
    ; seats : quantized
    ; phase : quantized
    ; layers : layer array
    ; temper : Constants.scale
    (** the sampling temper, log2(e) / T — folded with the exp2 form *)
    ; min_weight : int (** the min-p share of the peak weight 2^15 *)
    }

  (** [check_shape t] raises when the model breaks a rule that its consumers assume: the
      two widths, the ring depth and every field of the state and tap addresses are powers
      of two, the layers agree with the plan kind for kind, each block carries one
      constant for each head, the seat table holds one row for each seat and class, and
      the seat and phase tables share one exponent. The record is open, thus a model that
      no constructor here made can break a rule; the circuit calls this at elaboration,
      where a bad shape must fail loudly. *)
  val check_shape : t -> unit

  (** the ROM image of the circuit: every tensor of [Rom_data] in its order, one byte for
      each weight, two's complement *)
  val rom_bits : t -> Hardcaml.Bits.t array

  (** the base of each tensor inside the ROM, in the shape of the image — the address the
      circuit adds its own offsets to. The four seat tables stand inside one tensor, thus
      seat [s] begins at [seats + s * classes * d], which is a shift and an add. *)
  val rom_bases : t -> int Rom_data.t

  (** the tensors of the image, in its order; the gates read it beside [rom_bases] *)
  val rom_tensors : t -> quantized list

  (** [of_checkpoint config path] loads the float checkpoint and quantizes it. The
      temperature and min-p default to [Mamba.elected_temperature] and
      [Mamba.elected_min_p], and they are part of the model here, because the bitstream
      carries them: the board commits to the numbers its elaboration was given.

      It raises [Invalid_argument] when the file holds no tensor of a name the order
      wants, or when a tensor holds a count of values that the configuration does not fit. *)
  val of_checkpoint : ?temperature:float -> ?min_p:float -> Mamba.Config.t -> string -> t

  module For_test : sig
    (** the shape of a test model: small enough to run in a simulation, and the same
        structure as the model of the era. The gates of [Source] take it, thus the circuit
        and the reference are compared at a shape a test can afford. *)
    val config : Mamba.Config.t

    (** [init config ~seed] is a model of drawn weights in the shapes of [config],
        quantized under the draw of the era: the elaboration of a circuit and the engine
        both take one, thus a test reads no checkpoint and no file that git ignores. The
        draw is [Mamba.Params.init], thus the decay rates and the skip come out where the
        trainer puts them and a drift report over this model measures the arithmetic this
        era really runs. *)
    val init : Mamba.Config.t -> seed:int -> t
  end
end

(** The clamps a walk met, and the chances each one had.

    The formats of this era are chosen with margin and not metered on a trained
    checkpoint. A clamp that fires is therefore the finding that says which format is
    wrong, and it must be counted rather than assumed away: era four could let a hot
    signal die with its window, and here the state carries an error forward. *)
module Clamps : sig
  type t =
    { dt : int
    ; dt_seen : int
    ; beta : int
    ; beta_seen : int
    ; state : int
    ; state_seen : int
    }

  (** [share hit seen] is the share, and 0 when nothing was seen *)
  val share : int -> int -> float
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

  (** [init model ~seed] is the engine at its origin: a zero state, an empty tap ring, no
      residual, and the PRNG at [seed] — [Prng.create] and not the fold, thus a seed of
      the board names the walk of the board. The seed is any 32-bit value.

      0 is the standing walk. The slide switches can state it, because all of them down is
      the rest position of the panel, thus the circuit plays it and this reference plays
      it: the generator never moves, every uniform is 0, and each seat takes the first
      class that survives min-p.

      The lead-in is not here. It is the first steps of the walk itself, thus a caller
      that counts steps counts the steps [Mamba.sample] counts, and the two walks compare
      index for index. *)
  val init : Model.t -> seed:int -> t

  (** [next_step t] takes one step: the engine after it, and what the step states.

      Through the lead-in of one bar the step is a silent frame and no draw at all — the
      generator does not move, exactly as it does not move in the float sampler. After it,
      one step of the recurrence gives the stream, and the chain draws the four seats from
      the soprano down, each one reading the stream the seats above it have written.

      No mask stands before a draw, because no frame is illegal. The pick needs no
      fallback either: the threshold is a floor of [u * total] over 2^24 with [u] below
      2^24, thus it is below the total, thus some running total passes it and the class it
      names always holds the weight the floor left standing. *)
  val next_step : t -> t * step

  (** the clamps the walk has met so far. It accumulates over the whole walk, thus a
      caller that wants the clamps of one stretch takes the difference. *)
  val clamps : t -> Clamps.t

  (** The scalar rules of the engine that a circuit unit must reproduce exactly. The gate
      tests of [Sigmoid] and [Softplus] read them here rather than restate them, thus the
      unit and the reference cannot drift apart in a definition. *)
  module For_test : sig
    (** the residual stream of the engine *)
    val stream : t -> Tensor.t

    (** [layer_streams t ~frame ~phase] is the residual stream after the embed and then
        after each layer of the step [t] would take next — one entry for each time the
        circuit writes the whole stream, in the order it writes them. A frame gate that
        fails says only THAT the circuit and the reference parted; this says where, and it
        is the instrument that found the two address faults and the operand that followed
        its address instead of its data. *)
    val layer_streams : t -> frame:int -> phase:int -> Tensor.t list

    val isqrt : int -> int

    (** [exp2_of_magnitude m] is 2^-(m/2^12) in Q15. It is era four's rule read the other
        way round: that circuit exponentiated a nonpositive score and this one a decay
        that is a magnitude by construction. *)
    val exp2_of_magnitude : int -> int

    val sigmoid_q : int -> int
    val softplus : int -> int
    val silu : int -> int
  end
end

(** What the quantization costs, measured on the walk the board takes. *)
module Drift : sig
  type stats =
    { steps : int (** the steps of the walk, the silent lead-in inside *)
    ; draws : int (** four for each drawn step: one for each seat of the chain *)
    ; same_peak : int (** the draws where both models elect the same class *)
    ; same_draw : int (** the draws where both models pick the same class *)
    ; mean_cosine : float
    ; dt_clamped : float (** the share of dt values that rode the clamp of their format *)
    ; beta_clamped : float
    ; state_clamped : float
    }

  (** [walk config params ~steps ~seed] draws the quantized walk and scores the float
      model against it, draw for draw.

      The walk quantizes [params] itself, under the draw of the era, thus the pair cannot
      slip: one weights source and one policy. The float pass is teacher-forced on the
      quantized history and on the quantized chain — it reads the frames the engine drew
      and conditions each seat on the classes the engine chose — thus what the report
      measures is the quantization and never a walk that parted for another reason.

      **Both models take one step for one step.** Era four had to re-run a whole window at
      every step, which made a long comparison quadratic; here each carries its own
      memory, thus the walk can run past many decay lifetimes — which it must, because a
      state error is cumulative in a way era four never had.

      The same-draw share reads the float draw on the very uniform the engine took, thus a
      difference there is the arithmetic and not the generator. *)
  val walk : Mamba.Config.t -> Mamba.Params.t -> steps:int -> seed:int -> stats
end
