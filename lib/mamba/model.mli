(** The model as the circuit reads it: the formats, the plan, the contract file and the
    ROM image.

    Era five draws one frame of four voice codes for each step of music, as era four does,
    and the head above the trunk is era four's unchanged. What changed is the trunk: a
    Mamba-2 block holds a fixed state where era four held a window of keys and values,
    thus a trunk of blocks alone has no context length at all. THE MODEL IS NOT A TRUNK
    ALONE — a layer is a block, the Zamba attention head or the feed-forward, and the PLAN
    is a fact of the checkpoint. The design of the model is [docs/mamba.md] and the design
    of the circuit is [docs/mamba_rtl.md].

    **NO MODEL IS COMPUTED HERE.** The float model is [jax/mamba/model.py] and the integer
    twin is [jax/mamba/quantized.py]; [jax/tests/test_rtl_mamba.py] states what the
    circuit must do and [bin/gate_mamba.ml] states what it did. What stays here is what
    the CIRCUIT reads: the fixed-point formats, the model as data, its ROM image and the
    bases of that image.

    EVERY WIDTH AND THE PLAN ARE IN THE FILE, thus the elaboration states none of them.
    The seat table gives [d]; the image of the first block gives the projection, the inner
    width, the channels, the taps, the state and the heads; and the shape of each group's
    first tensor names that layer's kind. Two numbers are not facts of a training run and
    travel beside the tensors: the ALiBi [span], which no tensor sizes, and the [ring],
    which is the depth of the key and value ring at INFERENCE and a choice of the player. *)

(** The fixed-point formats of the machine, and the constants that cross between the twin
    and the circuit. A Q number holds [value * 2^-q]. What is one thing across the eras
    comes from [Mgen_nn.Quantized.Constants]; what stands here is era five's own. *)
module Constants : sig
  include module type of Mgen_nn.Quantized.Constants

  (** the value rows of a block and of the attention rings: Q12 in int16 *)
  val v_q : int

  (** the state of the recurrence: Q12 in int16 *)
  val s_q : int

  (** the decay of one step, and the input coefficient: Q15 *)
  val alpha_q : int

  val beta_q : int

  (** the gate product, in an int32: two Q[v_q] values multiply and nothing truncates them
      before the norm that reads them *)
  val gate_q : int

  (** the Q the Decay op's constant carries. The constant rides the 25-bit port and [dt]
      the 18-bit one, which is the way round that costs nothing — [dt] is int16 — and it
      leaves the constant three million units of room where the other order would clamp a
      decay rate above 22. *)
  val decay_q : int

  (** [score_shift ~head_d] carries a score walk's sum from Q(2 [v_q]) to Q[y_q] and
      applies the 1/sqrt([head_d]) of the model in the same shift, thus [head_d] is a
      power of four. *)
  val score_shift : head_d:int -> int
end

(** The kinds of layer. The PLAN of a model is the sequence of them, and a contract file
    states its own plan through the shapes of its image tensors.

    [Attention] is half a Zamba block: the query and the key read the ORIGINAL EMBEDDING
    beside the residual stream — their matrices are [2 d] by [d] — while the value reads
    the stream alone. Era four's plain attention, a square query over the stream, measured
    null in this trunk three times, thus it is not a kind here and a file that holds one
    is refused. *)
module Kind : sig
  type t =
    | Block
    | Attention
    | Feed_forward
  [@@deriving equal, sexp_of]

  (** the tensors a layer of this kind carries into the ROM image: three for a block, four
      for an attention layer, two for the feed-forward *)
  val tensors : t -> int

  (** [spell plan] is one letter for each layer — M a block, Z the Zamba head, F the
      feed-forward — which is how [docs/mamba.md], the checkpoint names and the [--plan]
      flag of the trainer all spell a plan. The elected model is MMMMMMZF. *)
  val spell : t array -> string
end

(** The tensors the ROM carries, in the order it carries them: the two tables, then the
    matrices of each layer. It is NOT the order of the checkpoint — a block holds six
    tensors there and three of them never reach the ROM — thus the two orders are two
    structures and neither is implied by the other. *)
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

  (** the flat order of the image; [of_list] reads the same order *)
  val to_list : 'a t -> 'a list

  (** [of_list ~plan items] reads the order of [to_list]. It raises [Invalid_argument]
      when the items are not two tables and a whole group for each kind of [plan]. *)
  val of_list : plan:Kind.t array -> 'a list -> 'a t
end

(** one quantized tensor: the int8 values flat in the row-major order of the image, and
    the exponent that reads them — [Mgen_nn.Quantized.quantized] *)
type quantized = Mgen_nn.Quantized.quantized =
  { q : int array
  ; e : int
  }

(** The weights of one block as the machine holds them: three tensors in the ROM, and the
    per-head numbers as constants the ops carry.

    [a_log], [dt_bias] and [d_skip] hold one value for each head, and an int8 tensor
    cannot carry them: the bias enters a softplus, where a step of one part in 127 of its
    range moves [dt] by more than a small [dt] is, and the decay would follow it. They
    quantize above the seam into the numbers the ops carry — [a * log2(e)] folds into one
    Q constant for each head — thus the run time never sees them as tensors. *)
type block =
  { w_in : quantized (** [projection] by [d]: TRANSPOSED, see [transpose] *)
  ; conv : quantized (** [channels] by [taps] *)
  ; w_out : quantized (** [d_in] by [d] *)
  ; decay : Constants.scale array (** [a * log2(e)], one for each head *)
  ; dt_bias : int array (** Q12, one for each head *)
  ; d_skip : int array (** Q12, one for each head *)
  }

type attention =
  { wq : quantized (** [2 d] by [d]: the query reads the stream beside the embedding *)
  ; wk : quantized
  ; wv : quantized (** [d] by [d]: the value reads the stream alone *)
  ; wo : quantized
  }

type feed_forward =
  { w1 : quantized (** [d] by [4 d] *)
  ; w2 : quantized (** [4 d] by [d] *)
  }

type layer =
  | Block of block
  | Attention of attention
  | Feed_forward of feed_forward

val kind_of_layer : layer -> Kind.t

(** the whole model the bitstream carries *)
type t =
  { d : int (** the width of the residual stream *)
  ; d_in : int (** the inner width of a block; the expansion of two puts it at [2 d] *)
  ; heads : int (** they split [d_in] and [d], thus both divide by them *)
  ; state : int (** N, the state width of one head *)
  ; taps : int (** K, the width of the depthwise convolution *)
  ; plan : Kind.t array (** the kind of each layer, in order *)
  ; span : int (** the ALiBi span: the slope of head k is [2 ** -(span (k+1) / heads)] *)
  ; ring : int (** the keys and values an attention layer holds behind the step *)
  ; seats : quantized (** the four tied tables in one tensor, seat 0 first *)
  ; phase : quantized (** the bar-phase table *)
  ; layers : layer array
  ; temper : Constants.scale
  (** the sampling temper, [log2e / T]: part of the model, because the bitstream carries
      it *)
  ; min_weight : int (** the min-p share of the peak weight 2^15 *)
  }

(** the count of each kind that owns a memory: the state RAM and the tap ring of the
    circuit are sized by [blocks], and the key and value rings by [attentions] *)
val blocks : t -> int

val attentions : t -> int

(** [ordinals t] is the ordinal of each layer among the layers of its own kind, and it is
    what indexes a memory: the state RAM and the tap ring hold one region for each block,
    and the key and value rings one for each attention layer, thus the seventh layer of
    the elected plan owns ring 0. The per-head rows of the contract file are indexed the
    same way. *)
val ordinals : t -> int array

(** [head t] is P, the head width of a block: the state is [heads] blocks of [head] by
    [state]. It is a power of four at the baseline, thus the shift rules of the machine
    hold. *)
val head : t -> int

(** [head_d t] is the head width of an attention layer, which splits [d] and not [d_in] *)
val head_d : t -> int

(** [channels t] is the channels the convolution walks: [d_in] of x, then B and C *)
val channels : t -> int

(** [projection t] is the width of the input projection: the gate, the convolution input
    and the raw dt, in that order *)
val projection : t -> int

(** [check_shape t] raises when the model breaks a rule its consumers assume: every field
    of an address is a power of two, the plan agrees with the layers, every block carries
    one per-head value for each head, and the seat and phase tables share one exponent.
    The record is open, thus a model no constructor here made can break a rule; the
    elaboration calls this, where a bad shape must fail loudly. *)
val check_shape : t -> unit

(** [of_int8_checkpoint path] is the model of one CONTRACT FILE — the quantized model that
    [jax/mamba/quantized.py] writes, and the only thing that crosses the seam for a build.
    The quantization happens above the seam, one time, thus this reader quantizes nothing:
    it takes the image, the exponents, the per-head rows, the span, the ring, the temper
    and the min-p share as they stand.

    THE PLAN AND EVERY WIDTH COME OUT OF THE SHAPES, by the rule the module comment
    states, thus the file carries no plan of its own that a reader could disagree with.

    It raises [Invalid_argument] when a tensor is missing, when the image does not fill
    whole layer groups, when a layer opens with a shape no kind holds, or when a rule of
    [check_shape] does not hold. *)
val of_int8_checkpoint : string -> t

(** the ROM image of the circuit: every tensor in the order of the image, one byte for
    each weight, two's complement *)
val rom_bits : t -> Hardcaml.Bits.t array

(** the base of each tensor inside the ROM, in the shape of the image — the address the
    circuit adds its own offsets to *)
val rom_bases : t -> int Rom_data.t

(** [transpose ~rows ~cols v] exchanges the two axes of a flat row-major tensor.

    THE IMAGE STORES W_in TRANSPOSED, and the reason is the address. The circuit reaches a
    weight by CONCATENATING the two walk counters, which costs nothing and is the
    row-major address only when the dimension under the outer counter is a power of two.
    [d] is one; the projection — [2 d_in + 2 N + H], 292 at the baseline — is not. Storing
    the tensor the other way round puts [d] under the outer counter and the concatenation
    is right again. The quantizer above the seam writes it so; this is the same rule for
    the drawn model of a test, and the circuit reads what both wrote. *)
val transpose : rows:int -> cols:int -> float array -> float array

module For_test : sig
  (** the shape numbers of a drawn model, without the tensors that carry them *)
  type shape =
    { d : int
    ; d_in : int
    ; heads : int
    ; state : int
    ; taps : int
    ; plan : Kind.t array
    ; span : int
    ; ring : int
    }

  (** the shape of a test model: small enough to run in a simulation, and the WHOLE PLAN
      of the era. A plan of blocks alone would elaborate no ring and no head, and the
      faults this era's gates found are address faults that only a second layer of a kind
      can show. *)
  val shape : shape

  (** the shape the ear elected, [docs/mamba.md]: six blocks, the Zamba head and the
      feed-forward, at span 4 and the elected ring of 256. The cost model of a real step
      is read at this shape, and no gate elaborates it. *)
  val elected : shape

  (** [drawn shape ~seed] is a model of drawn weights in [shape], quantized under the rule
      of the era: the elaboration of a circuit takes one, thus a test reads no checkpoint
      and no file that git ignores. The weights are not the weights the trainer draws from
      the same number — only a trained checkpoint crosses that seam.

      IT IS THE ONE PLACE THIS LIBRARY STILL QUANTIZES, and it does so through
      [Mgen_nn.Quantized] and two rules of its own — the decay's scale and the Q12 port
      clamp — which the quantizer above the seam states for every checkpoint and which the
      netlist gate of the era holds it to. The draw is what every expect test of this
      library has recorded, thus it may not move: a moved byte here would move every state
      table, every cycle bench and every frame, and none of those movements would name its
      cause. *)
  val drawn : shape -> seed:int -> t
end
