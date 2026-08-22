(** The integer rules both eras share: the fixed-point formats, the tables, the scalar
    rules of the engines, the quantization of a checkpoint and the integer draw.

    Each era's [Quantized] module is the integer twin of its float model and the reference
    its circuit must equal operation for operation. What stands HERE is the part of that
    arithmetic that is one thing across the eras: a rule written here is read by two
    references and two circuits, thus a change of it changes all four at one time — which
    is the point. The era modules keep what is theirs alone: the parameter structures, the
    state formats of the recurrence, and the engines themselves. *)

(** The shared fixed-point formats, the tables and the constants that cross between the
    references and the circuits. A Q number holds [value * 2^-q]. *)
module Constants : sig
  (** the residual stream: Q16 in int32 *)
  val h_q : int

  (** the normed vector, and the score of attention: Q12 in int16 *)
  val y_q : int

  (** the feed-forward hidden vector after its ReLU: Q10 in int16 *)
  val hid_q : int

  (** the rms epsilon of the float models, in the Q of the squared stream *)
  val eps_q : int

  (** A fixed-point multiplier: the value stands for [q_value * 2^-q]. The Q travels with
      the value because the two are one fact — a multiply that takes the wrong shift is
      silently wrong, and both the references and the circuits apply these scales. *)
  type scale =
    { q_value : int
    ; q : int
    }

  (** [apply s v] scales [v] by [s], toward negative infinity — an arithmetic shift, as
      the circuits'. *)
  val apply : scale -> int -> int

  (** log2(e): the exp2 form of an exponential *)
  val log2e : scale

  (** exp2 of -j/256 in Q15: one table serves the softmax and the sampler of era four and
      the decay of era five *)
  val exp2_table : int array

  (** the sigmoid of a Q12 value in Q15, 256 buckets at their centres; the halves sum to
      2^15, thus sigmoid(-v) = 1 - sigmoid(v) survives the quantization *)
  val sigmoid_table : int array

  (** the correction term of the softplus, ln(1 + exp(-|v|)), in Q12 over a Q12 magnitude:
      softplus(v) = relu(v) + this *)
  val softplus_table : int array

  (** the same three tables as circuit ROMs: 256 entries of 16 bits each *)
  val exp2_bits : Hardcaml.Bits.t array

  val sigmoid_bits : Hardcaml.Bits.t array
  val softplus_bits : Hardcaml.Bits.t array

  (** [sigmoid_index v] is the row of a signed Q12 int16: the top eight bits with the sign
      flipped, which is no arithmetic at all in a circuit *)
  val sigmoid_index : int -> int

  (** [softplus_index v] is the magnitude shifted; the clamp catches the one value -32768
      whose magnitude does not fit the table *)
  val softplus_index : int -> int

  (** [score_shift ~row_q ~head_d] carries a score walk's sum from Q(2 [row_q]) to Q[y_q]
      and applies the 1/sqrt([head_d]) of the references in the same shift, thus [head_d]
      is a power of four. [row_q] is the Q of the scored rows; each era names its own ring
      format and passes it here. *)
  val score_shift : row_q:int -> head_d:int -> int

  (** [slope_exponent ~span ~heads ~head] is the ALiBi exponent of one head: its slope is
      2^-(this), thus the penalty of an age is a shift and never a multiply. *)
  val slope_exponent : span:int -> heads:int -> head:int -> int
end

(** A vector of an integer model, and the two measures that compare one against the float
    vector of the same place. *)
module Tensor : sig
  type t = int array
  type floats = float array

  (** [same_peak q f] is true when the two vectors elect the same index — the top-1
      agreement of the drift reports. A tie keeps the first index on both sides. *)
  val same_peak : t -> floats -> bool

  (** [cosine q f] is the cosine between the two vectors. *)
  val cosine : t -> floats -> float
end

(** [clamp16 v] clamps to int16; [clamps16 v] is true where it would clamp — the detector
    of the clamp counters. *)
val clamp16 : int -> int

val clamps16 : int -> bool

(** the reductions of the engines: [sum n f] is the MAC — the sum of [f i] over
    [0 .. n - 1] — and [max_over n f] is the peak scan *)
val sum : int -> (int -> int) -> int

val max_over : int -> (int -> int) -> int

(** floor of the square root: the one answer the [Isqrt] unit must also give *)
val isqrt : int -> int

(** [exp2_of_magnitude m] is 2^-m of a nonnegative Q12 magnitude, in Q15: the integer part
    shifts, the top eight bits of the fraction index the table, and the peak — a magnitude
    of 0 — is 2^15. [exp2_q u] is the same rule over a Q12 value that is 0 or less: era
    four exponentiates a nonpositive score and era five a decay that is a magnitude by
    construction, thus the negation stands at the caller there and not at all here. One
    definition holds the two readings to one table, and to the [Exp2] unit. *)
val exp2_of_magnitude : int -> int

val exp2_q : int -> int

(** [sigmoid_q v] is the sigmoid of a Q12 value in Q15 — the rule of the [Sigmoid] unit;
    [silu v] is [v] times it, shifted back to Q12 and clamped *)
val sigmoid_q : int -> int

val silu : int -> int

(** [softplus v] is the ramp plus the correction the table holds, clamped to int16 on both
    sides — the rule of the [Softplus] unit *)
val softplus : int -> int

(** one quantized weight tensor: the int8 values, flat in the row-major order of the float
    checkpoint, and the exponent [e] that reads them — the value of an element is
    [q * 2^-e] *)
type quantized =
  { q : Tensor.t
  ; e : int
  }

(** [quantize ?e floats] is the int8 form under the rule of the eras: the largest exponent
    that keeps the rounded peak at 127 or less, or [e] where tensors share one because
    their rows add. *)
val quantize : ?e:int -> Tensor.floats -> quantized

(** [max_abs floats] and [max_exponent v]: the two steps of the exponent rule, exposed
    because a caller that shares an exponent across tensors takes the max over both. *)
val max_abs : Tensor.floats -> float

val max_exponent : float -> int

(** [rom_bits tensors] is the ROM image of a circuit: every tensor in the checkpoint
    order, one byte for each weight, two's complement. *)
val rom_bits : quantized list -> Hardcaml.Bits.t array

(** [policy ~temperature ~min_p] is the sampling policy in the integer forms of the
    machines: the temper — log2(e) / T, its Q one below [Constants.log2e]'s for headroom
    on an 18-bit port — and the min-p share of the peak weight 2^15. It checks the bounds
    through [Policy.check_policy]. *)
val policy : temperature:float -> min_p:float -> Constants.scale * int

(** [draw ~weights prng] is the integer pick of the engines: a 24-bit uniform from three
    bytes of the generator, a threshold of [u * total] over 2^24, and the class whose
    running total passes it. The threshold is below the total, thus the walk always lands
    on a class that holds weight. It gives the uniform back as a float, thus a drift
    report hands the very same number to [Policy.draw_class]. *)
val draw : weights:Tensor.t -> Prng.state -> Prng.state * float * int
