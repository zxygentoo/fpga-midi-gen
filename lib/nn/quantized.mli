(** The integer rules the eras share: the fixed-point formats, the tables, the scalar
    rules of the arithmetic, the ROM image of a model and the integer draw.

    THE INTEGER TWINS LIVE ABOVE THE SEAM, in [jax/quantized.py] and each era's
    [jax/<era>/quantized.py], and they are what the circuits must equal operation for
    operation. NOTHING HERE QUANTIZES: what stands here is the half of that arithmetic the
    CIRCUITS read — the formats, the tables and the scalar oracles that [Rtl] and each
    era's [Source] elaborate and that the unit gates hold their circuits against, the ROM
    image of a model that is already int8, and the integer draw. A rule written here is
    read by three circuits, thus a change of it changes all three at one time, and
    [jax/quantized.py] states the same rule for the twins with a gate on each shared
    table. *)

(** The shared fixed-point formats, the tables and the constants that cross between the
    twins and the circuits. A Q number holds [value * 2^-q]. *)
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
      silently wrong, and both the twins and the circuits apply these scales. *)
  type scale =
    { q_value : int
    ; q : int
    }

  (** [apply s v] scales [v] by [s], toward negative infinity — an arithmetic shift, as
      the circuits'. *)
  val apply : scale -> int -> int

  (** log2(e): the exp2 form of an exponential *)
  val log2e : scale

  (** the temper at temperature 1: log2(e) at the temper's own Q, one below [log2e]'s.

      The temper is log2(e) / T, and the spare bit is headroom for the temperature: the
      circuits multiply by this constant on an 18-bit signed port, thus [log2e]'s own Q
      would overflow that port under a temperature of about 0.36, and this Q holds down to
      about 0.18. [jax/quantized.py]'s [temper_of] states the rule for every temperature.

      A model of a CONTRACT FILE reads its own temper from the file. A DRAWN model has no
      training run behind it, thus it states this one. *)
  val temper_at_one : scale

  (** The three tables of the units as circuit ROMs, 256 entries of 16 bits each: exp2 of
      -j/256 in Q15, which serves the softmax and the sampler of era four and the decay of
      era five; the sigmoid of a Q12 value in Q15, 256 buckets at their centres, whose
      halves sum to 2^15 so that sigmoid(-v) = 1 - sigmoid(v) survives the quantization;
      and the correction term of the softplus, ln(1 + exp(-|v|)), in Q12 over a Q12
      magnitude, where softplus(v) = relu(v) + this.

      The tables THEMSELVES are the software's own — [sigmoid_q] and [softplus] read them
      here — thus only the ROM images cross the interface. *)
  val exp2_bits : Hardcaml.Bits.t array

  val sigmoid_bits : Hardcaml.Bits.t array
  val softplus_bits : Hardcaml.Bits.t array

  (** [softplus_index v] is the magnitude shifted; the clamp catches the one value -32768
      whose magnitude does not fit the table *)
  val softplus_index : int -> int

  (** [score_shift ~row_q ~head_d] carries a score walk's sum from Q(2 [row_q]) to Q[y_q]
      and applies the 1/sqrt([head_d]) of the twins in the same shift, thus [head_d] is a
      power of four. [row_q] is the Q of the scored rows; each era names its own ring
      format and passes it here. *)
  val score_shift : row_q:int -> head_d:int -> int

  (** [slope_exponent ~span ~heads ~head] is the ALiBi exponent of one head: its slope is
      2^-(this), thus the penalty of an age is a shift and never a multiply. *)
  val slope_exponent : span:int -> heads:int -> head:int -> int
end

(** A vector of a model as this side reads it: the integers a quantizer above the seam
    already stated. The float form and every measure that compares the two — the
    quantization itself, the top-1 agreement, the cosine of the drift reports — stand with
    the twins, in [jax/quantized.py] and [jax/<era>/quantized.py]. *)
module Tensor : sig
  type t = int array
end

(** [clamp16 v] clamps to the rails of int16 and never wraps. All three eras clamp through
    this rule and through [Rtl.clamp16], thus none of them writes a rail of its own and
    parts from the twin in silence. The rails themselves are not exported: a caller that
    wants them wants one of the two clamps. The detector that COUNTS a clamp lives above
    the seam with the twins that tally them. *)
val clamp16 : int -> int

(** The rules of this module as circuits, where a circuit needs the same rule.

    A value here and its software half above are TWO STATEMENTS of one rule — one over
    [int] and one over [Signal] — and nothing in the types welds them. What they share is
    the rails, and what holds them together is the expect test beside each one: it drives
    the circuit at the rails and past them and states what the software says. Read the
    pair as one rule, and change neither half without the other. *)
module Rtl : sig
  (** [clamp16 wide] saturates [wide] into an int16 and never wraps. A wrap would be
      silently wrong music, and the clamp is what the format election stands on.

      THE COMPARE STANDS AT [wide]'S OWN WIDTH, whatever that is: an [sresize ~width:32]
      before the compare truncates a 48-bit product and hands the clamp a value that has
      already wrapped, so there is nothing left for it to catch. Era six's epilogue clamps
      a 48-bit gain product, and the frozen eras alias this one rather than resize to 32
      first; the expect test beside it measures what that adoption fixed. *)
  val clamp16 : Hardcaml.Signal.t -> Hardcaml.Signal.t
end

(** one quantized weight tensor: the int8 values, flat in the row-major order of the float
    checkpoint, and the exponent [e] that reads them — the value of an element is
    [q * 2^-e] *)
type quantized =
  { q : Tensor.t
  ; e : int
  }

(** [rom_bits tensors] is the ROM image of a circuit: every tensor in the checkpoint
    order, one byte for each weight, two's complement. *)
val rom_bits : quantized list -> Hardcaml.Bits.t array

(** [bases_of sizes] is where each of a run of sizes opens: the exclusive prefix scan,
    [| 0; sizes.(0); sizes.(0) + sizes.(1); ... |]. Every era's [rom_bases] is one reading
    of it and the elaborations' banks are the others, thus the rule stands in one place. *)
val bases_of : int array -> int array

(** [draw ~weights prng] is the integer pick of the engines: a 24-bit uniform from three
    bytes of the generator, a threshold of [u * total] over 2^24, and the class whose
    running total passes it. The threshold is below the total, thus the walk always lands
    on a class that holds weight. It gives the uniform back as a float, thus a caller can
    hand the very same number to a float draw. *)
val draw : weights:Tensor.t -> Prng.state -> Prng.state * float * int

(** The scalar rules no engine reads any more. The twins moved above the seam this round,
    thus what is left of this arithmetic in OCaml is an ORACLE: each value states in one
    line what a unit circuit must give, and the expect test beside that circuit drives it
    at the rails and against this. Nothing in the elaboration calls them. *)
module For_test : sig
  (** the MAC as a reduction: the sum of [f i] over [0 .. n - 1] *)
  val sum : int -> (int -> int) -> int

  (** floor of the square root: the one answer the [Isqrt] unit must also give *)
  val isqrt : int -> int

  (** the sigmoid of a Q12 value in Q15 — the rule of the [Sigmoid] unit *)
  val sigmoid_q : int -> int

  (** [exp2_q u] is 2^u of a Q12 value that is 0 or less, in Q15 — the rule of the [Exp2]
      unit: the integer part shifts, the top eight bits of the fraction index the table,
      and the peak — a [u] of 0 — is 2^15. Era four exponentiates a nonpositive score and
      era five a decay that is a magnitude by construction, thus the negation stands at
      the caller there, and one definition holds both readings to one table. *)
  val exp2_q : int -> int

  (** [v] held inside the int8 the ROM carries: a value past a rail saturates. Era six
      draws straight into byte units and reads this alone. *)
  val clamp_byte : int -> int

  (** [drawn_tensor ~e values] is a DRAWN tensor at a STATED exponent — the floats scaled
      by 2^[e], rounded, and clamped into the byte.

      A quantizer picks an exponent from a tensor's own peak; that is a rule of a
      CHECKPOINT and it lives above the seam. A drawn model has no checkpoint behind it,
      thus one stated exponent covers every tensor and the tables that must share an
      exponent then do. The implementation carries the measurement behind the frozen eras'
      10 and behind era six's 14. These make TEST models: the walks they make are what the
      cycle benches record, thus the seeds, the scales and the exponents may not move. *)
  val drawn_tensor : e:int -> float array -> quantized
end
