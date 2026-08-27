(** The draw: one cell's logits and one uniform into one class.

    The last stage of a pass. The head states the logits of a cell as a column — the
    [classes] rows of one seat at one step ARE its [classes] logits — and this unit takes
    era four's pipeline over them: the peak, the tempered exponentials, their total, and
    the pick that a 24-bit uniform lands.

    It is [Mgen_nn.Quantized.draw] over [Quantized.draw_cell], operation for operation:

    {v
      peak      = the largest of the logits
      nn c      = -((logits c - peak) lsl (12 - activation_q) scaled by the temper)
      weight c  = exp2 (nn c)                       Q15, the shared table of the eras
      total     = the sum of the weights
      threshold = (uniform * total) asr 24
      drawn     = the first class whose running total passes the threshold
    v}

    **THE PICK ALWAYS LANDS, THUS THERE IS NO LAST-CLASS ARM.** The peak's own difference
    is zero and exp2 of zero is the whole of Q15, thus the total is 32768 or more; and the
    threshold is [(uniform * total) asr 24] with the uniform under 2^24, thus it stands
    STRICTLY under the total. The unit takes the total and the running totals in one order
    over the same integers, thus the last running total IS the total and a class passes at
    or before the last one. [Mgen_nn.Policy.pick] holds the same argument for the software
    side, and the reference round already deleted the arm that never ran from the JAX
    sampler. A default here would be a branch no walk reaches, and its gate is the
    top-of-the-grid uniform that reaches the last class THROUGH the totals.

    **THE MAGNITUDE SATURATES BEFORE THE TABLE, AND THE SATURATION IS EXACT.** A
    difference of two int16 logits reaches 65535, and the temper of temperature 1.0
    carries it past 23 bits where [Exp2] reads 22. Saturating there states no wrong
    weight: [Exp2] already gives zero for any magnitude of 16 or above, and every value
    the saturation touches stands far past 16.

    **THE UNIFORM ARRIVES FROM THE CALLER AND THE UNIT HOLDS NO GENERATOR.** THE
    CONSUMPTION ORDER IS THE CONTRACT of [docs/diffusion_rtl.md]: one generator serves the
    opening, every mask and every draw of a walk, in one order. A generator here would be
    a second stream, thus the walk takes the three steps of the uniform itself and hands
    the 24 bits over.

    **THE PEAK IS TAKEN HERE AND NOT HANDED IN.** The head's drain could track it for
    nothing while the columns assemble, and that is one walk of [classes] cycles saved.
    What it would cost is a precondition this unit cannot check: a caller that hands in a
    peak that is not the peak states a different distribution and nothing says so. The
    unit takes its own.

    **THE TABLE IS ERA SIX'S FORK OF [Exp2], ONE MAGNITUDE A CYCLE.** The shared unit
    registers its entry but takes the shift from its magnitude as it stands, thus it asks
    a caller to hold that magnitude for two cycles — which a walk of 48 classes would pay
    twice over. [Exp2] here registers the shift beside the entry, thus the walks step a
    class a cycle and a draw is [classes] + ([classes] + 1) + 1 + ([classes] + 1) cycles.
    A table walk takes one cycle more than its classes because the weight of the last
    class stands one cycle behind its magnitude.

    What a caller must know:

    - **[logits] must stand still for the whole draw**, from [start] to the fall of
      [busy]. The unit walks it three times and holds no copy.
    - **[uniform] must stand still as well.** It is read once, after the total, but the
      unit states no cycle for that.
    - **A draw is [busy_cycles] cycles, always.** The pick walks every class even after it
      has found one: the walk waits on [busy] and cannot spend the difference, thus the
      cost model states one number instead of a bound.
    - **The temper is baked.** It is a constant of the model and the bitstream carries it,
      as it carries the weights, thus it enters at elaboration and never at run time.
    - **[start] while [busy] is ignored, structurally.** The machine reads [start] in its
      idle state alone, thus the case cannot arise from a wire and not only from a rule.
      The walk states one draw at a time and waits.
    - There is no clear on the datapath: the caller waits, thus a stale value reaches
      nothing. The counters and [busy] clear. *)

open Hardcaml

(** The shape one instantiation is built for. The classes of a cell are the rows of a
    logit column — the vocabulary's 48 on the board — and the unit draws one of them. *)
module type Shape = sig
  val classes : int
end

module Make (Shape : Shape) : sig
  (** [busy] reads 1 in the cycle after [start] and reads 0 again this many cycles later:
      one draw, always. *)
  val busy_cycles : int

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a (** a strobe: draw the cell that [logits] and [uniform] state *)
      ; logits : 'a
      (** the cell's logits in the Q of the twin: [classes] int16, class 0 in the low
          bits. It is the head's output column, whole. *)
      ; uniform : 'a
      (** the 24-bit uniform of this draw, three steps of the walk's generator *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { busy : 'a
      (** 1 from the cycle behind [start] until the draw stands. [drawn] means nothing
          while it is 1, and is whole in the cycle it falls — the rule [Isqrt] states for
          its root. *)
      ; drawn : 'a (** the class the walk drew *)
      }
    [@@deriving hardcaml]
  end

  (** [create ~temper i] is the block. [temper] is the model's sampling scale,
      [log2e / T], thus one instantiation draws at the temperature the checkpoint was
      quantized for and no wire carries it. *)
  val create : temper:Mgen_nn.Quantized.Constants.scale -> Signal.t I.t -> Signal.t O.t
end
