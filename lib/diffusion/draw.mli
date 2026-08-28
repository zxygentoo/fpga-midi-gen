(** The draw: one cell's logits and one uniform into one class.

    The last stage of a pass. The head states a cell's logits as a column — the [classes]
    rows of one seat at one step ARE its logits — and this unit takes era four's pipeline
    over them: the peak, the tempered exponentials, their total, and the pick a 24-bit
    uniform lands. It is [Mgen_nn.Quantized.draw] over [draw_cell], operation for
    operation:

    {v
      peak      = the largest of the logits
      nn c      = -((logits c - peak) lsl (12 - activation_q) scaled by the temper)
      weight c  = exp2 (nn c)                       Q15, the shared table of the eras
      total     = the sum of the weights
      threshold = (uniform * total) asr 24
      drawn     = the first class whose running total passes the threshold
    v}

    **THE PICK ALWAYS LANDS, THUS THERE IS NO LAST-CLASS ARM.** The peak's own difference
    is zero and exp2 of zero is the whole of Q15, thus the total is 32768 or more; the
    threshold is [(uniform * total) asr 24] with the uniform under 2^24, thus it stands
    STRICTLY under it. The unit takes the total and the running totals in one order over
    the same integers, thus a class always passes. A default arm would be a branch no walk
    reaches; its gate is the top-of-the-grid uniform, which reaches the last class THROUGH
    the totals.

    **THE MAGNITUDE SATURATES BEFORE THE TABLE, AND THE SATURATION IS EXACT.** A
    difference of two int16 logits reaches 65535, which the temper carries past the 22
    bits [Exp2] reads. Saturating states no wrong weight: [Exp2] gives zero for any
    magnitude of 16 or above, and every value the saturation touches stands far past 16.

    **THE UNIT HOLDS NO GENERATOR.** One generator serves the opening, every mask and
    every draw of a walk, in one order — [docs/diffusion_rtl.md] — thus the walk takes the
    three steps itself and hands the 24 bits over.

    **THE PEAK IS TAKEN HERE AND NOT HANDED IN.** The head's drain could track it for
    nothing, but a caller that handed in a peak that is not the peak would state a
    different distribution and nothing would say so. The unit takes its own.

    A CLASS'S WEIGHT STANDS FOUR CYCLES BEHIND THE CYCLE THAT NAMED IT — the walk
    register, the temper register and [Exp2]'s two — thus [busy_cycles] is the peak walk
    of [classes] + 1, the weights of [classes] + 4, two cycles for the threshold, and the
    pick of [classes] + 4: 155 at the era's 48 classes.

    What a caller must know:

    - **[logits] and [uniform] must stand still for the whole draw**, from [start] to the
      fall of [busy]. The unit walks the logits three times and holds no copy.
    - **A draw is [busy_cycles] cycles, always.** The pick walks every class even after it
      has found one, thus the cost model states one number instead of a bound.
    - **The temper is baked.** It is a constant of the model, thus it enters at
      elaboration and never at run time.
    - **[start] while [busy] is ignored, structurally.** The machine reads [start] in its
      idle state alone.
    - There is no clear on the datapath: the caller waits, thus a stale value reaches
      nothing. The counters and [busy] clear. *)

open Hardcaml

(** the classes of a cell: the rows of a logit column, the vocabulary's 48 on the board *)
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
      (** the cell's logits in the Q of the twin: [classes] int16, class 0 in the low bits
          — the head's output column, whole *)
      ; uniform : 'a
      (** the 24-bit uniform of this draw, three steps of the walk's generator *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { busy : 'a
      (** 1 from the cycle behind [start] until the draw stands. [drawn] means nothing
          while it is 1, and is whole in the cycle it falls. *)
      ; drawn : 'a (** the class the walk drew *)
      }
    [@@deriving hardcaml]
  end

  (** [create ~temper i] is the block. [temper] is the model's sampling scale,
      [log2e / T], thus one instantiation draws at the temperature the checkpoint was
      quantized for. *)
  val create : temper:Mgen_nn.Quantized.Constants.scale -> Signal.t I.t -> Signal.t O.t
end
