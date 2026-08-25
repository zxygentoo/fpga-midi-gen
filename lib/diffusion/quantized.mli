(** The integer twin of the masked canvas: the reference the circuit of the next round
    must equal.

    The float model of [Diffusion] is what the trainer produced. This module is the same
    model in the arithmetic the board can hold — int8 weights with a power-of-two exponent
    for each tensor, int16 activations, the norm folded into per-channel constants, and
    the draw in integers — and the circuit must equal it operation for operation, not
    approximately. Nothing here approximates on purpose: every shift, every floor and
    every table is a rule the RTL will read from this module rather than restate.

    The formats, and where each rule comes from, are [docs/diffusion_rtl.md]:

    - Weights are int8 under the exponent rule of the eras, [Mgen_nn.Quantized.quantize].
    - Activations are Q[activation_q] in int16, clamped and counted — Q6, AND THE NUMBER
      IS MEASURED, not chosen: the trunk is a residual stack with no norm on the stream,
      thus a trained model's activations grow with depth, and the golden candidate peaks
      at 184 on half-masked corpus canvases and at 313 on the seeded openings the walk
      really visits. Q6 holds 512 with a 1.6 margin. The input planes enter exact — a cell
      is 0 or one.
    - The accumulator is int32 and cannot overflow below 58 input channels — 9 C products
      of int8 by int16 stand under 2^31 — thus the sum is exact and the order of the taps
      cannot matter. [Model.check_shape] refuses a wider layer, thus the bound is a rule
      and not a comment; the elected shapes stand far under it.
    - The norm folds at quantization: [gain = scale * rsqrt (variance + eps)] becomes a
      per-channel multiplier that also retires the weight exponent, and
      [bias = shift - mean * gain] becomes Q[activation_q] in int16. Then ReLU; the head
      keeps no ReLU, thus the logits carry the activation format.
    - The draw is era four's pipeline: the logit differences shift up to the Q12 the exp2
      unit reads — exact, a left shift — then temper against the peak under [log2e / T],
      exp2 over the shared table gives Q15 weights, and [Mgen_nn.Quantized.draw] picks
      with a 24-bit uniform.
    - The masks and the opening are the integer rules of the walk already, and the twin
      consumes the same uniforms in the same places as [Diffusion.gibbs].

    What the quantization costs is a measurement and not a promise: [Drift] states it, on
    the walk the board really takes. *)

(** the Q of the activation format: a value holds [v * 2^-activation_q] in int16. The
    circuit of the next round reads it here. *)
val activation_q : int

(** The weights in the form the bitstream will carry, and the policy folded with them. *)
module Model : sig
  (** one quantized tensor: the int8 values flat in the row-major order of the float
      checkpoint, and the exponent that reads them — [Mgen_nn.Quantized.quantized] *)
  type quantized = Mgen_nn.Quantized.quantized

  (** One layer as the machine holds it. The five float tensors become three facts: the
      kernel, and the two per-channel constant rows the norm folded into. The gains are
      [Mgen_nn.Quantized.Constants.scale] values whose shift retires the weight exponent,
      thus the accumulator goes to the activation format in one multiply; the biases are
      int16 in the same format. *)
  type layer =
    { kernel : quantized
    ; gain : Mgen_nn.Quantized.Constants.scale array
    ; bias : int array
    ; inputs : int
    (** the input channels: the flat kernel reads as [3; 3; inputs; outputs] *)
    ; outputs : int
    }

  type t =
    { layers : layer array
    ; temper : Mgen_nn.Quantized.Constants.scale
    (** the sampling temper, [log2e / T]: part of the model, because the bitstream will
        carry it *)
    }

  (** [check_shape t] raises when the model breaks a rule its consumers assume: the layers
      chain input to output, no layer reads more channels than the int32 accumulator is
      exact for, the stem reads the planes and the head states the voices, every kernel
      holds its count, and every constant row holds one entry for each output channel. The
      record is open, thus a model no constructor here made can break a rule; the
      elaboration of the next round calls this where a bad shape must fail loudly. *)
  val check_shape : t -> unit

  (** [of_params ?temperature params] quantizes the float model: the kernels under the
      exponent rule, the norms folded. This is the one quantization of the era — the drift
      walk, the tools and the elaboration all take their model here, thus the pair under
      comparison cannot slip. *)
  val of_params : ?temperature:float -> Diffusion.Params.t -> t

  (** [of_checkpoint ?temperature config path] is [Diffusion.Params.load] then
      [of_params]: the model of one checkpoint file. *)
  val of_checkpoint : ?temperature:float -> Diffusion.Config.t -> string -> t

  (** the ROM image of the circuit: every kernel in checkpoint order, one byte for each
      weight, two's complement. The gains and the biases are not in it — they are
      per-channel constants of the elaboration, as era five's per-head constants were. *)
  val rom_bits : t -> Hardcaml.Bits.t array

  (** the byte base of each kernel inside the image, in layer order *)
  val rom_bases : t -> int array

  (** the kernels of the image, in its order; the gates read them beside [rom_bases] *)
  val rom_tensors : t -> quantized list

  module For_test : sig
    (** the shape of a test model: small enough for an expect test, the same structure as
        the era's *)
    val config : Diffusion.Config.t

    (** [init config ~seed]: a model of drawn weights under the draw of the era, thus a
        test reads no checkpoint and no file that git ignores *)
    val init : Diffusion.Config.t -> seed:int -> t
  end
end

(** The clamps a walk met, and the chances each one had. The formats are chosen with
    margin and not metered on a trained checkpoint; a clamp that fires is the finding that
    says which format is wrong, thus it is counted and never assumed away. *)
module Clamps : sig
  type t =
    { activations : int (** the activation writes that rode the clamp *)
    ; activations_seen : int
    ; peak : int
    (** THE HOTTEST WRITE OF THE WALK, in activation units and BEFORE the clamp: the
        number the format election stands on. The Q6 election read peaks of 184 and 313
        with a throwaway probe; this counter is what makes that measurement repeatable
        when the checkpoint changes, thus the format question never again waits on a probe
        that no longer exists. *)
    }
end

(** One running walk, as a value: a pass gives the engine after it. *)
module Engine : sig
  type t

  (** one redraw of a pass: the cell, the logits the draw read, the uniform it took and
      the class it chose. The drift report reads all four; the walk itself needs only the
      class. *)
  type draw =
    { step : int
    ; voice : int
    ; logits : int array
    ; uniform : float
    ; drawn : int
    }

  (** what one pass states: the canvas it read, the mask it drew, and the redraws in the
      cell order. [before] is the canvas the forward pass saw — the drift report
      teacher-forces the float model on exactly it. *)
  type pass =
    { before : int array array
    ; hidden : bool array array
    ; draws : draw list
    }

  (** [init model ~steps ~walk ~seed] is the engine at its origin: the seeded opening
      already drawn, under [Prng.create] — the seed as it stands, under the rule of the
      SEED cell. A seed inside 32 bits walks the very stream [Diffusion.gibbs] folds to,
      thus the two openings are one opening and the walks part only where the arithmetic
      parts. *)
  val init : Model.t -> steps:int -> walk:int -> seed:int -> t

  (** [next_pass t] takes one Gibbs pass: the masks, one integer forward, the redraws.
      Past the last pass it raises [Invalid_argument]. *)
  val next_pass : t -> t * pass

  (** [run t] takes every remaining pass and gives the finished canvas *)
  val run : t -> int array array

  (** the clamps the walk has met so far, accumulated *)
  val clamps : t -> Clamps.t
end

(** What the quantization costs, measured on the walk the board takes. *)
module Drift : sig
  type stats =
    { passes : int
    ; cells : int (** the redrawn cells: the comparisons of the report *)
    ; same_peak : int (** the cells where both models elect the same class *)
    ; same_draw : int (** the cells where both models pick the same class *)
    ; mean_cosine : float
    ; activations_clamped : float
    (** the share of activation writes that rode the clamp *)
    ; activation_peak : float
    (** the hottest write of the walk in real units — [Clamps.peak] over the format's one.
        The format holds while it stands under the int16 ceiling, 512.0 at Q6. *)
    }

  (** [walk params ~steps ~walk ~seed] draws the quantized walk and scores the float model
      against it, cell for cell. The engine walks; at every pass the float model is
      teacher-forced on the ENGINE'S canvas and the ENGINE'S mask, thus the two read one
      context and what stands between them is the arithmetic alone. The same-draw share
      reads the float draw on the very uniform the engine took — a difference there is the
      arithmetic and never the generator. *)
  val walk : Diffusion.Params.t -> steps:int -> walk:int -> seed:int -> stats
end
