(** The int8 checkpoint as data: the model the bitstream carries, and the formats it is
    written in.

    THE ARITHMETIC OF THE TWIN IS NOT HERE. It is [jax/diffusion/quantized.py], beside the
    float model it quantizes: one framework holds both, the drift between them is measured
    in one place, and the seed sweep runs on the arithmetic the board plays. What crosses
    the seam is one CONTRACT FILE, and this module is its reader. The order that cut the
    OCaml twin is [docs/diffusion_ocaml_cut.md], and the gate that replaced it is
    [jax/tests/test_rtl.py]: Python states what the circuit must do, and
    [bin/gate_diffusion.ml] states what it did.

    The formats, and where each rule comes from, are [docs/diffusion_rtl.md]:

    - Weights are int8 under the exponent rule of the eras, [Mgen_nn.Quantized.quantize].
    - Activations are Q[activation_q] in int16, clamped and counted — Q6, AND THE NUMBER
      IS MEASURED, not chosen: the trunk is a residual stack with no norm on the stream,
      thus a trained model's activations grow with depth, and the golden candidate peaks
      at 184 on half-masked corpus canvases and at 313 on the seeded openings the walk
      really visits. Q6 holds 512 with a 1.6 margin. The input planes enter exact — a cell
      is 0 or one.
    - The accumulator is int32 and is exact up to 57 input channels — 9 C products of int8
      by int16 reach under 2^31 there and one channel more can pass it — thus the sum is
      exact and the order of the taps cannot matter. [Model.check_shape] refuses a wider
      layer, thus the bound is a rule and not a comment; the elected shapes stand far
      under it.
    - The norm folds at quantization, above the seam:
      [gain = scale * rsqrt (variance + eps)] becomes a per-channel multiplier that also
      retires the weight exponent, and [bias = shift - mean * gain] becomes
      Q[activation_q] in int16. The file carries the result and never the population.
    - The draw is era four's pipeline, and [Draw] holds it: the logit differences shift up
      to the Q12 the exp2 unit reads, temper against the peak under [log2e / T], exp2 over
      the shared table gives Q15 weights, and [Mgen_nn.Quantized.draw] picks with a 24-bit
      uniform. The temper travels with the model, because the bitstream carries it. *)

(** the Q of the activation format: a value holds [v * 2^-activation_q] in int16. The
    circuit of the next round reads it here. *)
val activation_q : int

(** the width of the activation format: 16. It is the other half of [activation_q]'s
    sentence, thus it stands beside it and every unit of the circuit slices on this one
    value rather than on a 16 of its own. *)
val activation_bits : int

(** the width of the accumulator one dwell sums into: 32. [Model.widest_inputs] is the
    promise this width keeps, and the column array is sized on it. *)
val accumulator_bits : int

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

  (** THE WIDEST LAYER THE INT32 ACCUMULATOR IS EXACT FOR: 9 C products of int8 by int16
      reach under 2^31 at this many input channels and one channel more can pass it.
      [check_shape] refuses a wider layer, thus the bound is a rule and not a comment. The
      circuit reads it here — its accumulator is sized on this promise, and the gate of
      its array drives the promise to its edge. *)
  val widest_inputs : int

  (** [check_shape t] raises when the model breaks a rule its consumers assume: the layers
      chain input to output, no layer reads more channels than the int32 accumulator is
      exact for, the stem reads the planes and the head states the voices, every kernel
      holds its count, and every constant row holds one entry for each output channel. The
      record is open, thus a model no constructor here made can break a rule; the
      elaboration of the next round calls this where a bad shape must fail loudly. *)
  val check_shape : t -> unit

  (** [of_int8_checkpoint path] is the model of one CONTRACT FILE — the quantized model
      that [jax/diffusion/quantized.py] writes, and the only thing that crosses the seam
      for a build. The quantization happens above the seam, one time, thus this reader
      folds nothing: it takes the kernels, the two per-channel rows and the temper as they
      stand, and [check_shape] holds every rule the consumers assume.

      The layout is that module's docstring. Two of its facts are facts of THIS reader:
      every tensor is int32, because [Nx_io] skips every dtype it does not hold and int8
      is one of them; and the temper and the Q travel as the named tensors ["temper"] and
      ["activation_q"], because [Nx_io] gives no access to [__metadata__]. A file
      quantized at another Q refuses here and not in the middle of a walk.

      It raises [Invalid_argument] when the tensor count does not divide into layers, when
      a tensor is missing, or when a shape or a rule does not hold; the message names the
      tensor that refused. *)
  val of_int8_checkpoint : string -> t

  (** the ROM image of the circuit: every kernel in checkpoint order, one byte for each
      weight, two's complement. The gains and the biases are not in it — they are
      per-channel constants of the elaboration, as era five's per-head constants were. *)
  val rom_bits : t -> Hardcaml.Bits.t array

  (** the byte base of each kernel inside the image, in layer order *)
  val rom_bases : t -> int array

  module For_test : sig
    (** the shape of a test model: small enough for an expect test, the same structure as
        the era's *)
    val config : Diffusion.Config.t

    (** [drawn config ~seed] is a model of the shape [config] states, its values drawn
        under [Prng] STRAIGHT INTO THE FORMATS: the kernel bytes at one fixed exponent, a
        per-channel gain in int16 whose shift fits the six bits of the norm word, a bias
        in int16, and the temper of temperature 1.0. It passes [check_shape], thus a test
        reads no file and quantizes nothing.

        THE DRAWN TRUNK HOLDS O(1) ACTIVATIONS, and that is what makes the model worth
        drawing: a gain drawn flat inside int16 would clamp every write of the trunk or
        zero it, and the pictures, the frames and the cycle counts of the tests that read
        this model would all read a machine that no checkpoint makes. The implementation
        states the arithmetic that holds it.

        The tests that read it need a model OF A SHAPE — cycle counts, tile counts, the
        turn order, the images' self-consistency, a picture — and none of them needs the
        twin's arithmetic: the two gates that did are [jax/tests/test_rtl.py]'s. *)
    val drawn : Diffusion.Config.t -> seed:int -> t

    (** the kernels of the image, in its order; the gates read them beside [rom_bases] *)
    val rom_tensors : t -> quantized list
  end
end
