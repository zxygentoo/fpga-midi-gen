(** The model as the circuit reads it: the roll, the formats, the contract file, the walk
    and the frames.

    Era six draws one sheet of T steps of four voices over the 48-class vocabulary of
    [Vocab], beside a mask over its cells. The model reads the sheet with the masked cells
    hidden and states a distribution over the classes of every cell at once; the Gibbs
    walk hides and restores cells in turn until a piece stands. Nothing is causal and
    nothing is written left to right. The design of the model is [docs/diffusion.md], and
    the design of this layer and its gates is [docs/diffusion_rtl.md].

    **NO MODEL IS COMPUTED HERE.** The float model is [jax/diffusion/model.py] and the
    integer twin is [jax/diffusion/quantized/], its model half and its walk;
    [jax/tests/test_rtl_diffusion.py] states what the circuit must do and
    [bin/gate_diffusion.ml] states what it did. What stays here is what the CIRCUIT reads,
    and every one of those facts is a rule the RTL must equal rather than restate:

    - **The formats**, which every unit of the machine slices on.
    - **The model as data**: the record the contract file carries, and its reader.
    - **The walk**: the registers of the seats, the cell order, the opening, the masks and
      the anneal rule.
    - **The frames**: what a sheet states to the sequencer.

    Two things stay here although they look like contract-file facts, and the walk gate
    holds both:

    - **The anneal table.** [anneal_threshold] depends on N, which is a parameter of the
      ELABORATION and not of the model, thus a table in the file would tie a checkpoint to
      one geometry.
    - **The seat registers.** [seat_openings] is [Jsb.voice_ranges] read through the class
      map, thus the corpus library is the authority. A file written by JAX would make
      [jax/measure.py]'s copy the master.

    **THE CONSUMPTION ORDER IS THE CONTRACT** of [docs/diffusion_rtl.md]: the opening
    draws one uniform for each cell, each pass draws one for each cell (the masks) and
    then one for each hidden cell (the classes), over [cell_order] everywhere. Every cell
    of a sheet is free. One seed names one sheet — in JAX, here and on the board. *)

(** {1 The roll} *)

(** the rows of the roll: the classes of [Vocab], row 0 silence *)
val rows : int

(** the seats of a step: the voices of [Frame] *)
val voices : int

(** the planes the stem reads: one class plane and one mask plane for each seat *)
val planes : int

(** {1 The formats} *)

(** the Q of the activation format: a value holds [v * 2^-activation_q] in int16. Every
    unit of the circuit reads it here. *)
val activation_q : int

(** the width of the activation format: 16. It is the other half of [activation_q]'s
    sentence, thus every unit slices on this and never on a 16 of its own. *)
val activation_bits : int

(** the bottom rail of the activation format, and the opening value of the draw's peak
    scan: no logit stands below it. The rails the CLAMP compares against are
    [Mgen_nn.Quantized.int16_high] and [int16_low], because saturating to int16 is the
    repository's rule and not this era's. *)
val activation_low : int

(** the width of the accumulator one dwell sums into: 32. [widest_inputs] is the promise
    this width keeps, and the column array is sized on it. *)
val accumulator_bits : int

(** THE WIDEST LAYER THE INT32 ACCUMULATOR IS EXACT FOR: 9 C products of int8 by int16
    reach under 2^31 at this many input channels and one channel more can pass it.
    [check_shape] refuses a wider layer, thus the bound is a rule and not a comment. *)
val widest_inputs : int

(** {1 The model} *)

(** one quantized tensor: the int8 values flat in the row-major order of the checkpoint,
    and the exponent that reads them — [Mgen_nn.Quantized.quantized] *)
type quantized = Mgen_nn.Quantized.quantized

(** One layer as the machine holds it: the kernel, and the two per-channel rows the norm
    folded into. A gain's shift retires the weight exponent, thus the accumulator reaches
    the activation format in one multiply; the biases are int16 in that format. *)
type layer =
  { kernel : quantized
  ; gain : Mgen_nn.Quantized.Constants.scale array
  ; bias : int array
  ; inputs : int
  (** the input channels: the flat kernel reads as [3; 3; inputs; outputs] *)
  ; outputs : int
  }

(** the whole model the bitstream carries *)
type t =
  { layers : layer array
  ; temper : Mgen_nn.Quantized.Constants.scale
  (** the sampling temper, [log2e / T]: part of the model, because the bitstream carries
      it *)
  }

(** [check_shape t] raises when the model breaks a rule its consumers assume: the layers
    chain input to output, no layer reads more channels than the int32 accumulator is
    exact for, the stem reads the planes and the head states the voices, every kernel
    holds its count, and every constant row holds one entry for each output channel. The
    record is open, thus a model no constructor here made can break a rule. *)
val check_shape : t -> unit

(** [of_int8_checkpoint path] is the model of one CONTRACT FILE — the quantized model that
    [jax/diffusion/quantized/model.py] writes, and the only thing that crosses the seam
    for a build. The quantization happens above the seam, one time, thus this reader folds
    nothing: it takes the kernels, the two per-channel rows and the temper as they stand,
    and [check_shape] holds every rule the consumers assume.

    The layout is that module's docstring; the two facts of the archive itself — every
    tensor int32, every scalar a named tensor — stand in [Mgen_nn.Contract_file], which
    reads it. A file quantized at another Q refuses here and not inside a walk.

    It raises [Invalid_argument] when the tensor count does not divide into layers, when a
    tensor is missing, or when a shape or a rule does not hold; the message names the
    tensor that refused. *)
val of_int8_checkpoint : string -> t

(** the ROM image of the circuit: every kernel in checkpoint order, one byte for each
    weight, two's complement. The gains and the biases are not in it — they are the
    elaboration's per-channel constants. *)
val rom_bits : t -> Hardcaml.Bits.t array

(** the byte base of each kernel inside the image, in layer order *)
val rom_bases : t -> int array

(** {1 The walk} *)

(** The register of each seat as classes: what [opening_sheet] draws inside. The circuit
    reads these through its elaboration, thus the walk and the board open by one rule. *)
type opening =
  { low : int (** the lowest class of the seat's register *)
  ; width : int (** the classes of the register: [high - low + 1] *)
  }

(** the opening of each seat, seat 0 the bass: [Jsb.voice_ranges] turned around and read
    through the class map of the roll *)
val seat_openings : opening array

(** [anneal_threshold ~step ~walk] is the masking threshold of pass [step] of [walk], on
    the 24-bit grid of the generator: [floor (alpha * 2^24)] of the paper's annealed
    probability [alpha = max (0.1, 0.9 - 0.8 step / (0.7 walk))]. A cell hides exactly
    when its uniform times 2^24 falls under this integer, and it is the entry the
    circuit's table holds. *)
val anneal_threshold : step:int -> walk:int -> int

(** [cell_order ~steps] is the order the walk visits the cells of a sheet in: a step at a
    time, and the seats of a step inside it, as [step, voice] pairs.

    Every uniform of the walk is drawn in this order — the opening, each mask, and the
    draw of each hidden cell — thus the twin takes the same walk by taking the same order. *)
val cell_order : steps:int -> (int * int) list

(** [opening_sheet state ~steps] is the seeded opening of the walk: one uniform for each
    cell in the cell order, the class [low + floor (u * width)] inside the register of the
    cell's own seat. [jax/diffusion/model.py]'s [opening_sheet] is the same rule and the
    same consumption. *)
val opening_sheet : Prng.state -> steps:int -> Prng.state * int array array

(** [hidden_cells state ~steps ~threshold] is the mask of one pass: one uniform for each
    cell in the cell order, hidden exactly when [u * 2^24] falls under [threshold]. The
    same sharing argument as [opening_sheet]. *)
val hidden_cells
  :  Prng.state
  -> steps:int
  -> threshold:int
  -> Prng.state * bool array array

(** {1 The frames} *)

(** [frames_of_sheet sheet] is the frame of each step: the classes of a step become the
    voice codes of one word, seat 0 in the low byte. It is the rule [jax/corpus.py]
    states, thus a walk here and a walk there compare as text. *)
val frames_of_sheet : int array array -> int array

(** {1 The drawn model} *)

module For_test : sig
  (** [drawn ~layers ~width ~seed] is a model of that shape, its values drawn under [Prng]
      STRAIGHT INTO THE FORMATS: the kernel bytes at one fixed exponent, a per-channel
      gain in int16 whose shift fits the six bits of the norm word, a bias in int16, and
      the temper of temperature 1.0. It passes [check_shape], thus a test reads no file
      and quantizes nothing.

      THE DRAWN TRUNK HOLDS O(1) ACTIVATIONS, and that is what makes it worth drawing: a
      gain drawn flat inside int16 would clamp every write of the trunk or zero it, and
      the pictures, frames and cycle counts that read this model would read a machine no
      checkpoint makes.

      Its readers need a model OF A SHAPE and never the twin's arithmetic; the gates that
      need that are [jax/tests/test_rtl_diffusion.py]'s. *)
  val drawn : layers:int -> width:int -> seed:int -> t

  (** the kernels of the image, in its order; the gates read them beside [rom_bases] *)
  val rom_tensors : t -> quantized list

  (** [over_cells state ~steps ~f] draws one uniform for each cell in [cell_order] and
      hands each to [f ~step ~voice]. It is the fold [opening_sheet] and [hidden_cells]
      stand on, exported so that [Sheet.For_test.stem_input] draws by the same rule and
      never a second statement of it. *)
  val over_cells
    :  Prng.state
    -> steps:int
    -> f:(step:int -> voice:int -> float -> unit)
    -> Prng.state
end
