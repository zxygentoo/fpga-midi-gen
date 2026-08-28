(** The model as the circuit reads it: the roll, the formats, the contract file, the walk
    and the frames.

    Era six draws one canvas of T steps of four voices over the 48-class vocabulary of
    [Vocab], beside a mask over its cells. The model reads the canvas with the masked
    cells hidden and states a distribution over the classes of every cell at once; the
    Gibbs walk hides and restores cells in turn until a piece stands. Nothing is causal
    and nothing is written left to right. The design of the model is [docs/diffusion.md],
    and the design of this layer and its gates is [docs/diffusion_rtl.md].

    **NO MODEL IS COMPUTED HERE.** The float model is [jax/diffusion/model.py] and the
    integer twin is [jax/diffusion/quantized.py]; the order that cut them out of OCaml is
    [docs/diffusion_ocaml_cut.md], and the gate that replaced them is
    [jax/tests/test_rtl.py] — Python states what the circuit must do and
    [bin/gate_diffusion.ml] states what it did. What stays here is what the CIRCUIT reads,
    and every one of those facts is a rule the RTL must equal rather than restate:

    - **The formats**, which every unit of the machine slices on.
    - **The model as data**: the record the contract file carries, and its reader.
    - **The walk**: the registers of the seats, the cell order, the opening, the masks and
      the anneal rule.
    - **The frames**: what a canvas states to the sequencer.

    Two things that look like they belong in the contract file do not, and stay here:

    - **The anneal table.** [anneal_threshold] depends on N, and N is a parameter of the
      ELABORATION and not of the model — [gen_verilog] states it, and the gates run N 3,
      4, 8 and 512 over one drawn model. A table in the file would tie a checkpoint to one
      geometry.
    - **The seat registers.** [seat_openings] is [Jsb.voice_ranges] read through the class
      map, thus the corpus library is the authority and its own test pins the ranges. A
      file written by JAX would make [jax/measure.py]'s copy the master.

    The walk gate holds both: a threshold apart from the twin's moves every mask of a
    pass, and the opening's classes are compared cell for cell.

    **THE CONSUMPTION ORDER IS THE CONTRACT** of [docs/diffusion_rtl.md]: the opening
    draws one uniform for each cell, each pass draws one for each cell (the masks) and
    then one for each hidden cell (the classes), and the cells walk in [cell_order]
    everywhere. Every cell of a canvas is free — nothing is given to a walk of this era.
    One seed names one canvas — in JAX, here and on the board — and the same seed gives
    the same piece. *)

(** {1 The roll} *)

(** the rows of the roll: the classes of [Vocab], row 0 silence *)
val rows : int

(** the seats of a step: the voices of [Frame] *)
val voices : int

(** {1 The formats} *)

(** the Q of the activation format: a value holds [v * 2^-activation_q] in int16. Every
    unit of the circuit reads it here. *)
val activation_q : int

(** the width of the activation format: 16. It is the other half of [activation_q]'s
    sentence, thus it stands beside it and every unit of the circuit slices on this one
    value rather than on a 16 of its own. *)
val activation_bits : int

(** the width of the accumulator one dwell sums into: 32. [widest_inputs] is the promise
    this width keeps, and the column array is sized on it. *)
val accumulator_bits : int

(** THE WIDEST LAYER THE INT32 ACCUMULATOR IS EXACT FOR: 9 C products of int8 by int16
    reach under 2^31 at this many input channels and one channel more can pass it.
    [check_shape] refuses a wider layer, thus the bound is a rule and not a comment. The
    circuit reads it here — its accumulator is sized on this promise, and the gate of its
    array drives the promise to its edge. *)
val widest_inputs : int

(** {1 The model} *)

(** one quantized tensor: the int8 values flat in the row-major order of the checkpoint,
    and the exponent that reads them — [Mgen_nn.Quantized.quantized] *)
type quantized = Mgen_nn.Quantized.quantized

(** One layer as the machine holds it. The five float tensors of a checkpoint become three
    facts: the kernel, and the two per-channel constant rows the norm folded into. The
    gains are [Mgen_nn.Quantized.Constants.scale] values whose shift retires the weight
    exponent, thus the accumulator goes to the activation format in one multiply; the
    biases are int16 in the same format. *)
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
    record is open, thus a model no constructor here made can break a rule;
    [Elaboration.create] calls this where a bad shape must fail loudly. *)
val check_shape : t -> unit

(** [of_int8_checkpoint path] is the model of one CONTRACT FILE — the quantized model that
    [jax/diffusion/quantized.py] writes, and the only thing that crosses the seam for a
    build. The quantization happens above the seam, one time, thus this reader folds
    nothing: it takes the kernels, the two per-channel rows and the temper as they stand,
    and [check_shape] holds every rule the consumers assume.

    The layout is that module's docstring. Two of its facts are facts of THIS reader:
    every tensor is int32, because [Nx_io] skips every dtype it does not hold and int8 is
    one of them; and the temper and the Q travel as the named tensors ["temper"] and
    ["activation_q"], because [Nx_io] gives no access to [__metadata__]. A file quantized
    at another Q refuses here and not in the middle of a walk.

    It raises [Invalid_argument] when the tensor count does not divide into layers, when a
    tensor is missing, or when a shape or a rule does not hold; the message names the
    tensor that refused. *)
val of_int8_checkpoint : string -> t

(** the ROM image of the circuit: every kernel in checkpoint order, one byte for each
    weight, two's complement. The gains and the biases are not in it — they are
    per-channel constants of the elaboration, as era five's per-head constants were. *)
val rom_bits : t -> Hardcaml.Bits.t array

(** the byte base of each kernel inside the image, in layer order *)
val rom_bases : t -> int array

(** {1 The walk} *)

(** The register of each seat as classes: what [opening_canvas] draws inside, stated one
    time. The circuit reads these through its elaboration, thus the opening of the walk
    and the opening of the board are one rule and never a second reading of it. *)
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
    when its uniform times 2^24 falls under this integer. JAX and this side compare
    against the same number, and it is the entry the circuit's table holds. *)
val anneal_threshold : step:int -> walk:int -> int

(** [cell_order ~steps] is the order the walk visits the cells of a canvas in: a step at a
    time, and the seats of a step inside it, as [step, voice] pairs.

    Every uniform of the walk is drawn in this order — the opening, each mask, and the
    draw of each hidden cell — thus the integer twin of [jax/diffusion/quantized.py] takes
    the same walk by taking the same order, and the circuit reads this rule rather than
    restate it. *)
val cell_order : steps:int -> (int * int) list

(** [opening_canvas state ~steps] is the seeded opening of the walk: one uniform for each
    cell in the cell order, the class [low + floor (u * width)] inside the register of the
    cell's own seat. The twin's engine opens on the same rule in [infer.opening_canvas],
    thus the two openings are one rule and one consumption. *)
val opening_canvas : Prng.state -> steps:int -> Prng.state * int array array

(** [hidden_cells state ~steps ~threshold] is the mask of one pass: one uniform for each
    cell in the cell order, hidden exactly when [u * 2^24] falls under [threshold]. The
    same sharing argument as [opening_canvas]. *)
val hidden_cells
  :  Prng.state
  -> steps:int
  -> threshold:int
  -> Prng.state * bool array array

(** {1 The frames} *)

(** [frames_of_canvas canvas] is the frame of each step: the classes of a step become the
    voice codes of one word, seat 0 in the low byte. [Frame.events_of_frames] then states
    the events of the wire, and the step lines of the JAX audition print them — the decode
    is the rule of the frame, and it is the same rule [jax/data.py] states, thus a walk
    here and a walk there compare as text. *)
val frames_of_canvas : int array array -> int array

(** {1 The drawn model} *)

module For_test : sig
  (** [drawn ~layers ~width ~seed] is a model of that shape, its values drawn under [Prng]
      STRAIGHT INTO THE FORMATS: the kernel bytes at one fixed exponent, a per-channel
      gain in int16 whose shift fits the six bits of the norm word, a bias in int16, and
      the temper of temperature 1.0. It passes [check_shape], thus a test reads no file
      and quantizes nothing.

      THE DRAWN TRUNK HOLDS O(1) ACTIVATIONS, and that is what makes the model worth
      drawing: a gain drawn flat inside int16 would clamp every write of the trunk or zero
      it, and the pictures, the frames and the cycle counts of the tests that read this
      model would all read a machine that no checkpoint makes. The implementation states
      the arithmetic that holds it.

      The tests that read it need a model OF A SHAPE — cycle counts, tile counts, the turn
      order, the images' self-consistency, a picture — and none of them needs the twin's
      arithmetic: the two gates that did are [jax/tests/test_rtl.py]'s. *)
  val drawn : layers:int -> width:int -> seed:int -> t

  (** the kernels of the image, in its order; the gates read them beside [rom_bases] *)
  val rom_tensors : t -> quantized list
end
