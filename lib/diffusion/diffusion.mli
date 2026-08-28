(** The facts of the walk: the roll, the registers of the seats, the consumption order,
    and the frames a canvas states.

    One canvas is T steps of four voices over the 48-class vocabulary of [Vocab], beside a
    mask over its cells. The model reads the canvas with the masked cells hidden and
    states a distribution over the classes of every cell at once; the Gibbs walk hides and
    restores cells in turn until a piece stands. Nothing is causal and nothing is written
    left to right. The design of the model is [docs/diffusion.md], and the design of this
    layer and its gates is [docs/diffusion_rtl.md].

    **THE MODEL IS NOT HERE ANY MORE.** The float model is [jax/diffusion/model.py] and
    the integer twin is [jax/diffusion/quantized.py]; the order that cut them out of OCaml
    is [docs/diffusion_ocaml_cut.md]. What stays is what the CIRCUIT reads — the
    elaboration, the source and the benches — and every one of those facts is a rule the
    RTL must equal rather than restate.

    Every draw of the walk comes from [Prng], the xorshift32 of the circuit, and THE
    CONSUMPTION ORDER IS THE CONTRACT of [docs/diffusion_rtl.md]: the opening draws one
    uniform for each cell, each pass draws one for each cell (the masks) and then one for
    each hidden cell (the classes), and the cells walk in step-major, seat-minor order
    everywhere. Every cell of a canvas is free — nothing is given to a walk of this era.
    One seed names one canvas — in JAX, here and on the board — and the same seed gives
    the same piece. *)

(** the rows of the roll: the classes of [Vocab], row 0 silence *)
val rows : int

(** the seats of a step: the voices of [Frame] *)
val voices : int

(** The shape of a model: the two numbers that size every tensor. The elected shape of the
    era is L 48 by H 20. The checkpoint states its own shape and this record no longer
    reads one — [Quantized.Model.of_int8_checkpoint] takes the shape out of the tensors —
    thus what stays here is the shape a DRAWN model is asked for. *)
module Config : sig
  type t =
    { layers : int (** L: the stem, the residual pairs, and the head *)
    ; width : int (** H, the channels of the trunk *)
    }
end

(** [tensor_column x ~step ~channel ~channels] is one column of any tensor of this era:
    the [rows] values that one step and one channel hold, read out of the flat array of a
    [steps; rows; channels] tensor.

    EVERY TENSOR OF THE ERA HAS THE ONE SHAPE — the stem's planes, a layer's output, the
    head's logits, the stem's planes of [Canvas] — thus THE INDEX RULE STANDS HERE ALONE
    and no reader of a tensor slices one by hand. *)
val tensor_column : 'a array -> step:int -> channel:int -> channels:int -> 'a array

(** [anneal_threshold ~step ~walk] is the masking threshold of pass [step] of [walk], on
    the 24-bit grid of the generator: [floor (alpha * 2^24)] of the paper's annealed
    probability [alpha = max (0.1, 0.9 - 0.8 step / (0.7 walk))]. A cell hides exactly
    when its uniform times 2^24 falls under this integer. The float sides and the twin
    compare against the same number, and it is the entry the circuit's table will hold. *)
val anneal_threshold : step:int -> walk:int -> int

(** [cell_order ~steps] is the order the walk visits the cells of a canvas in: a step at a
    time, and the seats of a step inside it, as [step, voice] pairs.

    Every uniform of the walk is drawn in this order — the opening, each mask, and the
    draw of each hidden cell — thus the integer twin of [jax/diffusion/quantized.py] takes
    the same walk by taking the same order, and the circuit reads this rule rather than
    restate it. *)
val cell_order : steps:int -> (int * int) list

(** The register of each seat as classes: what [opening_canvas] draws inside, stated one
    time. The circuit reads these through its elaboration, thus the opening of the walk
    and the opening of the board are one rule and never a second reading of it. *)
type opening =
  { low : int (** the lowest class of the seat's register *)
  ; width : int (** the classes of the register: [high - low + 1] *)
  }

(** the opening of each seat, seat 0 the bass: [Jsb.voice_ranges] turned around and read
    through the class map of this module *)
val seat_openings : opening array

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

(** [frames_of_canvas canvas] is the frame of each step: the classes of a step become the
    voice codes of one word, seat 0 in the low byte. [Frame.events_of_frames] then states
    the events of the wire, and the step lines of the players print them — the decode is
    the rule of the frame, and it is the same rule [jax/data.py] states, thus a walk here
    and a walk there compare as text. *)
val frames_of_canvas : int array array -> int array
