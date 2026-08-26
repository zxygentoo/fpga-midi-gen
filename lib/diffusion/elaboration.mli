(** The elaboration of era six: one model at one geometry, as a value.

    L1 of the diffusion source. The circuit reads its shape out of the checkpoint the way
    [Diffusion.Config.of_checkpoint] does — no flag states a dimension — thus THIS MODULE
    IS THE ONLY PLACE THAT READS THE MODEL. The layer table, the weight image, the
    per-channel norms, the anneal table and the cycle cost all come out of [create], and
    every width, every depth and every base of the circuit is a field of the result. A new
    checkpoint moves nothing else.

    The design is [docs/diffusion_rtl.md], "The circuit". What a caller must know:

    - **The weight image is packed in the DWELL order and not the checkpoint order.** The
      twin ([Quantized.Model.rom_bits]) stays the authority on every value; the
      permutation belongs here, and it buys the circuit a ROM address that only counts. A
      group that runs past a layer's channels pads with zero bytes — those lanes multiply
      by zero and the drain does not read them — thus each [(tap, input channel)] row of
      the image is a whole number of words.
    - **A layer's role states its two ends, its ReLU and its residual together.** There
      are no independent flags that can disagree with each other.
    - **The cycle counts are exact and not bounds.** [create] refuses a layer whose dwell
      is shorter than its drain and the band loads behind it, thus no dwell ever waits and
      [layer_cycles] is the number the cycle bench must measure.
    - The tables are [Hardcaml.Bits.t] arrays because the bitstream carries them: they
      initialize the memories of the circuit as they stand. *)

(** A layer's role in the trunk. It states the two ends, the ReLU and the residual
    together: the stem decodes the planes into X under a ReLU; a pair opens X into Y under
    a ReLU and closes Y back into X under none, because the [max 0] of the join belongs to
    the residual add behind it; and the head reads X under no ReLU and states the logits
    of one step. *)
type layer_role =
  | Stem
  | Pair_open
  | Pair_close
  | Head

(** One layer of the table. The counters of the circuit size on these numbers, and its ROM
    and norm addresses start at these bases. *)
type layer =
  { role : layer_role
  ; inputs : int (** the input channels; the kernel reads as [3; 3; inputs; outputs] *)
  ; outputs : int (** the output channels *)
  ; groups : int (** the output groups the dwell walks: [ceil (outputs / lanes)] *)
  ; weight_base : int (** the first word of this layer in [weight_rom] *)
  ; norm_base : int (** the first word of this layer in [norm_rom] *)
  }

(** One model at one geometry: the four numbers of the geometry, the table, and the tables
    the bitstream carries. *)
type t =
  { steps : int (** T: the steps of one canvas *)
  ; rows : int (** P: the pitch rows of a column, and the classes of one cell *)
  ; lanes : int (** G: the output channels of one group, thus [rows * lanes] lanes *)
  ; walk : int (** N: the passes of the walk, and the depth of [alpha_rom] *)
  ; store_channels : int
  (** H: the channels that each of the two activation stores holds *)
  ; layers : layer array
  ; weight_rom : Hardcaml.Bits.t array
  (** the weight words, [lanes] bytes each and lane 0 in the low byte, in the dwell order:
      the group, then the input channel, then the tap. One column's dwell therefore walks
      a layer's whole range straight through, thus the circuit's ROM address is ONE
      COUNTER that reloads once for each column; any other order makes it a stride. *)
  ; norm_rom : Hardcaml.Bits.t array
  (** one word for each output channel of the model, in the layer order: the bias in the
      low [bias_bits], the shift in the [shift_bits] above it, and the value of the gain
      in the top [gain_bits]. The three stand at one address because the epilogue wants
      the three at one time, thus no two of them can fall out of step. *)
  ; alpha_rom : Hardcaml.Bits.t array
  (** the anneal thresholds of [Diffusion.anneal_threshold], one for each pass, on the
      24-bit grid of the generator *)
  ; temper : Mgen_nn.Quantized.Constants.scale
  (** the sampling temper of the model, [log2e / T]: the draw multiplies by it *)
  }

(** the taps of one 3 by 3 kernel: 9. A dwell counts them. *)
val taps : int

(** the bits of one [alpha_rom] entry: 24, the grid of the generator *)
val alpha_bits : int

(** The fields of a norm word, low to high: the bias, the shift, the value of the gain.

    THE SHIFT FIELD SIZES ON THE RULE AND NOT ON THE CHECKPOINT. The q of a gain is
    [e + weight_exponent], and the two exponent rules cap at 30 and at 14, thus 44 is the
    largest q a quantizer can state and six bits hold it. A field sized on the elected
    model's own peak would save a bit or two and would make a drawn-weight timing probe
    elaborate a DIFFERENT netlist from the trained build; the bits are worth less than
    that. [create] refuses a q the field cannot hold. *)
val bias_bits : int

val shift_bits : int
val gain_bits : int
val norm_bits : int

(** [norm_word gain ~bias] is one word of [norm_rom]. THE FIELD ORDER IS A FACT OF THIS
    FUNCTION AND OF NOTHING ELSE, thus a consumer that slices another way disagrees with
    the image the bitstream carries — and a gate that packs here and slices there is what
    says so, rather than a board. It raises [Invalid_argument] when the gain's shift does
    not fit [shift_bits]. *)
val norm_word : Mgen_nn.Quantized.Constants.scale -> bias:int -> Hardcaml.Bits.t

(** [column_address t ~step ~channel] is where a store holds the column:
    [step * store_channels + channel]. THE MAP IS A FACT OF THIS FUNCTION AND OF NOTHING
    ELSE — the circuit's ports and the stream instrument slice one rule, the argument
    [norm_word] already makes. The map is t-major, thus the writes of one group land
    consecutive; nothing else distinguishes the orders. *)
val column_address : t -> step:int -> channel:int -> int

(** the words of one activation store: [steps * store_channels] *)
val store_depth : t -> int

(** [create ?rows model ~steps ~lanes ~walk] elaborates [model] at the geometry those
    numbers state.

    [rows] is [Diffusion.rows] by default: the board pins P there and Gate B compares
    there, because the twin holds P at that number. The circuit takes it as a parameter so
    that the twin can follow later — the deferral of [docs/diffusion_rtl.md], "The
    iteration strategy".

    It raises [Invalid_argument], and the message names what it refused:

    - whatever [Quantized.Model.check_shape] refuses;
    - a layer whose dwell [9 * inputs] is shorter than [rows + lanes + 2], thus the
      [rows]-stage drain chain always empties before the next capture AND the residual
      columns and norm words of the next group always land before the drain that reads
      them — the engine holds one band for every group and loads it in the window the
      drain before it just freed;
    - a gain shift that does not fit [shift_bits];
    - a canvas of no steps, a column of no rows, a group of no lanes, or a walk of no
      passes. *)
val create : ?rows:int -> Quantized.Model.t -> steps:int -> lanes:int -> walk:int -> t

(** [dwell layer] is the cycles of one (column, group): one for each (tap, input channel)
    pair. *)
val dwell : layer -> int

(** [layer_cycles t layer] is one layer, EXACTLY: the dwells of every column and group,
    and one drain tail of [rows] behind the last of them. It is a count and not a bound,
    because [create] refuses a layer whose chain cannot empty. What the layer turn itself
    costs is the cycle bench's to measure. *)
val layer_cycles : t -> layer -> int

(** the layers of one forward pass, summed *)
val forward_cycles : t -> int

(** [cell_walk_cycles t] is one uniform for each cell in the cell order: what the opening
    costs, and what the mask of one pass costs. *)
val cell_walk_cycles : t -> int

(** [pass_cycles t] is one pass LESS THE DRAW: the mask and the forward. The draw's cycles
    belong to the walk and to the bench — the design estimates about [2 * rows] for each
    hidden cell — and a number this module cannot state exactly it does not state. *)
val pass_cycles : t -> int

(** The table as text: the geometry, one line for each layer, the sizes of the memories
    and the cycles. The schedule prints — the discipline of the eras — thus the cost model
    of the document and the machine cannot part without a test saying so. *)
val to_string : t -> string
