(** The elaboration of era six: one model at one geometry, as a value.

    L1 of the diffusion source. The circuit reads its shape out of the contract file's own
    tensor shapes — no flag states a dimension — thus THIS MODULE IS THE ONLY PLACE THAT
    READS THE MODEL. The layer table, the weight image, the per-channel norms, the anneal
    table and the cycle cost all come out of [create], and every width, every depth and
    every base of the circuit is a field of the result. A new checkpoint moves nothing
    else.

    The design is [docs/diffusion_rtl.md], "The circuit". What a caller must know:

    - **The weight image is packed in the DWELL order and not the checkpoint order.** The
      twin ([Model.rom_bits]) stays the authority on every value; the permutation belongs
      here, and it buys the circuit a ROM address that only counts. A group that runs past
      a layer's channels pads with zero bytes — those lanes multiply by zero and the drain
      does not read them — thus each [(tap, input channel)] row of the image is a whole
      number of words.
    - **THE MEMORIES ARE BANKED, AND A BANK IS A POWER OF TWO.** Vivado pads an inferred
      memory to its full address space and says nothing: rung 2's image asked 64 tiles
      against 49 free and the mapper demoted every ROM of the design to fabric, and a
      store of 1280 columns maps as [2048x768]. [weight_banks] and [store_banks] are the
      plans that take the padding back — the concatenation of the weight banks IS
      [weight_rom] in the dwell order, thus the banking is a permutation of ADDRESS SPACE
      and never of values, and [to_string] prints both plans. A store carries no image:
      the circuit writes it, and only the tiling is banked.
    - **A layer's role states its two ends, its ReLU and its residual together.** There
      are no independent flags that can disagree with each other.
    - **The cycle counts are exact and not bounds.** [create] refuses a layer whose dwell
      is shorter than its drain and the band loads behind it, thus no dwell ever waits and
      [forward_cycles] is the number the cycle bench must measure.
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

(** One bank of a memory: where it starts in the flat address space, and the words it
    addresses. THE DEPTH IS A POWER OF TWO AND THE BASE IS A MULTIPLE OF IT, thus the
    offset inside a bank is the LOW BITS of the flat address and the bank is the bits
    above them: no subtractor stands on the address side, and the circuit's one address
    feeds every bank as it stands. The banks tile the space from address zero, and the
    last of them pads. The weight ROM and the two activation stores bank by the one rule;
    only the ROM carries an image. *)
type bank =
  { base : int
  ; depth : int
  }

(** One turn of the walk: the layer of phase A, and the layer of phase B where the turn is
    a pair. The stem and the head are turns of one phase.

    A TURN IS WHAT THE ENGINE PRIMES ONCE AND DRAINS ONCE. Inside a pair the two layers
    interleave block by block — A at column [s], then B at column [s - 2] — thus a layer
    is no longer a unit the walk can name, and the cycle model counts turns. *)
type turn =
  { first : int
  ; second : int option
  }

(** One model at one geometry: the four numbers of the geometry, the table, and the tables
    the bitstream carries. *)
type t =
  { steps : int (** T: the steps of one sheet *)
  ; rows : int (** P: the pitch rows of a column, and the classes of one cell *)
  ; lanes : int (** G: the output channels of one group, thus [rows * lanes] lanes *)
  ; walk : int (** N: the passes of the walk, and the depth of [alpha_rom] *)
  ; store_channels : int
  (** H: the channels that each of the two activation stores holds *)
  ; layers : layer array
  ; turns : turn array
  (** the stem, then one turn for each pair, then the head. [create] refuses a model whose
      roles do not walk that shape. *)
  ; weight_rom : Hardcaml.Bits.t array
  (** the weight words, [lanes] bytes each and lane 0 in the low byte, in the dwell order:
      the group, then the input channel, then the tap. One column's dwell therefore walks
      a layer's whole range straight through, thus the circuit's ROM address is ONE
      COUNTER that reloads once for each column; any other order makes it a stride. *)
  ; weight_banks : bank array
  (** how the weight image is banked over the memories: 8192 and 512 at rung 1, 32768 and
      4096 at rung 2. The plan is the top bit of the word count and one tail, taken only
      where it costs less than one bank, and a bank is never below 512 words — the
      smallest tile a word of this width fills. The image does not move: a bank holds
      [weight_rom] from [base] for [depth] words, and the words above the last are the
      pad. *)
  ; store_banks : bank array
  (** how each of the two activation stores is banked over the memories: the same plan
      over [store_depth]. One bank of 2048 at rung 2; 2048 and 512 at T 128 and H 20,
      where one memory of 2560 columns would map as 4096 and cost 86 tiles against 54. A
      store holds no image, thus a bank of it is a range of columns and nothing more. *)
  ; ring_banks : bank array
  (** how the Y ring is banked: [ring_depth] columns, one bank of 512 at every elected
      rung. Y never exists as a tensor — the fused pair keeps four columns of it live —
      thus the ring costs the WIDTH of a column and not the length of the sheet: eleven
      tiles at any T. *)
  ; norm_rom : Hardcaml.Bits.t array
  (** one word for each output channel of the model, in the layer order: the bias in the
      low [bias_bits], the shift in the [shift_bits] above it, and the value of the gain
      in the top [gain_bits]. The three stand at one address because the epilogue wants
      the three at one time, thus no two of them can fall out of step. *)
  ; alpha_rom : Hardcaml.Bits.t array
  (** the anneal thresholds of [Model.anneal_threshold], one for each pass, on the 24-bit
      grid of the generator *)
  ; openings : Model.opening array
  (** the register of each seat as classes, [Model.seat_openings] carried: the walk's OPEN
      multiplies its uniform by [width] and adds [low]. It rides here so that the circuit
      reads ONE value for everything it holds, as it reads its widths and its bases here. *)
  ; temper : Mgen_nn.Quantized.Constants.scale
  (** the sampling temper of the model, [log2e / T]: the draw multiplies by it *)
  }

(** the taps of one 3 by 3 kernel: 9. A dwell counts them. *)
val taps : int

(** the bits of one [alpha_rom] entry: the grid of the generator, [Prng.uniform_bits] *)
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

(** The fields of one norm word, unpacked. [norm_word] packs them and [Rtl.norm_fields]
    slices them back, thus the format's two halves stand in one module and the gate below
    holds them together over every value a quantizer can state. *)
type 'a norm_fields =
  { gain : 'a
  ; shift : 'a
  ; bias : 'a
  }

(** The nest of a turn, as the lead frame carries it: the input channel inside a block,
    the group inside a phase, the pair's step counter, and the phase. Inside a pair the
    lead frame can be in B while the now frame is still in A, thus the phase travels in
    the frames as the step and the group already do. *)
type 'a nest =
  { cin : 'a
  ; group : 'a
  ; step : 'a
  ; phase : 'a
  }

(** what one advance of the nest gives: the state after it, and whether the turn closed *)
type 'a block_walk =
  { next : 'a nest
  ; ends : 'a
  }

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

(** [is_pair turn] is whether the turn interleaves two layers. *)
val is_pair : turn -> bool

(** [turn_cycles t turn] is one turn, exactly: the dwells of its one or two layers and ONE
    drain tail behind the last of them. *)
val turn_cycles : t -> turn -> int

(** [weight_bank_image t bank] is the bank as the bitstream carries it: the bank's slice
    of [weight_rom], and then its pad of zero words. It is what initializes one weight
    memory of the circuit. A store has no image, thus this belongs to the ROM alone. *)
val weight_bank_image : t -> bank -> Hardcaml.Bits.t array

(** the bits a store address takes: [address_bits_for (store_depth t)] *)
val store_bits : t -> int

(** the bits a channel index takes. A ragged group runs past its layer's own channels — a
    head of four seats in a group of three reaches channel five — thus the width follows
    the GROUPS and not the outputs. The circuit sizes its ports on this and does not
    derive it again. *)
val channel_bits : t -> int

(** [create ?rows model ~steps ~lanes ~walk] elaborates [model] at the geometry those
    numbers state.

    [rows] is [Model.rows] by default: the board pins P there and Gate B compares there,
    because the twin holds P at that number. The circuit takes it as a parameter so that
    the twin can follow later — the deferral of [docs/diffusion_rtl.md], "The iteration
    strategy".

    It raises [Invalid_argument], and the message names what it refused:

    - whatever [Model.check_shape] refuses;
    - a layer whose dwell [9 * inputs] is shorter than [rows + lanes + 2], thus the
      [rows]-stage drain chain always empties before the next capture AND the residual
      columns and norm words of the next group always land before the drain that reads
      them — the engine holds one band for every group and loads it in the window the
      drain before it just freed;
    - a gain shift that does not fit [shift_bits];
    - a sheet of no steps, a column of no rows, a group of no lanes, or a walk of no
      passes. *)
val create : ?rows:int -> Model.t -> steps:int -> lanes:int -> walk:int -> t

(** [forward_cycles t] is one forward pass, EXACTLY: for each layer the dwells of every
    column and group — one cycle for each (tap, input channel) pair — and one drain tail
    of [rows] behind the last of them, summed over the layers. It is a count and not a
    bound, because [create] refuses a layer whose chain cannot empty. What the layer turns
    themselves cost is the cycle bench's to measure. *)
val forward_cycles : t -> int

(** [cell_walk_cycles t] is one uniform for each cell in the cell order: what the opening
    costs, and what the mask of one pass costs. The machine spends TWO CYCLES MORE on each
    of them — the frame that writes a cell stands two cycles behind the frame that draws
    it — and [Source]'s walk bench measures that tail. *)
val cell_walk_cycles : t -> int

(** [pass_cycles t] is one pass LESS THE DRAW: the mask and the forward. The draw's cycles
    belong to the walk and to the bench — [Source]'s walk bench measures a standing cell
    at 1 cycle and a hidden one at 162: its seat read, its uniform and [Draw]'s own
    [busy_cycles], which is 155 at the era's 48 classes — and a number this module cannot
    state exactly it does not state. *)
val pass_cycles : t -> int

(** The table as text: the geometry, one line for each layer, the sizes of the memories
    and the cycles. The schedule prints — the discipline of the eras — thus the cost model
    of the document and the machine cannot part without a test saying so. *)
val to_string : t -> string

(** The maps the image and the circuit must agree on, over any combinational type.

    Signal land cannot call [column_address] or [norm_word] above — it holds signals and
    not integers — thus the circuit's answer until now was to state each map a second time
    and let a run-time gate weld the two. [Vocab.Rtl] already held the better answer for
    the vocabulary's own map: ONE RULE OVER [Comb], evaluated at [Bits] by a test and
    elaborated at [Signal] by the circuit. The two halves then stop being two, and the
    fused rung that moves the store map moves it in one place.

    **THE PIN IS A LABELLED ARGUMENT.** The address maps carry a multiply, and the
    multiply is kept out of the DSPs the array owns ([Column_array.no_dsp]).
    [add_attribute] is [Signal]'s alone and means nothing to a [Bits] value, thus the pin
    cannot live inside [Comb]; as a functor argument it would make [Epilogue] — which
    wants only the norm fields, and multiplies nothing — name the array's rule for no
    reason. A caller that pins every address fixes it one time by partial application. *)
module Rtl : sig
  module Make (Comb : Hardcaml.Comb.S) : sig
    (** [column_address ~pin t ~step ~channel] is [column_address] as a circuit, at
        [store_bits t] bits. *)
    val column_address
      :  pin:(Comb.t -> Comb.t)
      -> t
      -> step:Comb.t
      -> channel:Comb.t
      -> Comb.t

    (** [channel_of ~pin t ~group ~lane] is the channel a lane of a group holds, as a
        circuit, at [channel_bits t] bits. *)
    val channel_of : pin:(Comb.t -> Comb.t) -> t -> group:Comb.t -> lane:Comb.t -> Comb.t

    (** [bank_at banks ~address] is which bank of [banks] holds a flat address, at
        [address_bits_for (Array.length banks)] bits: the last bank whose base the address
        has reached. The offset inside the bank is the low bits of the same address, thus
        the address side holds no subtractor and the select is the top address bits alone.
        A caller carries the answer beside the DATA and not beside the address, and holds
        it as many times as the data is registered before the mux. *)
    val bank_at : bank array -> address:Comb.t -> Comb.t

    (** [ring_address ~pin t ~step ~channel] is where the ring holds a column of Y: the
        store's own map over the low bits of the semantic step, at the width a ring
        address takes. The caller drives the SEMANTIC column and this takes the bits the
        ring keeps, thus no modulo and no second counter stands anywhere.

        THE MAP IS A FACT OF THIS FUNCTION AND OF NOTHING ELSE. It has no software half:
        the ring is the circuit's own memory and no image, gate or instrument addresses
        it, thus a second statement of the map would have nothing to weld it to. *)
    val ring_address
      :  pin:(Comb.t -> Comb.t)
      -> t
      -> step:Comb.t
      -> channel:Comb.t
      -> Comb.t

    (** [layer_of t ~turn ~phase] is which layer of the table a frame is in. Every fact of
        the table is muxed by this and not by the turn, thus the table's mux is the one it
        always was and only its index has learned to travel between the frames. *)
    val layer_of : t -> turn:Comb.t -> phase:Comb.t -> Comb.t

    (** [next_block t ~is_pair ~cin_count ~group_count nest] advances the lead frame's
        nest by one cycle and says whether the turn closed. The counts are the LEAD
        layer's, because it is the lead frame that walks. This is the circuit half of
        [blocks_of_turn]. *)
    val next_block
      :  t
      -> is_pair:Comb.t
      -> cin_count:Comb.t
      -> group_count:Comb.t
      -> Comb.t nest
      -> Comb.t block_walk

    (** [norm_fields word] slices one word of [norm_rom]: the inverse of [norm_word], and
        the only reader of the field order besides it. *)
    val norm_fields : Comb.t -> Comb.t norm_fields
  end

  include module type of Make (Hardcaml.Signal)
end
