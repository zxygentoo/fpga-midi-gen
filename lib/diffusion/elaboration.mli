(** The elaboration of era six: one model at one geometry, as a value.

    The circuit reads its shape out of the contract file's own tensor shapes — no flag
    states a dimension — thus THIS MODULE IS THE ONLY PLACE THAT READS THE MODEL. Every
    width, depth and base of the circuit is a field of [create]'s result, and a new
    checkpoint moves nothing else.

    The design is [docs/diffusion_rtl.md], "The circuit". What a caller must know:

    - **The weight image is packed in the DWELL order and not the checkpoint order.** The
      twin ([Model.rom_bits]) stays the authority on every value; the permutation buys the
      circuit a ROM address that only counts. A group that runs past a layer's channels
      pads with zero bytes, thus each [(tap, input channel)] row is a whole number of
      words.
    - **THE MEMORIES ARE BANKED, AND A BANK IS A POWER OF TWO.** Vivado pads an inferred
      memory to its full address space and says nothing: rung 2's image asked 64 tiles
      against 49 free and the mapper demoted every ROM to fabric. [weight_banks] and
      [store_banks] take that padding back. The concatenation of the weight banks IS
      [weight_rom] in the dwell order, thus the banking permutes ADDRESS SPACE and never
      values. A store carries no image — only its tiling is banked.
    - **A layer's role states its two ends, its ReLU and its residual together.** There
      are no independent flags that can disagree with each other.
    - **The cycle counts are exact and not bounds.** [create] refuses a layer whose dwell
      is shorter than its drain and the band loads behind it, thus no dwell ever waits.
    - The tables are [Hardcaml.Bits.t] arrays because the bitstream carries them: they
      initialize the memories of the circuit as they stand. *)

(** A layer's role in the trunk, stating its two ends, its ReLU and its residual together:
    the stem decodes the planes into X under a ReLU; a pair opens X into Y under a ReLU
    and closes Y back into X under none, the join's own [max 0] standing behind the
    residual add; the head reads X under no ReLU. *)
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
    above them — no subtractor on the address side, and one address feeds every bank as it
    stands. The banks tile from zero and the last of them pads. *)
type bank =
  { base : int
  ; depth : int
  }

(** One turn of the walk: the layer of phase A, and the layer of phase B where the turn is
    a pair. The stem and the head are turns of one phase.

    A TURN IS WHAT THE ENGINE PRIMES ONCE AND DRAINS ONCE. Inside a pair the two layers
    interleave block by block — A at column [s], then B at column [s - 2] — thus a layer
    is not a unit the walk can name, and the cycle model counts turns. *)
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
      the group, then the input channel, then the tap. One column's dwell walks a layer's
      whole range straight through, thus the ROM address is ONE COUNTER that reloads once
      for each column; any other order makes it a stride. *)
  ; weight_banks : bank array
  (** how the weight image is banked: the top bit of the word count and one tail, taken
      only where it costs less than one bank, and never a bank below 512 words — the
      smallest tile a word of this width fills. A bank holds [weight_rom] from [base] for
      [depth] words, and the words above the last are the pad. *)
  ; store_banks : bank array
  (** the same plan over [store_depth]: 2048 and 512 at T 128 and H 20, where one memory
      of 2560 columns would map as 4096 and cost 86 tiles against 54. A store holds no
      image, thus a bank of it is a range of columns. *)
  ; ring_banks : bank array
  (** how the Y ring is banked: one bank of 512 at every elected rung. Y never exists as a
      tensor — the fused pair keeps four columns live — thus the ring costs the WIDTH of a
      column and not the length of the sheet: eleven tiles at any T. *)
  ; norm_rom : Hardcaml.Bits.t array
  (** one word for each output channel, in the layer order: the bias in the low bits, the
      shift above it, the gain's value on top — the format [norm_word] packs and
      [Rtl.norm_fields] slices. The three stand at one address because the epilogue wants
      them at one time. *)
  ; alpha_rom : Hardcaml.Bits.t array
  (** the anneal thresholds of [Model.anneal_threshold], one for each pass, on the 24-bit
      grid of the generator *)
  ; openings : Model.opening array
  (** the register of each seat as classes, [Model.seat_openings] carried: the walk's OPEN
      multiplies its uniform by [width] and adds [low] *)
  ; temper : Mgen_nn.Quantized.Constants.scale
  (** the sampling temper of the model, [log2e / T]: the draw multiplies by it *)
  }

(** the taps of one 3 by 3 kernel: 9. A dwell counts them. *)
val taps : int

(** the bits of one [alpha_rom] entry: the grid of the generator, [Prng.uniform_bits] *)
val alpha_bits : int

(** The widths of a norm word above its bias field, and the width of the whole.

    THE SHIFT FIELD SIZES ON THE RULE AND NOT ON THE CHECKPOINT: the two exponent rules
    cap a gain's q at 44, which six bits hold. Sizing it on the elected model's own peak
    would save a bit and would make a drawn-weight timing probe elaborate a DIFFERENT
    netlist from the trained build. [create] refuses a q the field cannot hold. *)
val shift_bits : int

val gain_bits : int
val norm_bits : int

(** The fields of one norm word, unpacked. [norm_word] packs them and [Rtl.norm_fields]
    slices them back, thus the format's two halves stand in one module. *)
type 'a norm_fields =
  { gain : 'a
  ; shift : 'a
  ; bias : 'a
  }

(** The nest of a turn, as the lead frame carries it: the input channel inside a block,
    the group inside a phase, the pair's step counter, and the phase. Inside a pair the
    lead frame can be in B while the now frame is still in A, thus the phase travels in
    the frames as the step and the group do. *)
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
    FUNCTION AND OF NOTHING ELSE. It raises [Invalid_argument] when the gain's shift does
    not fit [shift_bits]. *)
val norm_word : Mgen_nn.Quantized.Constants.scale -> bias:int -> Hardcaml.Bits.t

(** [column_address t ~step ~channel] is where a store holds the column:
    [step * store_channels + channel]. THE MAP IS A FACT OF THIS FUNCTION AND OF NOTHING
    ELSE. It is t-major, thus the writes of one group land consecutive; nothing else
    distinguishes the orders. *)
val column_address : t -> step:int -> channel:int -> int

(** the words of one activation store: [steps * store_channels] *)
val store_depth : t -> int

(** [is_pair turn] is whether the turn interleaves two layers. *)
val is_pair : turn -> bool

(** [turn_cycles t turn] is one turn, exactly: the dwells of its one or two layers and ONE
    drain tail behind the last of them. *)
val turn_cycles : t -> turn -> int

(** [weight_bank_image t bank] is what initializes one weight memory: the bank's slice of
    [weight_rom], then its pad of zero words. *)
val weight_bank_image : t -> bank -> Hardcaml.Bits.t array

(** the bits a store address takes: [address_bits_for (store_depth t)] *)
val store_bits : t -> int

(** the bits a channel index takes. A ragged group runs past its layer's own channels — a
    head of four seats in a group of three reaches channel five — thus the width follows
    the GROUPS and not the outputs. *)
val channel_bits : t -> int

(** [create ?rows model ~steps ~lanes ~walk] elaborates [model] at the geometry those
    numbers state.

    [rows] is [Model.rows] by default, where the board pins P and Gate B compares. The
    circuit takes it as a parameter so that the twin can follow later — the deferral of
    [docs/diffusion_rtl.md], "The iteration strategy".

    It raises [Invalid_argument], and the message names what it refused:

    - whatever [Model.check_shape] refuses;
    - a layer whose dwell [9 * inputs] is shorter than [rows + lanes + 2], so that the
      drain chain always empties before the next capture and the next group's band always
      lands before the drain that reads it;
    - a gain shift that does not fit [shift_bits];
    - a sheet of no steps, a column of no rows, a group of no lanes, or a walk of no
      passes. *)
val create : ?rows:int -> Model.t -> steps:int -> lanes:int -> walk:int -> t

(** [forward_cycles t] is one forward pass, EXACTLY: for each layer the dwells of every
    column and group — one cycle for each (tap, input channel) pair — and one drain tail
    of [rows] behind the last of them. It is a count and not a bound, because [create]
    refuses a layer whose chain cannot empty. *)
val forward_cycles : t -> int

(** [cell_walk_cycles t] is one uniform for each cell in the cell order: what the opening
    costs, and what one pass's mask costs. The machine spends TWO CYCLES MORE on each —
    the frame that writes a cell stands two behind the frame that draws it — and
    [Source]'s walk bench measures that tail. *)
val cell_walk_cycles : t -> int

(** [pass_cycles t] is one pass LESS THE DRAW: the mask and the forward. What a draw costs
    depends on how many cells the mask hid, thus this module does not state it; [Source]'s
    walk bench measures a standing cell at 1 cycle and a hidden one at 162. *)
val pass_cycles : t -> int

(** The table as text: the geometry, one line for each layer, the sizes of the memories
    and the cycles. The schedule prints, thus the document's cost model and the machine
    cannot part without a test saying so. *)
val to_string : t -> string

(** The maps the image and the circuit must agree on, over any combinational type.

    Signal land cannot call [column_address] or [norm_word] above — it holds signals and
    not integers — thus each map would otherwise be stated a second time and welded by a
    run-time gate. [Vocab.Rtl]'s answer is taken instead: ONE RULE OVER [Comb], evaluated
    at [Bits] by a test and elaborated at [Signal] by the circuit.

    **THE PIN IS A LABELLED ARGUMENT** and not a functor argument. The address maps carry
    a multiply that must stay out of the DSPs ([Mgen_nn.Placement.no_dsp]), but
    [add_attribute] is [Signal]'s alone and means nothing at [Bits], thus the pin cannot
    live inside [Comb]; as a functor argument it would make [Epilogue], which multiplies
    nothing, name the array's rule for no reason. A caller fixes it once by partial
    application. *)
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

    (** [bank_at banks ~address] is which bank of [banks] holds a flat address: the last
        bank whose base the address has reached, which is the top address bits alone. A
        caller carries the answer beside the DATA and not beside the address, and holds it
        as many times as the data is registered before the mux. *)
    val bank_at : bank array -> address:Comb.t -> Comb.t

    (** [ring_address ~pin t ~step ~channel] is where the ring holds a column of Y: the
        store's own map over the low bits of the semantic step. The caller drives the
        SEMANTIC column and this takes the bits the ring keeps, thus no modulo and no
        second counter stands anywhere. It has no software half — no image, gate or
        instrument addresses the ring. *)
    val ring_address
      :  pin:(Comb.t -> Comb.t)
      -> t
      -> step:Comb.t
      -> channel:Comb.t
      -> Comb.t

    (** [layer_of t ~turn ~phase] is which layer of the table a frame is in. Every fact of
        the table is muxed by this and not by the turn. *)
    val layer_of : t -> turn:Comb.t -> phase:Comb.t -> Comb.t

    (** [next_block t ~is_pair ~cin_count ~group_count nest] advances the lead frame's
        nest by one cycle and says whether the turn closed. The counts are the LEAD
        layer's, because it is the lead frame that walks. *)
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
