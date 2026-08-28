(** The model as the circuit reads it: the formats, the contract file and the ROM.

    Era four draws one frame of four voice codes for each step of music. The four classes
    of a step enter through four tied tables that sum, and they leave through the same
    tables in a chain from the soprano down; the network under that head is a decoder with
    no bias terms, RMSNorm before each sublayer, ALiBi for the position and [d_ff = 4 d].
    The design of the model is [docs/transformer.md] and the design of the circuit is
    [docs/transformer_rtl.md].

    **NO MODEL IS COMPUTED HERE.** The float model is [jax/transformer/model.py] and the
    integer twin is [jax/transformer/quantized.py]; [jax/tests/test_rtl_transformer.py]
    states what the circuit must do and [bin/gate_transformer.ml] states what it did. What
    stays here is what the CIRCUIT reads, and every one of those facts is a rule the RTL
    must equal rather than restate: the fixed-point formats, the model as data, its ROM
    image and the bases of that image.

    The shape numbers are fields of the model because the ELABORATION reads a file and no
    flag. The width and the layer count are in the tensors; the heads, the context and the
    ALiBi span are not — the heads only split the width at run time, ALiBi holds no
    position table, and the context is a choice of the draw — thus they travel in the
    contract file, where [jax/transformer/infer.py quantize] puts them. *)

(** The fixed-point formats of the machine, and the constants that cross between the twin
    and the circuit. A Q number holds [value * 2^-q]. What is one thing across the eras
    comes from [Mgen_nn.Quantized.Constants]; what stands here is era four's own. *)
module Constants : sig
  include module type of Mgen_nn.Quantized.Constants

  (** the query, the keys, the values and the context: Q12 in int16. It is a name of its
      own because the rings store these rows and the ring is where the format is a design
      choice, not an accident of the datapath. *)
  val kv_q : int

  (** [score_shift ~head_d] carries a score walk's sum from Q(2 [kv_q]) to Q[y_q] and
      applies the 1/sqrt([head_d]) of the model in the same shift, thus [head_d] is a
      power of four. *)
  val score_shift : head_d:int -> int
end

(** The parameter structure over any type, and the flat order of the checkpoint with it:
    the two tables, then six tensors for each layer. ONE DEFINITION HOLDS THE ORDER, and
    the contract file, the ROM image and the bases are three readings of it. *)
module Params_data : sig
  type 'a t =
    { seats : 'a (** the four tied tables in one tensor, seat 0 first *)
    ; phase : 'a (** the bar-phase table *)
    ; layers : 'a layer array
    }

  and 'a layer =
    { wq : 'a
    ; wk : 'a
    ; wv : 'a
    ; wo : 'a
    ; w1 : 'a
    ; w2 : 'a
    }

  (** the flat order of the tensors — the order of the checkpoint, of the contract file
      and of the ROM; [of_list] reads the same order *)
  val to_list : 'a t -> 'a list

  (** [of_list ~layers items] reads the order of [to_list]. It raises [Invalid_argument]
      when the count of items is not two tables and six for each of [layers]. *)
  val of_list : layers:int -> 'a list -> 'a t
end

(** one quantized tensor: the int8 values flat in the row-major order of the checkpoint,
    and the exponent that reads them — [Mgen_nn.Quantized.quantized] *)
type quantized = Mgen_nn.Quantized.quantized =
  { q : int array
  ; e : int
  }

(** the whole model the bitstream carries *)
type t =
  { d : int (** the width of the residual stream *)
  ; heads : int (** they split the width at run time, thus [d] divides by them *)
  ; context : int (** the attention window, in steps *)
  ; slope_span : int
  (** the ALiBi exponent span: the slope of head k is 2^-(span (k+1) / heads) *)
  ; params : quantized Params_data.t
  ; temper : Constants.scale
  (** the sampling temper, [log2e / T]: part of the model, because the bitstream carries
      it *)
  ; min_weight : int (** the min-p share of the peak weight 2^15 *)
  }

(** the layer count: the tensors state it *)
val layers : t -> int

(** [check_shape t] raises when the model breaks a rule its consumers assume: [d] and the
    context are powers of two, the heads divide [d], the head width is a power of four,
    the seat table holds one row for each seat and class, every tensor holds its count,
    and the seat and phase tables share one exponent. The record is open, thus a model no
    constructor here made can break a rule; the elaboration calls this, where a bad shape
    must fail loudly. *)
val check_shape : t -> unit

(** [of_int8_checkpoint path] is the model of one CONTRACT FILE — the quantized model that
    [jax/transformer/quantized.py] writes, and the only thing that crosses the seam for a
    build. The quantization happens above the seam, one time, thus this reader quantizes
    nothing: it takes the tensors, the exponents, the shape numbers, the temper and the
    min-p share as they stand, and [check_shape] holds every rule the consumers assume.

    The layout is that module's docstring, and two of its facts are facts of THIS reader:
    every tensor is int32, because [Nx_io] skips every dtype it does not hold; and every
    scalar travels as a named tensor, because [Nx_io] gives no access to [__metadata__].

    It raises [Invalid_argument] when the tensor count does not divide into layers, when a
    tensor is missing, or when a shape or a rule does not hold. *)
val of_int8_checkpoint : string -> t

(** the ROM image of the circuit: every tensor in the flat order, one byte for each
    weight, two's complement *)
val rom_bits : t -> Hardcaml.Bits.t array

(** the base of each tensor inside the ROM, in the shape of the parameters — the address
    the circuit adds its own offsets to. The four seat tables stand inside one tensor,
    thus seat [s] begins at [seats + s * classes * d], which is a shift and an add. *)
val rom_bases : t -> int Params_data.t

(** [coarse_to_ring row] is what the KV ring keeps of a Q12 row: the top byte, with eight
    zero low bits restored at the read. The circuit stores eight bits, thus the
    granularity is 2^-4 and the format stays Q12. The query does not pass here — only the
    stored rows coarsen. *)
val coarse_to_ring : int array -> int array

module For_test : sig
  (** the shape numbers of a drawn model, without the tensors that carry them *)
  type shape =
    { d : int
    ; layers : int
    ; heads : int
    ; context : int
    ; slope_span : int
    }

  (** the shape of a test model: small enough to run in a simulation, and the same
      structure as the model of the era *)
  val shape : shape

  (** the shape the ear elected on 2026-08-18 — d 64, six layers, four heads, context 256,
      span 4 — which the flash carried until era six took it. The cost model of a real
      step is read at this shape, and no gate elaborates it. *)
  val elected : shape

  (** [drawn shape ~seed] is a model of drawn weights in [shape]: the elaboration of a
      circuit takes one, thus a test reads no checkpoint and no file that git ignores. The
      weights are not the weights the trainer draws from the same number — only a trained
      checkpoint crosses that seam.

      IT QUANTIZES NOTHING. A quantizer picks an exponent from a tensor's own peak, and
      that is a rule of a CHECKPOINT: it lives above the seam with
      [jax/transformer/quantized.py], which is the only thing that reads one. This draw
      states ONE exponent for every tensor and rounds the normal at it, as era six's does,
      thus the seat and phase tables share an exponent by construction and [check_shape]'s
      rule about them costs nothing to keep. The temper and the min-p floor are stated the
      same way, from [Mgen_nn.Quantized.Constants.temper_at_one] and the elected floor.

      The draw and the rule are what every expect test of this library and of the socket
      simulation has recorded, thus neither may move: a moved byte here would move every
      state table, every cycle bench and every frame, and none of those movements would
      name its cause. *)
  val drawn : shape -> seed:int -> t
end
