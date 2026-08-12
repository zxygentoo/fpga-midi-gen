(** The integer twin: the transformer of the prototype circuit, in exact integer
    arithmetic.

    The circuit of [Vaswani] must match this module bit for bit — the twin is the
    reference of the circuit, as [Pink] is the reference of [Voss]. The float model of
    [Transformer] is not: post-training quantization separates them, and the audition
    judges that distance. [docs/transformer_rtl_proto.md] holds the design: the formats,
    the operations and their order.

    The twin is generic over [Transformer.Config], within the shift rules of the circuit's
    arithmetic: the width and the context are powers of two, and the head width is a power
    of four. The circuit alone fixes one shape — its address packing — and [Vaswani]
    states that check; the king of the era elaborates at [Transformer.Config.baseline].

    The engine is a value: each operation gives the engine after it, thus a state can be
    kept, compared and replayed. *)

(** The design constants of the fixed-point scheme: the formats, and the values derived
    from them and from the mathematics. The circuit elaboration reads these, thus the twin
    and the circuit share one definition. The model dimensions are not here —
    [Transformer.Config] carries them — and neither is the sampling policy, which the
    model carries. *)
module Constants : sig
  (** the residual stream: Q16 in int32 *)
  val h_q : int

  (** the normed vector, the query, the keys, the values and the context: Q12 in int16 *)
  val y_q : int

  val kv_q : int

  (** the feed-forward hidden vector: Q10 in int16 *)
  val hid_q : int

  (** the rms epsilon of the float model, in the Q of the squared stream *)
  val eps_q : int

  (** log2(e) in Q15: the exp2 form of the softmax exponent *)
  val log2e_q15 : int

  (** the exp2 table of the softmax and the sampler as the circuit's ROM: 256 entries of
      [round(2^15 * 2^-(j/256))], 16 bits each — the quantized exponential, the same
      species as the weights *)
  val exp2_bits : Hardcaml.Bits.t array
end

(** The vectors of the file, flat. [t] is the machine's integer vector — the value side of
    a weight and every signal of the engine, each in its own Q format. [floats] is the
    checkpoint side. The drift measures compare the two over one pair of logit vectors;
    [checkpoint_tool drift] reports both over a walk. *)
module Tensor : sig
  type t = int array
  type floats = float array

  (** [same_peak q f] is true when the two vectors elect the same index — the top-1
      agreement; a tie keeps the first. *)
  val same_peak : t -> floats -> bool

  (** [cosine q f] is the cosine between the two vectors. *)
  val cosine : t -> floats -> float
end

(** The quantized model: the configuration and the tensors quantized under it, one value —
    the unit the engine and the circuit consume. The pairing invariant lives in the
    constructors: after them, no caller can mispair a configuration with another model's
    tensors. The three tables share one exponent, because their rows add. *)
module Model : sig
  (** one quantized weight tensor: the int8 values, flat in the row-major order of the
      checkpoint, and the exponent of the power-of-two scale — [w ~ q * 2^-e]. The
      arithmetic that makes one is private: the constructors apply it. *)
  type quantized =
    { q : Tensor.t
    ; e : int
    }

  (** the structure of [Transformer.Params_data], over the quantized tensor: one
      definition holds the shape and the flat order of the checkpoint *)
  type params = quantized Transformer.Params_data.t

  type layer = quantized Transformer.Params_data.layer

  type t =
    { config : Transformer.Config.t
    ; params : params
    ; temper_q14 : int
    (** the sampling temper, log2(e) / T in Q14 — folded with the exp2 form *)
    ; min_weight : int (** the min-p share of the peak weight 2^15 *)
    }

  (** the ROM image of the circuit: every tensor in the checkpoint order, one byte for
      each weight, two's complement, padded to the next power of two — the depth of the
      address *)
  val rom_bits : t -> Hardcaml.Bits.t array

  (** the base of each tensor inside the ROM, in the shape of the parameters — the address
      book of the circuit elaboration *)
  val rom_bases : t -> int Transformer.Params_data.t

  (** [of_checkpoint config path] loads the float checkpoint and quantizes it. It raises
      when the file does not hold the tensors of the shapes of [config]. [temperature] and
      [min_p] set the sampling policy — the machine commits to them, as it commits to the
      weights — and their defaults are the settled values of the era; the rules of the
      float sampler apply to both. *)
  val of_checkpoint
    :  ?temperature:float
    -> ?min_p:float
    -> Transformer.Config.t
    -> string
    -> t

  module For_test : sig
    (** [init config ~seed] is a model of drawn weights in the shapes of [config] — the
        initial parameters of [Transformer.Params.init], quantized: the elaboration and
        the engine need no checkpoint file. *)
    val init : ?temperature:float -> ?min_p:float -> Transformer.Config.t -> seed:int -> t
  end
end

(** The inference engine: the state of one seeded run, and the operations that advance it
    one token at a time. [Model] is what the circuit keeps in ROM; the engine is what it
    keeps in registers and BRAM — the KV ring, the residual stream, the PRNG, the sounding
    state and the seats. One seed names one walk: the same model and the same seed give
    the same events here, in the circuit and on the board. *)
module Engine : sig
  (** one running inference, as a value: an operation gives the engine after it *)
  type t

  (** one socket event of a drawn sentence *)
  type event =
    { voice : int (** the voice that sounds it, 0 to [Token.seats - 1] *)
    ; pitch : int (** the MIDI pitch *)
    ; on : bool (** [true] is Note On, [false] is Note Off *)
    }
  [@@deriving sexp_of]

  (** [init model ~seed] is the engine at its origin: the PRNG at [seed] — the rule of the
      SEED cell, thus 0 raises — the ring empty, the sounding state silent, and START
      forwarded at phase 0, bucket 0. The shape obeys the shift rules of the module
      header, and [init] checks them. *)
  val init : Model.t -> seed:int -> t

  (** [next_step t] draws one sentence: the engine after it, and its socket events in the
      drawn order. An On takes the highest free seat; an Off names the seat that holds its
      pitch. The END that closes the sentence is forwarded and not reported. *)
  val next_step : t -> t * event list

  (** The block-level interface below: the tests and the drift report drive the engine one
      operation at a time; [next_step] is the interface of the players. *)

  (** [logits t] is the Q12 logits of the position after the last forwarded token. *)
  val logits : t -> Tensor.t

  (** [next_code t] draws the next token code — the mask of the sounding state, the temper
      and min-p of the model, then three PRNG bytes pick from the weights — and gives the
      engine after the draw. *)
  val next_code : t -> t * int

  (** [forward t ~code ~phase ~bucket] runs the engine over one token — the forward pass —
      and gives the engine after it. The sounding state steps with the token, thus the
      mask of the next draw can never run ahead of or behind the engine. [code] is the
      token; [phase] and [bucket] are the rows of the bar-phase and the piece-position
      tables. *)
  val forward : t -> code:int -> phase:int -> bucket:int -> t
end
