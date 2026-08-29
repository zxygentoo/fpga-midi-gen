(** THE DRAW OF THE CHAIN: the four ops that turn a row of logits into a drawn class, and
    the seat registers they write.

    The step-frame eras draw a frame one seat at a time, from the soprano down, and the
    four ops of a seat are one text: [tempered_weights] writes the Q15 weight of every
    class, [uniform_word] takes three bytes of the generator, [threshold] scales that
    uniform by the total, and [pick] walks the classes until the running total passes it.
    That text stood twice, under two module names; it stands here.

    THIS IS NOT A UNIT. [Exp2], [Sigmoid], [Mac] and the others in this library are
    circuits with ports of their own, and a caller wires them. The sampler is not: it
    reads the era's SHARED value RAM and the era's ONE multiplier, and a version with
    ports of its own would need a second multiplier — a second DSP, which moves the
    netlist and the resource story. What stands here is CODE: builders of [Always]
    statements over variables the era declares. The two case forms these are written in
    are L3's idiom and not the draw's: they stand in [Program].

    NOTHING HERE DECLARES A REGISTER OR A WIRE, and that is a rule and not an accident.
    Hardcaml names an unnamed signal [signal_<op>_N] by the ORDER OF ITS CREATION, thus a
    declaration moved into this module would create its signal at another point of the
    elaboration and rename every signal after it. An era declares its own and passes them
    in through [t]. For the same reason a term the era's text wrote out more than once is
    written out more than once here: naming it would be one node where the era made
    several.

    The tick positions inside [exp_weight_chain] hand-encode the latency of the multiply
    and of the [Exp2] read. THEY ARE A DORMANT DEBT: if the pipe ever deepens, replace
    them with a valid bit — never renumber them. *)

open Hardcaml

(** The ports the draw reads. Every field is a variable or a signal the ERA declared: this
    record is a view of the era's datapath and holds nothing of its own.

    It is large, and the size is the honest cost of the lift — the draw touches the shared
    RAM, the one multiplier, the generator, the [Exp2] answer and eleven registers, and a
    shared text must name each of them. *)
type t =
  { classes : int (** the class count of the vocabulary the era draws over *)
  ; temper : Quantized.Constants.scale (** log2(e) / T, as the model states it *)
  ; min_weight : int (** the min-p floor, as a share of the peak weight 2^15 *)
  ; tick : Always.Variable.t (** the position inside a bespoke chain *)
  ; oo : Always.Variable.t (** the output walk: here, the class *)
  ; u24 : Always.Variable.t (** the 24-bit uniform, three bytes shifted in *)
  ; total : Always.Variable.t (** the sum of the tempered weights *)
  ; thi : Always.Variable.t (** the high pass of the threshold multiply *)
  ; thr : Always.Variable.t (** the threshold the pick walks against *)
  ; cum : Always.Variable.t (** the running total of the pick *)
  ; found : Always.Variable.t (** set once the pick has landed *)
  ; diff : Always.Variable.t (** a value's distance below the peak *)
  ; nn : Always.Variable.t (** the argument of the [Exp2] unit *)
  ; vram_raddr : Always.Variable.t (** the shared value RAM: its width states [vbits] *)
  ; vram_wen : Always.Variable.t
  ; vram_waddr : Always.Variable.t
  ; vram_wdata : Always.Variable.t
  ; vramd : Signal.t (** what the value RAM read *)
  ; below_peak : Signal.t
  (** [vramd - peak], NAMED BY THE ERA: an era runs the weight chain from more than one
      op, and a term written inside the chain would be elaborated once for each of them *)
  ; mul_a : Always.Variable.t (** the one multiplier of the era *)
  ; mul_b : Always.Variable.t
  ; product : Signal.t
  ; prng_step : Always.Variable.t (** the generator's advance, and the byte it gives *)
  ; prng_byte : Signal.t
  ; exp2_e : Signal.t (** what the [Exp2] unit answered *)
  ; weight_addr : Signal.t
  (** the class the weight walk is at, as the era slices it: the two eras cut [oo] to
      different widths and each states its own *)
  ; oo_class : Signal.t (** the same walk at the width a drawn class takes *)
  ; write_drawn : Signal.t -> Always.t
  (** the write of the seat register the chain is at — [write_drawn] below builds it *)
  }

(** [exp_weight_chain t ~addr ~scale ~land_ ~advance] turns one value of the shared RAM
    into its exp2 weight, over the same address: read the value, take its distance below
    the peak, scale that into the exp2 argument, and land the weight where the value
    stood. [land_] is what the caller writes beside the weight and [advance] is how the
    walk moves. An era's softmax and its temper are two callers of this one chain. *)
val exp_weight_chain
  :  t
  -> addr:Signal.t
  -> scale:Quantized.Constants.scale
  -> land_:Always.t list
  -> advance:Always.t list
  -> Always.t list

(** The four ops of the draw. Each gives [entry, body] as an era's own compiler takes
    them: [entry] is what the op before it runs, and [body] is what runs while it is the
    program counter. [finish] is the entry of the op that follows. *)

(** the tempered weight of each class: exp2, and refused under min-p *)
val tempered_weights : t -> finish:Always.t list -> Always.t list * Always.t list

(** three PRNG bytes, high first: the walk of [Prng.uniform] *)
val uniform_word : t -> finish:Always.t list -> Always.t list * Always.t list

(** [(u24 * total) >> 24] in two DSP passes: the high twelve bits of the total, then the
    low twelve — the same integer as one wide multiply *)
val threshold : t -> finish:Always.t list -> Always.t list * Always.t list

(** The first class whose running total passes the threshold, and the last class catches a
    walk that no weight stopped — which is the rule of the reference and not a fallback:
    the threshold is below the total by construction, thus the class the walk names always
    holds weight. The class lands in the register of its seat. *)
val pick : t -> finish:Always.t list -> Always.t list * Always.t list

(** [drawn_at_seat ~seat ~drawn] is the class the seat the chain is at has drawn *)
val drawn_at_seat : seat:Always.Variable.t -> drawn:Always.Variable.t array -> Signal.t

(** [write_drawn ~seat ~seats value] writes [value] into the register of the seat the
    chain is at — the other half of that seat's port, as one parallel case *)
val write_drawn
  :  seat:Always.Variable.t
  -> seats:Always.Variable.t array
  -> Signal.t
  -> Always.t

(** [frame_word ~code_of_class ~drawn] is the frame word of the classes the chain drew:
    the caller states the class-to-code map, and seat 0 takes the low byte *)
val frame_word
  :  code_of_class:(Signal.t -> Signal.t)
  -> drawn:Always.Variable.t array
  -> Signal.t
