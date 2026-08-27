(** The column array: the lanes of era six, and the chain that carries their sums out.

    L2 of the diffusion source. The unit is ONE OP SHAPE and nothing else — [rows] by
    [lanes] multiply-accumulate lanes that take one term each cycle, and a chain that
    drains their accumulators one row a cycle. It knows no layer, no memory and no walk:
    the caller states the terms and the array states the drained rows.

    The design is [docs/diffusion_rtl.md], "The dwell" and "The drain". One memory word is
    one pitch column — the [rows] of one time step and one channel — thus one column feeds
    every row, one activation serves the [lanes] of a row and one weight serves the [rows]
    of a channel. The broadcast is two trees and never a mesh.

    **THE THREE-COLUMN WINDOW STANDS OUTSIDE THIS UNIT**, with the store it caches: a term
    names its own column. A window is a read cache for the column port, thus what fills it
    and what the zero column is beyond the ends of the roll are the memory's questions and
    not this unit's. The path is the same either way — the caller's window register, its
    time mux, this unit's row shift, the operand register — thus the cut costs no logic
    and no stage.

    What a caller must know:

    - **[column] and [weights] must hold the term's operands on its [term] cycle.** The
      accumulator is order-free — the twin's gift to this round — thus the caller takes
      the taps and the input channels in any order.
    - **[term] is a positive strobe, and a cycle without it changes nothing.** The array
      holds no walk of its own to freeze, thus it takes no [hold]: what it has is a stream
      of terms, and a caller that must pause inside a dwell simply states no term. A
      free-running accumulator would read a pause as arithmetic.
    - **Dwells run back to back.** The next dwell's [term_first] may stand on the cycle
      behind the [term_last] of the one before it: that term reaches the accumulator on
      the very edge that loads the chain, and the chain takes the value the register held
      BEFORE the edge, which is the finished sum. The array never asks for a gap.
    - **BUT TWO CLOSES MUST STAND [rows] CYCLES APART.** A capture reloads every stage,
      thus one that arrives while the chain is still draining throws away the rows that
      have not left. The array cannot refuse it; [Elaboration.create] refuses the layer
      instead — a dwell of [9 * inputs] cycles against a chain of [rows] stages — and a
      test beside the unit states what breaking the rule costs.
    - **A ragged group needs no mask.** The elaboration pads a group that runs past a
      layer's channels with zero bytes, thus those lanes drain zero and the caller ignores
      them.
    - There is no clear on the datapath: what is real is what the strobes mark, and the
      DSP packs best with no reset. The drain counter clears. *)

open Hardcaml

(** The shape one instantiation is built for. Every width of the interface follows from
    the two numbers, thus a test runs the whole unit at a shape a simulation can hold. *)
module type Shape = sig
  (** P: the pitch rows of a column, and the row slices of the array *)
  val rows : int

  (** G: the output channels of one group, and the lanes of one row slice *)
  val lanes : int
end

(** [no_dsp product] pins a product into LUTs.

    THE ARRAY OWNS THE DSPS AND EVERY OTHER UNIT PINS ITS PRODUCTS AWAY FROM THEM: the
    fused rung at G 5 is 48 by 5, which is the device's whole 240, thus a multiply that
    drifted into a DSP elsewhere would have to move at the moment the design is tightest.
    The rule stands here because this unit is the one that holds the primitives — the
    epilogue, the draw and the walk each state it to Vivado through this function. *)
val no_dsp : Signal.t -> Signal.t

(** the rows one replica slice covers: 8.

    Ring 3's rule, and the array's own scale is what imposes it — no net of this scale
    keeps a single driver, thus a bank stands as one [dont_touch] register slice for each
    [slice_rows] rows. It is a placement fact of the device and not of a model, thus a
    caller that banks a column of its own ([Forward]'s window and bands) slices on this
    value rather than on an 8 of its own. *)
val slice_rows : int

(** [slices_for ~rows] is the replica slices a column of [rows] takes:
    [ceil (rows / slice_rows)]. *)
val slices_for : rows:int -> int

(** Cycles from a [term] strobe to the edge that takes its term into the accumulator. The
    DSP48E1 packs as an operand pair, a product and a sum, thus the depth is the
    primitive's and not a choice. *)
val accumulate_latency : int

(** Cycles from [term_last] to the first [drained]: the accumulator settles one cycle
    behind its last term, the chain takes it on that cycle's edge, and the bottom stage
    stands one cycle later. *)
val first_row_latency : int

module Make (Shape : Shape) : sig
  (** [Shape.rows], re-exported: a caller sizes its row counter and its drain from it *)
  val rows : int

  (** [Shape.lanes], re-exported: a caller sizes its weight word from it *)
  val lanes : int

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; term : 'a (** a strobe: this cycle carries one term of the dwell *)
      ; term_first : 'a
      (** the term opens the dwell, thus the accumulator loads it and does not add it *)
      ; term_last : 'a (** the term closes the dwell, thus the chain takes the sums *)
      ; column : 'a
      (** the pitch column this term reads: [rows] activations, row 0 in the low bits. The
          caller's window holds the three time taps and states the one this term takes. *)
      ; row_shift : 'a
      (** which of three rows this term takes: row [r] of the output reads row
          [r + row_shift - 1] of [column] — 0 the row below, 1 the row itself, 2 the row
          above — and zeros stand outside the roll. The array knows nothing of the
          kernel's taps: it shifts rows, and the caller's tap index states the shift. *)
      ; weights : 'a (** the weight word: [lanes] bytes, lane 0 in the low byte *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { drained : 'a
      (** 1 while [row] and [sums] hold a drained row; it stands for [rows] cycles from
          [first_row_latency] behind [term_last]. The epilogue takes this wire under the
          same name, thus one strobe is one word across the seam. *)
      ; row : 'a (** the row [sums] holds; the rows leave in row order, 0 upward *)
      ; sums : 'a
      (** the accumulators of that row: [lanes] int32 sums, lane 0 in the low bits *)
      }
    [@@deriving hardcaml]
  end

  (** [create i] is the block: [rows] by [lanes] lanes, and a chain of [rows] stages
      behind them. It holds no configuration and no state that outlives a dwell — the
      terms state everything — thus ONE INSTANTIATION SERVES EVERY LAYER OF THE MODEL, and
      the layer table decides only what the caller feeds it. *)
  val create : Signal.t I.t -> Signal.t O.t
end
