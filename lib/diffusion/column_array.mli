(** The column array: the lanes of era six, and the chain that carries their sums out.

    The unit is ONE OP SHAPE and nothing else — [rows] by [lanes] multiply-accumulate
    lanes that take one term each cycle, and a chain that drains their accumulators one
    row a cycle. It knows no layer, no memory and no walk. The design is
    [docs/diffusion_rtl.md], "The dwell" and "The drain".

    One memory word is one pitch column — the [rows] of one time step and one channel —
    thus one column feeds every row, one activation serves the [lanes] of a row and one
    weight serves the [rows] of a channel. The broadcast is two trees and never a mesh.

    THE THREE-COLUMN WINDOW STANDS OUTSIDE THIS UNIT, with the store it caches: a window
    is a read cache for the column port, thus what fills it and what stands beyond the
    ends of the roll are the memory's questions. A term names its own column.

    What a caller must know:

    - **[column] and [weights] must hold the term's operands on its [term] cycle.** The
      accumulator is order-free, thus the caller takes the taps and the input channels in
      any order.
    - **[term] is a positive strobe, and a cycle without it changes nothing.** The array
      holds no walk of its own to freeze, thus it takes no [hold]: a caller that must
      pause inside a dwell simply states no term.
    - **Dwells run back to back.** The next dwell's [term_first] may stand on the cycle
      behind the [term_last] of the one before it: the chain takes the value the register
      held BEFORE the edge, which is the finished sum.
    - **BUT TWO CLOSES MUST STAND [rows] CYCLES APART.** A capture reloads every stage,
      thus one that arrives while the chain is still draining throws away the rows that
      have not left. The array cannot refuse it; [Elaboration.create] refuses the layer
      instead, and a test beside the unit states what breaking the rule costs.
    - **A ragged group needs no mask.** The elaboration pads a group that runs past a
      layer's channels with zero bytes, thus those lanes drain zero.
    - There is no clear on the datapath: what is real is what the strobes mark, and the
      DSP packs best with no reset. The drain counter clears. *)

open Hardcaml

(** The shape one instantiation is built for; every width of the interface follows from
    the two numbers. *)
module type Shape = sig
  (** P: the pitch rows of a column, and the row slices of the array *)
  val rows : int

  (** G: the output channels of one group, and the lanes of one row slice *)
  val lanes : int
end

(** {1 The placement family}

    Three rules the whole era states to Vivado through this unit, because this unit holds
    the primitives they are about. *)

(** [no_dsp product] pins a product into LUTs. THE ARRAY OWNS THE DSPS — the fused rung is
    48 by 5, the device's whole 240 — thus every other unit pins its products away from
    them. *)
val no_dsp : Signal.t -> Signal.t

(** [replica copy] states one copy of a broadcast and keeps it apart from its siblings. No
    net of this scale keeps a single driver, thus a broadcast stands as one register for
    each slice of its consumers and every copy is [dont_touch] — so the tools neither
    merge the copies back into one net nor absorb them into what they feed. *)
val replica : Signal.t -> Signal.t

(** the rows one replica slice covers: 8. It is a placement fact of the device and not of
    a model, thus a caller that banks a column of its own slices on this value. *)
val slice_rows : int

(** [slices_for ~rows] is the replica slices a column of [rows] takes:
    [ceil (rows / slice_rows)]. *)
val slices_for : rows:int -> int

module Make (Shape : Shape) : sig
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
          [r + row_shift - 1] of [column], and zeros stand outside the roll. The array
          knows nothing of the kernel's taps — it shifts rows. *)
      ; weights : 'a (** the weight word: [lanes] bytes, lane 0 in the low byte *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { drained : 'a
      (** 1 while [row] and [sums] hold a drained row; it stands for [rows] cycles, from
          four behind [term_last]. The epilogue takes this wire under the same name. *)
      ; row : 'a (** the row [sums] holds; the rows leave in row order, 0 upward *)
      ; sums : 'a
      (** the accumulators of that row: [lanes] int32 sums, lane 0 in the low bits *)
      }
    [@@deriving hardcaml]
  end

  (** [create i] is the block: [rows] by [lanes] lanes, and a chain of [rows] stages
      behind them. It holds no state that outlives a dwell, thus ONE INSTANTIATION SERVES
      EVERY LAYER OF THE MODEL. *)
  val create : Signal.t I.t -> Signal.t O.t
end
