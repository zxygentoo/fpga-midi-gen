(** The MAC and its walk: the multiplier pipe of the sources, one term a cycle.

    The unit walks [outer] rows of [inner] terms. Each cycle it names one term by its
    issue counters [ii] (the term within the row) and [oo] (the row); the caller turns the
    counters into memory addresses, and [read_latency] cycles later presents the term's
    operands on [a] and [b]. Two registers behind the operands stands the product; the
    accumulator behind the product folds a row into [sum]. A tag travels beside each term
    from its address to its retirement, thus the control needs no knowledge of the pipe's
    depth: the row's first tag loads the accumulator, the last raises [row_done]. Rows
    stream back to back with no cycle between.

    THE WALK WIDTH IS THE FUNCTOR'S ARGUMENT, and it is the one place a bigger model shows
    through an otherwise model-free unit: era four's longest walk ran 256 rows and takes
    nine bits, era five's state update walks [d_in * state] rows and takes fourteen. Each
    source instantiates the width its own walks need, thus both netlists stand as their
    boards proved them and neither pays for the other.

    The contract:

    - [go] starts a walk; the counters clear, and the first term issues on the next cycle.
      [go] lands even under [hold] — the caller starts a walk only when the previous walk
      is done, thus nothing real is in flight.
    - [inner] and [outer] are counts, read live: the caller must hold them stable from the
      cycle after [go] to [done_]. At the [go] cycle itself they may be anything.
    - [a] and [b] must carry the operands of the term named [read_latency] cycles earlier.
      The caller aligns every memory it serves the walk from to that latency, and freezes
      those read registers with the same [hold].
    - [hold] freezes the walk: counters, tags, pipe. The caller must freeze its memory
      read registers with the same signal, or the data and the tags fall out of step.
    - [row_done] pulses for one cycle as each row completes; [sum] is whole only in that
      cycle, and [row] names the completed row. [done_] coincides with the last
      [row_done]. The caller's landing logic runs on these pulses.
    - [product] is the free-running product of [a] and [b], two enabled registers behind
      them — the tap for the bespoke chains that pulse the multiplier outside a walk.
      Present the operands, wait one cycle, read [product] on the next.

    The datapath registers have no clear: correctness comes from the tags, and the DSP
    packs best with no reset. The control registers clear. *)

open Hardcaml

(** Cycles from an issued address to the operands at [a]/[b]. Every memory that serves the
    walk reads through two registers: the array's read register, then the output register
    that packs into the block RAM. It does not depend on the width, thus a cost model
    reads it without naming one. *)
val read_latency : int

(** the width of the walk counters of one instantiation *)
module type Width = sig
  val walk_bits : int
end

module Make (W : Width) : sig
  (** [W.walk_bits], re-exported: the era sources size their command wires from it *)
  val walk_bits : int

  val read_latency : int

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; go : 'a (** start a walk: the counters clear, issue begins next cycle *)
      ; inner : 'a (** terms in a row, a count; stable during the walk *)
      ; outer : 'a (** rows in the walk, a count; stable during the walk *)
      ; hold : 'a (** freeze the walk and its tags *)
      ; a : 'a (** operand, 25 bits signed, [read_latency] behind the counters *)
      ; b : 'a (** operand, 18 bits signed, [read_latency] behind the counters *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { ii : 'a (** the term this cycle's addresses must name *)
      ; oo : 'a (** the row this cycle's addresses must name *)
      ; product : 'a (** the free-running product — the bespoke tap *)
      ; sum : 'a (** the finished row's sum; whole only while [row_done] *)
      ; row_done : 'a (** one pulse per completed row *)
      ; row : 'a (** the completed row's index, retire side *)
      ; done_ : 'a (** the last row's [row_done] *)
      }
    [@@deriving hardcaml]
  end

  val create : Signal.t I.t -> Signal.t O.t
end
