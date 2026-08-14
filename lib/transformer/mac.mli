(** The MAC and its walk: the multiplier pipe of the source, one term a cycle.

    The unit walks [outer] rows of [inner] terms. Each cycle it names one term by its
    issue counters [ii] (the term within the row) and [oo] (the row); the caller turns the
    counters into memory addresses, and [read_latency] cycles later presents the term's
    operands on [a] and [b]. Two registers behind the operands stands the product; the
    accumulator behind the product folds a row into [sum]. A tag travels beside each term
    from its address to its retirement, thus the control needs no knowledge of the pipe's
    depth: the row's first tag loads the accumulator, the last raises [row_done]. Rows
    stream back to back with no cycle between.

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
    that packs into the block RAM. *)
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
