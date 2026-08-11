(** The board button: the synchronizer, the debounce and the toggle strobe.

    Two flip-flops synchronize the raw pin. The debounced level changes when the
    synchronized input holds the new level for [debounce_clocks] cycles, and [toggle]
    strobes at each rising edge of the debounced level. A bounce shorter than the window
    moves nothing.

    On the board [debounce_clocks] is 10 ms of the 100 MHz clock; the simulation gives a
    small value. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; button : 'a (** the raw pin *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { toggle : 'a (** a strobe at each push *) } [@@deriving hardcaml]
end

val create : debounce_clocks:int -> Signal.t I.t -> Signal.t O.t
