(** The restoring square root: the floor of the square root of an unsigned value.

    One radicand bit pair a cycle, from the top. The root grows a bit at a time, and the
    trial subtrahend is the root so far with a 1 under it. Therefore the root is the
    floor, as the reference square root is.

    The contract:

    - [start] loads [value] and begins the walk. It wins over a walk in flight: a [start]
      in a busy cycle discards that walk and begins a new one.
    - [busy] reads 1 in the cycle after [start], and it reads 0 again 21 cycles later —
      one cycle for each bit pair of the radicand. [root] is whole in the cycle [busy]
      reads 0, and it stands until the next [start]. Therefore the caller waits on [busy]
      and reads the result in the cycle the wait releases.
    - [value] is read in the [start] cycle only, and the caller may move it after. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a (** a strobe: load the radicand and begin the walk *)
    ; value : 'a (** the radicand, 42 bits, unsigned *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { root : 'a (** 21 bits, unsigned; whole in the cycle [busy] reads 0 *)
    ; busy : 'a (** a walk is in flight *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
