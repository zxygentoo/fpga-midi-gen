(** What every Cyclesim bench of the repository mounts its circuit in: the port moves, the
    node lookup, and the tally of a state machine's cycles.

    It stands here, in the core library, and not in the library of one era: a bench is a
    bench in every era, and the era that copied these would be the era whose bench drifted
    from the one before it. The draws a bench takes stay with the generator, in
    [Prng.For_test]; a bench draws its values there and puts them on a port here.

    The trace is not here because Hardcaml holds it: [Cyclesim.Waveform.create_if] is the
    recorder a picture asks for and a long gate does not. *)

open Hardcaml

(** [set port value] writes [value] into a simulation port at the width the shape gave it,
    thus a bench states a number and never a width. *)
val set : Bits.t ref -> int -> unit

(** [pack values ~width] is the values as one word, value 0 in the low bits and each one
    signed at [width] bits — the shape a wide port takes, thus a bench states the port and
    never the slicing. *)
val pack : int array -> width:int -> Bits.t

(** [unpack word ~width] is the inverse of [pack]: the signed values a wide port carries,
    the low [width] bits first. A column as the twin holds it reads out of a port with
    this, thus a gate reads a column and never an index. *)
val unpack : Bits.t -> width:int -> int array

(** The configuration a LONG bench mounts its circuit under: the named signals traced, and
    the simulator's own deduplication pass.

    [`All_named] reaches every signal a probe looks up and every signal a waveform shows —
    a bench of thousands of cycles over a wide array cannot afford [`Everything]. The
    deduplication shares structurally equal signals before the compile, thus a lane that
    states the same term as its neighbour is evaluated one time; measured 2026-08-28, it
    took era six's library gates from 38.5 s to 31.2 s and no cycle count moved. It is a
    SIMULATOR setting — [Rtl.output] never sees this record — thus no netlist follows it. *)
val long_bench : Cyclesim.Config.t

(** [verdict complaints] is what a bench says its run disagreed about: ["ok"] when every
    check held, and the names of the checks that did not, separated by commas. A check is
    a pair of the name and whether it went WRONG, thus a bench states its checks in one
    list and never an if for each of them. *)
val verdict : (string * bool) list -> string

(** [node sim name] is the traced signal a circuit named [name], whether it is a node or a
    register. It raises with the name when the trace does not hold it, thus a rename
    cannot silently blind a probe. *)
val node : _ Cyclesim.t -> string -> Cyclesim.Node.t

(** the states of an [Always] state machine, as [State_machine.create] takes them *)
module type Enumerated = sig
  type t

  val all : t list
  val compare : t -> t -> int
end

(** A tally of the cycles a state machine spends in each of its states, read off its state
    register.

    THE ENCODING IS THE INDEX IN [all]: Hardcaml's binary encoding states a state as its
    position in the enumeration, thus the register's value indexes the tally directly and
    the bench decodes nothing. A value the enumeration does not hold raises, and never
    reads as a small error. *)
module Tally : sig
  type 'state t

  (** [create (module State)] is the empty tally over [State.all]. *)
  val create : (module Enumerated with type t = 'state) -> 'state t

  (** [encoded t state] is the value the register holds in [state]: its index in [all]. A
      bench that carries the reading itself compares against this. *)
  val encoded : 'state t -> 'state -> int

  (** [count t ~encoded ~cycle] adds one cycle to the state the register holds as
      [encoded]; [cycle] is the bench's own count, and the first one a state is counted at
      is what [entered] gives. *)
  val count : 'state t -> encoded:int -> cycle:int -> unit

  (** [clear t] empties the tally, for a bench that runs more than one walk. *)
  val clear : 'state t -> unit

  (** the cycles counted in one state *)
  val spent : 'state t -> 'state -> int

  (** the first cycle a state was counted at, or [None] while it never was *)
  val entered : 'state t -> 'state -> int option
end
