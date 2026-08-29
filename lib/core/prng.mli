(** The PRNG: Marsaglia xorshift32, in software and as a circuit.

    One step is three shift-and-XOR layers. The software gives the reference, the circuit
    computes the same recurrence combinationally, and the vector test in this module
    drives the two side by side. Thus the definition of the walk has one home.

    The state walks all 32-bit values except 0, and 0 is the fixed point: the recurrence
    carries it to itself for ever. The board can state that seed — the slide switches set
    it and [docs/seed_switches_rtl.md] states why the design accepts it — thus [create]
    takes it and the walk stands still, which is what the circuit does with it.

    A draw carries the state from one draw to the next, and [run] gives it a state to
    start from. [let*] sequences the draws, and the order of the bindings is the order of
    the walk. Therefore one seed names one sequence, in the software, in the simulation
    and on the board.

    A draw comes out as a plain float or float array. This module owns the randomness
    alone; a caller that wants a tensor gives the array a shape with [Nx.create]. *)

open Hardcaml

(** the state of the walk: 32 bits *)
type state

(** a draw, and the state it carries to the draw that follows *)
type 'a t

(** [create ~seed] is the walk that starts at [seed]. It raises [Invalid_argument] when
    [seed] does not fit 32 bits. A seed of 0 gives the walk that stands still, thus the
    reference plays what the board plays for that seed. *)
val create : seed:int -> state

(** [create_folded ~seed] is the walk that [seed] names, for any integer. The fold
    squeezes the seed into 32 bits, and 0 takes the top state. A seed already inside the
    range names itself, thus [create_folded ~seed:7] is the walk of the board's seed 7.
    Two seeds can fold together; this is harmless, because a seed only names a walk.

    **0 is the one seed that this does not carry to the board.** [create ~seed:0] stands
    still, which is what the circuit does; the fold gives a walk instead, because a tool
    that draws its seed from a flag, a counter or a stream must not get a dead walk from
    the value those give when nobody chose one. Take [create] where the seed is the seed
    of a piece, and this where the seed only has to name a walk. *)
val create_folded : seed:int -> state

val return : 'a -> 'a t
val ( let* ) : 'a t -> ('a -> 'b t) -> 'b t
val ( let+ ) : 'a t -> ('a -> 'b) -> 'b t

(** [all draws] runs the draws from left to right, thus the list holds them in the order
    of the walk. *)
val all : 'a t list -> 'a list t

(** one step, and the draw of that step: the low 8 bits of the new state, 0 to 255. This
    is the draw the circuit gives, thus a walk of [next] and a walk of [Rtl] agree byte
    for byte. *)
val next : int t

(** the bits of one step's draw: 8 *)
val byte_bits : int

(** the steps one uniform takes: 3. A circuit that assembles a uniform a byte a cycle
    counts them. *)
val uniform_bytes : int

(** the bits of the grid a uniform stands on: [uniform_bytes * byte_bits], 24.

    THE GRID HAS ONE HOME. A threshold, an anneal table and a draw port are all sized on
    it, and each reads the width here rather than states 24 again. *)
val uniform_bits : int

(** [uniform_bytes] steps, and the word they make: the first byte highest, a value on
    [0, 2 ** uniform_bits). It is the uniform AS THE CIRCUITS TAKE IT — a walk hands its
    draw a word and never a float — thus [uniform] is this word on the grid, and the two
    cannot part. *)
val uniform_word : int t

(** [uniform_word] on the grid: one draw of the uniform distribution over \[0, 1). The
    three bytes make a grid of 2 ** -24, which is fine enough to hold the tail of a
    Box-Muller draw that one byte would cut at 3.3 sigma.

    The grid holds 0. Therefore a caller that takes the logarithm of a draw must hold that
    case. *)
val uniform : float t

(** [normals ~count ~scale] is [count] draws of the normal distribution whose mean is 0
    and whose standard deviation is [scale]. Box-Muller makes them, thus one draw takes
    two uniforms and six steps of the walk. A count of 0 draws nothing and gives an empty
    array; a count below 0 raises [Invalid_argument]. *)
val normals : count:int -> scale:float -> float array t

(** An independent walk, drawn from this one — four steps make its 32 bits. Give one to a
    part of a computation that must draw on its own: it then holds no place in the order
    of the parent, thus the parent can gain or lose draws without moving it. *)
val split : state t

(** [run draw state] is the state after [draw], and the value of [draw]. *)
val run : 'a t -> state -> state * 'a

(** The circuit: the same recurrence, one step in one cycle. *)
module Rtl : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a (** the state takes 1 *)
      ; load : 'a (** a strobe: the state takes [seed] *)
      ; seed : 'a (** 32 bits *)
      ; step : 'a (** a strobe: the state advances one time *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { value : 'a (** the 32-bit state; a draw is the low 8 bits *) }
    [@@deriving hardcaml]
  end

  (** [create i] is the block. The state register takes the new value at a [step] strobe
      and holds between the strobes. The clear puts 1 into the state: the state has no use
      before the first [load], and 1 walks where 0 stands still. A [seed] of 0 is another
      question — the panel can set it and the walk then stands still by design. [load]
      wins over [step] in the same cycle. *)
  val create : Signal.t I.t -> Signal.t O.t
end

(** What a seeded gate needs: the draws it takes.

    Every random thing in this repository comes from this module and the seed is an input,
    thus a fuzz that finds a fault finds it again. The draws carry the state as the rest
    of the module does, and a caller sequences them by threading it; they map the grid of
    [uniform] onto a range and never take a modulo of a byte. The moves that put a drawn
    value on a simulation port stand in [Harness], beside the rest of the bench.

    It stands here, beside the generator, and not in the library of one era: a gate is a
    gate in every era, and the era that copied these would be the era whose fuzz drifted
    from the one before it. *)
module For_test : sig
  (** [draw state ~limit] is one value in [-limit, limit] and the state behind it. *)
  val draw : state -> limit:int -> state * int

  (** [draw_between state ~low ~high] is one value in [low, high], both ends included. *)
  val draw_between : state -> low:int -> high:int -> state * int

  (** [draw_array state ~len ~limit] is [len] values of [draw], in the order the generator
      states them. *)
  val draw_array : state -> len:int -> limit:int -> state * int array
end
