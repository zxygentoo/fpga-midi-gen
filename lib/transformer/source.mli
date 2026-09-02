(** The transformer as a circuit: the note source of era four.

    The block computes the integers of the JAX twin, operation for operation.
    [jax/transformer/quantized/infer.py] is the reference, and the gates of
    [jax/tests/test_rtl_transformer.py] prove the match through
    [bin/gate_transformer.exe]: the frames of the circuit must equal the frames of the
    twin, step for step. Neither side can pass a gate by agreeing with itself.

    One step of music is one pass of the network and one frame on the socket. Nothing here
    chooses a seat and nothing masks a draw: a frame states which voice holds which pitch,
    thus the grammar of the instrument needs no registers and no frame is illegal. This is
    the first source of the frame socket, and [Source_intf] states that contract.

    The shape, in five layers. L0 is the primitives — [Divider], [Isqrt], [Exp2] here, and
    [Prng.Rtl] in the core. L1 is the datapath: the RAMs, the KV rings, the banked weight
    ROM, and [Mac] behind the one multiplier. L2 is the schedule, a value: the forward
    pass and the chain of the four seats are lists of operations, and one operation holds
    the facts of one step. L3 compiles those lists into cases of a program counter. L4 is
    a small outer machine that holds the drawn frame and answers the socket. source.ml
    holds the design and its reasons, and [docs/transformer_rtl.md] holds the design of
    the whole.

    What a caller must know:

    - The weights arrive at elaboration — the bitstream carries them — thus [create] takes
      the whole model. The seed arrives live from the SEED cell, and a [rewind] captures
      it, as the pink era does.
    - **The source answers [step] from a frame it has already drawn.** It states the frame
      at once, runs the pass of that frame, and then draws the frame of the next step,
      thus the tempo waits for the wire and never for the network. The first bar is the
      lead-in of the model: every frame of it is silence, nothing is drawn, and the
      generator does not move.
    - Cycle counts are not a contract. The socket is latency-insensitive, and the PRNG
      steps only on command, thus a draw cannot move when the timing of the circuit
      changes. The cost model beside the schedule states the cost of a step exactly, and a
      bench pins the model to the circuit.
    - There is one reset, and it is [rewind]: it loads the PRNG and returns the walk to
      its origin. It runs no pass, thus [idle] never falls for it. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [create ~model ~seed i] is the block. [model] is the model of the CONTRACT FILE: the
    ROM image, the tensor bases, every exponent and every shape number come from it, and
    they decide the width of every counter and address. The elaboration checks the shape
    rules of the packing loudly. [seed] is the 32-bit seed of the walk, read at [rewind],
    thus one seed names one sequence in the simulation and on the board. *)
val create : model:Model.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t

module For_test : sig
  (** The walk, driven. [bin/gate_transformer.exe walk] runs it and prints what the
      circuit answered, and [jax/tests/test_rtl_transformer.py] states what it must have
      answered against the twin. The two sides cannot agree with each other by accident:
      the driver knows nothing of the twin, and the twin runs no circuit. *)
  module Bench : sig
    type t =
      { rewind : unit -> unit (** load the generator and return the walk to its origin *)
      ; play : unit -> int
      (** strobe one step and give the frame it answers. It raises [Failure] when the step
          is not answered, or when the walk runs past the budget the cost model states — a
          machine that stalls must fail a gate and not hang it. *)
      }

    (** [harness ~model ~seed ()] builds the simulation and its probes *)
    val harness : model:Model.t -> seed:int -> unit -> t
  end
end
