(** The state-space model as a circuit: the note source of era five.

    The block computes the integers of the JAX twin, operation for operation.
    [jax/mamba/quantized.py] is the reference, and the gates of
    [jax/tests/test_rtl_mamba.py] prove the match through [bin/gate_mamba.exe]: the frames
    of the circuit must equal the frames of the twin, step for step, and every write of
    the h RAM must equal the twin's own. Neither side can pass a gate by agreeing with
    itself.

    One step of music is one step of the recurrence and one frame on the socket. Nothing
    here chooses a seat and nothing masks a draw: a frame states which voice holds which
    pitch, thus the grammar of the instrument needs no registers and no frame is illegal.
    [Source_intf] states that contract and this era does not move it.

    The shape, in five layers. L0 is the primitives — [Divider], [Isqrt] and [Exp2] from
    era four, [Prng.Rtl] in the core, and [Sigmoid] and [Softplus] here. L1 is the
    datapath: the RAMs, the state RAM, the tap rings, the key and value rings, the banked
    weight ROM, and [Mac] behind the one multiplier. L2 is the schedule, a value: the
    forward pass and the chain of the four seats are lists of operations. L3 compiles
    those lists into cases of a program counter. L4 is a small outer machine that holds
    the drawn frame and answers the socket. source.ml holds the design and its reasons,
    and [docs/mamba_rtl.md] holds the design of the whole.

    THE PLAN OF THE MODEL DECIDES THE PROGRAM. A layer is a Mamba block, the Zamba
    attention head or the feed-forward, and the schedule reads the kind of each one out of
    the checkpoint. The elected model is six blocks, then the head, then the feed-forward.

    What a caller must know:

    - The weights arrive at elaboration — the bitstream carries them — thus [create] takes
      the whole model. The seed arrives live from the SEED cell, and a [rewind] captures
      it.
    - **The source answers [step] from a frame it has already drawn.** It states the frame
      at once, runs the recurrence of that frame, and then draws the frame of the next
      step, thus the tempo waits for the wire and never for the network. The first bar is
      the lead-in of the model: every frame of it is silence, nothing is drawn, and the
      generator does not move.
    - Cycle counts are not a contract. The socket is latency-insensitive, and the PRNG
      steps only on command, thus a draw cannot move when the timing of the circuit
      changes. The cost model beside the schedule states the cost of a step exactly, and a
      bench pins the model to the circuit. The trunk's cost is a constant of the shape — a
      recurrence holds a state of one size for ever — and the head's grows with the fill
      of its ring until that ring is full, thus a step is constant after the first
      [Model.ring] of them.
    - There is one reset, and it is [rewind]: it loads the PRNG and returns the walk to
      its origin. It runs no pass, thus [idle] never falls for it. No memory is cleared
      and none needs to be: at step 0 the state reads as zero through one mux, a tap reads
      zero by its age rule, and the key and value rings are masked by the fill count. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [create ~model ~seed i] is the block. [model] is the model of the CONTRACT FILE: the
    ROM image, the tensor bases, every exponent, every per-head constant and every shape
    number come from it, and they decide the width of every counter and address. The
    elaboration checks the shape rules of the packing loudly. [seed] is the 32-bit seed of
    the walk, read at [rewind], thus one seed names one sequence in the simulation and on
    the board. *)
val create : model:Model.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t

module For_test : sig
  (** The walk, driven. [bin/gate_mamba.exe] runs it and prints what the circuit did, and
      [jax/tests/test_rtl_mamba.py] states what it must have done against the twin. The
      two sides cannot agree with each other by accident: the driver knows nothing of the
      twin, and the twin runs no circuit. *)
  module Bench : sig
    type t =
      { rewind : unit -> unit (** load the generator and return the walk to its origin *)
      ; play : unit -> int
      (** strobe one step and give the frame it answers. It raises [Failure] when the step
          is not answered, or when the walk runs past the budget the cost model states — a
          machine that stalls must fail a gate and not hang it. *)
      ; streams : unit -> int array list
      (** the stream snapshots of the LAST step: one for each time the circuit wrote the
          whole stream — the embed, then the join of each layer — in the order it wrote
          them. It probes the write port of the h RAM, thus it reads what the machine did
          and not what its output says. *)
      }

    (** [harness ~model ~seed ()] builds the simulation and its probes *)
    val harness : model:Model.t -> seed:int -> unit -> t
  end
end
