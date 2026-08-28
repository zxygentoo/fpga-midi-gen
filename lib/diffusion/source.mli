(** The walk: the machine around the column engine, and era six's face to the sequencer.

    L4, whole. This unit owns the outer FSM, the generator and the uniform shift register,
    the opening multiply, the alpha ROM, the draw service and the socket answers; it
    instantiates [Canvas], [Forward], [Draw] and [Prng.Rtl] and wires the plane face
    straight across. Everything that S2 and S3 did not already make a unit of is here,
    because nothing that is left is reusable: the walk is one machine.

    The design is [docs/diffusion_rtl.md], "The walk" and "The seam to the sequencer". The
    shape of the machine:

    {v
      Idle --rewind--> OPEN --> [ MASK --> SERVE ] N times --> Idle
    v}

    and Idle is also the PLAY phase: the canvas stands, the score face answers [step], and
    the next [rewind] opens the next piece.

    **THE CONSUMPTION ORDER IS THE CONTRACT.** One generator serves the opening, every
    mask and every draw of a walk, in the order [Model] states: one uniform for each cell
    of the opening, then for each pass one uniform for each cell (the masks) and one for
    each HIDDEN cell (the redraws), the cells in [Model.cell_order] everywhere. A walk
    that draws for a standing cell, or assembles a uniform low byte first, states a
    different piece and NO GATE BELOW THIS ONE SAYS SO. The canvas agreement in this
    module therefore compares PER PHASE and not only the finished canvas.

    What a caller must know:

    - **[rewind] is read at rest alone, and the generator's [load] rides that condition.**
      One seed names one canvas — in the simulation and on the board — thus a rewind that
      arrived inside a walk would leave the generator where it stood and the seed silently
      unread. The sequencer's contract already strobes only under [idle].
    - **[step] is answered one cycle behind the strobe** with the frame of the canvas step
      the walk has reached, and the counter then advances. Past step [T - 1] the frame is
      FOUR ZERO BYTES, for ever, until the next [rewind] puts the counter back at 0.
    - **A pass is not a cycle count.** The socket is latency-insensitive and the generator
      steps only on command, thus the piece a seed names cannot move when the timing of
      the circuit changes. [Elaboration.pass_cycles] states what a pass costs less the
      draw, and the walk bench in this module measures the rest.
    - **THE CELL-PORT DRIVES CARRY PINNED NAMES**: [cell_step], [cell_seat],
      [write_class], [cell_class], [write_mask] and [cell_hidden], beside [state]. The
      per-phase gate probes the one port that all three phases share, thus a rename cannot
      silently blind it.
    - There is no clear on the canvas and none on the datapath: the opening writes every
      class and a mask writes every bit before anything reads either. The counters, the
      state and [idle] clear. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [create ~e ~seed i] is the block. [e] is the elaboration of the elected checkpoint:
    the layer table, the ROM images, the anneal table, the register of each seat and every
    width of the machine come out of it, thus the top level names this and nothing
    narrower. [seed] is the 32-bit seed of the walk, read at [rewind] — the SEED cell on
    the board — thus one seed names one piece in the simulation and on the hardware.

    It raises [Invalid_argument] when the canvas cannot hold what the opening draws: the
    registers of the seats are the corpus's, thus a probe geometry that narrows P below
    the top class of a register states a class no cell can hold. *)
val create : e:Elaboration.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t

(** The bench, narrowed to what the driver of the RTL gate reads.

    The walk gate of this era runs in [jax/tests/test_rtl.py]: THE ORACLE IS THE JAX TWIN,
    [bin/gate_diffusion.ml] drives the circuit through this harness and prints what the
    circuit did, and Python states what it must have done. Nothing here states that, thus
    the driver cannot pass a walk by agreeing with itself. *)
module For_test : sig
  module Bench : sig
    (** one write of the cell port, as the probe sees it *)
    type write =
      { mask : bool (** the mask face, and not the class face *)
      ; step : int
      ; seat : int
      ; value : int (** the class a face wrote, or the mask bit *)
      }

    (** the walk, driven *)
    type t =
      { rewind : unit -> unit
      (** run one whole walk, from the rest to the rest, and clear the write log behind
          it. It raises [Failure] when the walk does not finish inside the budget the cost
          model states — a machine that stalls must fail a gate and not hang it. *)
      ; play : unit -> int
      (** strobe one step and give the frame it answers. It raises [Failure] when the step
          is not answered. *)
      ; writes : unit -> write list
      (** every write of the last walk, in the order the walk made them *)
      }

    (** [harness ~e ~seed ()] builds the simulation and its probes *)
    val harness : e:Elaboration.t -> seed:int -> unit -> t
  end
end
