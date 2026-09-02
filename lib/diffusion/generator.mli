(** The generator: the era's draw machine — one sheet of music for each [start].

    It owns the outer FSM, the PRNG and its uniform shift register, the opening multiply,
    the alpha ROM, the draw service and the transfer face; it instantiates [Sheet],
    [Forward], [Draw] and [Prng.Rtl] and wires the plane face straight across.

    Phase I's whole [Source], narrowed: the walk, the engine, the sheet and the draw stand
    as they were, and what left is the socket identity. The era's face is [Source] still —
    the [Scheduler] over this unit at the elected gap — and this unit answers to the
    scheduler alone.

    The design is [docs/diffusion_rtl.md], "The walk" and "Phase II: the locked design".
    The shape of the machine:

    {v
      Idle --start--> OPEN --> [ MASK --> SERVE ] N times --> Idle
    v}

    and Idle is also where the sheet stands: the transfer face answers [step], and the
    next [start] draws the next sheet over it.

    **THE CONSUMPTION ORDER IS THE CONTRACT.** One PRNG serves the opening, every mask and
    every draw of a walk, in the order [Model] states: one uniform for each cell of the
    opening, then for each pass one for each cell (the masks) and one for each HIDDEN cell
    (the redraws), over [Model.cell_order] everywhere. A walk that draws for a standing
    cell, or assembles a uniform low byte first, states a different piece and NO GATE
    BELOW THIS ONE SAYS SO — thus the sheet agreement here compares PER PHASE and not only
    the finished sheet.

    What a caller must know:

    - **[start] is read at rest alone, and the PRNG's [load] rides that condition.** A
      start inside a walk would leave the PRNG where it stood and the seed silently
      unread. The [Scheduler] strobes it only under [idle].
    - **[step] is answered one cycle behind the strobe** with the frame of the sheet step
      the face has reached, and the counter then advances. It too is read at rest alone.
    - **THE TRANSFER FACE IS CYCLIC**: the counter wraps at [T], thus [T] strobes read the
      sheet whole and leave the face at frame 0. There is no silence rule here — silence
      is the scheduler's gap.
    - **A pass is not a cycle count.** The socket above is latency-insensitive and the
      PRNG steps only on command, thus the piece a seed names cannot move when the timing
      of the circuit changes. [Elaboration.pass_cycles] states what a pass costs less the
      draw.
    - **THE CELL-PORT DRIVES CARRY PINNED NAMES** — [cell_step], [cell_seat],
      [write_class], [cell_class], [write_mask], [cell_hidden], beside [walk_state] —
      because the per-phase gate probes them, and a rename would silently blind it.
    - There is no clear on the sheet and none on the datapath: the opening writes every
      class and a mask writes every bit before anything reads either. The counters, the
      state and [idle] clear. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a (** a strobe: draw one sheet; read at rest alone *)
    ; step : 'a (** a strobe: state one frame of the standing sheet *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { frame : 'a
    (** the voice codes of the transfer step; seat 0 is the low byte, as [Frame] states *)
    ; valid : 'a (** answers [step] one cycle behind the strobe *)
    ; idle : 'a (** 1 at rest: the sheet stands and [start] is read *)
    }
  [@@deriving hardcaml]
end

(** [create ~e ~seed i] is the block. Every width, table and ROM image of the machine
    comes out of [e], thus the caller names this and nothing narrower. [seed] is read at
    [start] — the SEED cell of the run, stepped by the scheduler for each sheet — thus one
    seed names one piece in the simulation and on the hardware.

    It raises [Invalid_argument] when the sheet cannot hold what the opening draws: a
    probe geometry that narrows P below the top class of a seat's register states a class
    no cell can hold. *)
val create : e:Elaboration.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t

(** The bench, narrowed to what the driver of the RTL gate reads.

    THE ORACLE IS THE JAX TWIN. [bin/gate_diffusion.ml] drives the circuit through this
    harness and prints what it did; [jax/tests/test_rtl_diffusion.py] states what it must
    have done. Nothing here states that, thus the driver cannot pass a walk by agreeing
    with itself. *)
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
      { start : unit -> unit
      (** run one whole walk, from the rest to the rest, and clear the write log behind
          it. It raises [Failure] when the walk does not finish inside the budget the cost
          model states — a machine that stalls must fail a gate and not hang it. *)
      ; play : unit -> int
      (** strobe one transfer step and give the frame it answers. It raises [Failure] when
          the step is not answered. *)
      ; writes : unit -> write list
      (** every write of the last walk, in the order the walk made them *)
      }

    (** [harness ~e ~seed ()] builds the simulation and its probes *)
    val harness : e:Elaboration.t -> seed:int -> unit -> t
  end
end
