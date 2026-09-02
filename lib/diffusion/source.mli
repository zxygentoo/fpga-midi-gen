(** Era six's note source: the face the board seats, and the one line that says what the
    era is made of.

    EVERY ERA HAS A [Source] AND THE BOARD KNOWS NO OTHER NAME. [bin/gen_verilog] hands
    [Source.create] to [Top] for all four of them, thus the seat reads the same whatever
    is under it, and what parts one era from the next stays inside its own library.

    Under this one stand two units and one elected number:

    - **[Generator]** draws one sheet of music for each [start]. It is phase I's whole
      machine — the walk, the engine, the sheet, the draw — and it speaks no socket.
    - **[Scheduler]** drives the generator and answers the socket: the playing copy, the
      succession of seeds, the gap and the restart. It is the behaviour of this source,
      and it takes the gap as a parameter because a unit test states its own.
    - **[gap]**, below, is the number the ear elected for the board.

    The design is [docs/diffusion_rtl.md], "Phase II: the locked design — the generator
    and the scheduler". Nothing else in the era names the socket. *)

open Hardcaml

(** The silent steps between two sheets: 32, two bars at STEP_MS 200, and the software
    default of [jax/diffusion/infer.py]'s [--gap]. The FIRST of them is the drain, which
    is why [Scheduler] refuses a gap of 0; the election of 32 is this module's, because it
    is a fact of the performance and not of the machine.

    It is exported for the succession gate, which must know how many silent steps its run
    may overshoot into before it reaches a third sheet. *)
val gap : int

(** [create ~e ~seed i] is the era's source. Every width, table and ROM image comes out of
    [e]; [seed] is the SEED view, latched at [rewind], and sheet k of the run plays
    [(seed + k) mod 2^32].

    It raises [Invalid_argument] when the sheet cannot hold what the opening draws — the
    generator's own refusal, at a probe geometry that narrows P below the top class of a
    seat's register. *)
val create
  :  e:Elaboration.t
  -> seed:Signal.t
  -> Signal.t Source_intf.I.t
  -> Signal.t Source_intf.O.t
