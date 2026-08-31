(** The column engine: one forward pass over the sheet as it stands.

    It owns the two activation stores, the three bands that cache them — the column
    window, the residual columns and the output columns — the weight and norm ROMs, the
    counters and the layer turn, and one instance each of [Column_array] and [Epilogue].
    IT TAKES THE ELABORATION ITSELF as its functor argument, thus no width is stated
    twice.

    The design is [docs/diffusion_rtl.md], "The dwell", "The drain" and "The memories and
    their ports". The loop order is the column, then the output group, then the input
    channel, then the tap:

    {v
      for t   in 0 .. T - 1
        for g   in 0 .. groups - 1
          for cin in 0 .. Cin - 1
            for tap in 0 .. 8                  -- nine cycles of work
    v}

    **THE WINDOW IS AN IN-PLACE ROTATION OF THREE REGISTERS, AND THE TAP ORDER PAYS FOR
    IT.** The array takes the taps in any order; here they run dy-major, thus tap [k]
    takes the time slot [k / 3] and the pitch shift [k mod 3]. The slots hold
    [(t - 1, c)], [(t, c)] and [(t + 1, c)]; their last reads stand at the block cycles 2,
    5 and 8, and the next block's fetches go out at 0, 3 and 6 and land exactly there
    through the two-cycle read. Every turn therefore hides its fetches under the running
    dwell.

    **THE ZERO COLUMN AND THE STEM ENTER AT THE SLOT LOAD.** Beyond the ends of the roll a
    slot loads zero; on the stem it loads the sheet's plane column. One mux, one place,
    and the array never knows that the stem reads no store.

    **THE MEMORY ADDRESSES REGISTER BEFORE THE MEMORY AND THE DATA REGISTERS AFTER IT** —
    era four's trap, where a combinational address lets the tools retime the data register
    onto the address pins of every primitive and rebuild the address cone inside each one.
    Every read is therefore two cycles, and the schedule is written on that.

    **EVERY MEMORY STANDS IN THE BANKS ITS PLAN STATES**, because Vivado rounds an
    inferred memory's depth up to a power of two and warns of nothing. One address feeds
    every bank, each bank keeps its own data register — the block RAM's own latch — and
    the mux stands BEHIND those registers, selecting by the top address bits held twice. A
    mux before the data register evicts it into fabric and the address cone lands on the
    memory's pins instead: the first banked build lost setup by 0.354. Where the mux is a
    whole column wide the select rides one replica for each slice, as every array-scale
    take does.

    What a caller must know:

    - **[start] while [busy] is ignored, structurally.** The machine reads it in its idle
      state alone.
    - **The sheet answers [plane_step] and [plane] combinationally**, and this unit drives
      them so that the answer stands in the cycle the window slot loads it. This unit
      never names a step outside the roll — a tap beyond an end takes the zero column.
    - **A DWELL MUST COVER ITS OWN DRAIN AND THE BAND LOADS BEHIND IT:
      [9 * Cin >= rows + lanes + 2], WHICH [Elaboration.create] REFUSES.** A group's
      residual columns and norm words are fetched the moment the drain before it has read
      its last residual row, thus one buffer serves every group. The array's own rule —
      the [rows]-stage chain — is weaker by [lanes + 2], and the gap is one lane wide and
      real: H 6 at G 5 dwells 54 against 55, and G 5 is the fused rung's geometry. A shape
      inside the gap would read a half-loaded band and nothing below would say so.
    - **The pass includes the head's waits**, thus [busy] is the whole of it.
    - **A pass leaves the stores as it found them, except X.** Every column a layer reads
      was written by the layer before it, in the same pass, thus the walk needs no clear.
    - **THE STORE WRITE SIGNALS CARRY PINNED NAMES** — [x_write], [y_write], [x_data],
      [y_data] and the one [flush_address] both destinations share, beside [state] —
      because the stream gate probes them. The address is ONE name because one flush nest
      states both destinations: Y's port carries a ring address below it, but the COLUMN a
      write means is the nest's own.
    - There is no clear on the datapath or the memories: what is real is what the strobes
      mark. The counters, the state and [busy] clear. *)

open Hardcaml

(** The seam to the draw, stated once for both sides.

    [step_ready] is a LEVEL and not a strobe: "the logit file stands whole" is a state of
    this unit's registers. A strobe would export the event and ask every consumer to
    rebuild that state with a latch of its own.

    **THE STEPS ARE OFFERED IN ORDER, 0 to T - 1, EACH ONE EXACTLY ONE TIME**, and the
    level falls before it rises again. Thus the walk's count of its own acknowledgements
    IS the step, and NO TAG CROSSES THIS SEAM — the seam is a full interlock and has no
    pipe depth to model. *)
module Make (M : sig
    val e : Elaboration.t
  end) : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a
      (** a strobe: run one forward pass over the sheet as it stands. Read in the idle
          state alone. *)
      ; plane_column : 'a
      (** the sheet's answer to [plane_step] and [plane], combinational from its
          registers: [rows] activations of 16 bits, row 0 in the low bits *)
      ; logit_seat : 'a
      (** which seat's column [logits] states: a mux over the standing file, thus the draw
          reads any seat at any cycle *)
      ; step_taken : 'a
      (** a strobe: the draw is done with the offered step. Read in the offer state alone,
          thus a stray strobe reaches nothing. *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { busy : 'a (** 1 from [start] until the last step's acknowledgement *)
      ; plane_step : 'a (** the sheet face, wired through: the step... *)
      ; plane : 'a (** ...and the plane the stem is loading *)
      ; step_ready : 'a
      (** a LEVEL: the logit file holds the whole of the offered step. It rises the cycle
          after the last group's band of that step lands and falls on the edge after
          [step_taken]. *)
      ; logits : 'a
      (** the column of [logit_seat], whole: [rows] int16, class 0 in the low bits. It
          stands still exactly while [step_ready] stands, which is [Draw]'s own
          precondition, satisfied by construction. *)
      }
    [@@deriving hardcaml]
  end

  (** [create i] is the block. The elaboration's ROM images initialize its memories, thus
      no weight crosses a port. *)
  val create : Signal.t I.t -> Signal.t O.t
end

(** The bench, narrowed to what the driver of the RTL gate reads.

    THE ORACLE IS THE JAX TWIN. [bin/gate_diffusion.ml] drives one pass through this bench
    and prints every column the stores took; [jax/tests/test_rtl_diffusion.py] states what
    they must have been. Nothing here states that. *)
module For_test : sig
  module Bench (M : sig
      val e : Elaboration.t
    end) : sig
    (** one column write of a store, as the probe sees it: [address] is the SEMANTIC
        column, the flush nest's own, and [to_y] alone says where it landed *)
    type write =
      { to_y : bool
      ; address : int
      ; column : int array (** [rows] signed activations, row 0 first *)
      }

    (** what one pass did *)
    type pass =
      { written : write list
      (** the columns the stores took, in the order they went out *)
      ; offered : int array array list
      (** the logit columns of every step the head offered: one array for each step, and
          one column inside it for each seat *)
      }

    (** [run ~planes ()] is one forward pass. [planes] is the sheet as the stem's fetch
        reads it: the [rows] activations of one step and one plane. *)
    val run : planes:(step:int -> plane:int -> int array) -> unit -> pass
  end
end
