(** The epilogue: one drained row of accumulators into one row of activations.

    The column array states one row of its sums each cycle; this unit turns each into the
    activations the tail of the twin's [layer_forward] states, [lanes] at a time, and
    hands them back one row a cycle. It holds no memory and no walk. The design is
    [docs/diffusion_rtl.md].

    One lane is the tail of the twin, operation for operation:

    {v
      value = ((sums * gain) asr shift) + bias
      value = max value 0                        when [relu]
      value = clamp16 value
      value = clamp16 (max 0 (residual + value))     when [join]
    v}

    **A LAYER THAT CLOSES A PAIR CLAMPS TWICE**, and the second clamp is arithmetic and
    not hygiene: a value that rides the first one and then meets the residual row gives a
    different answer under one clamp than under two. Gate B is bit for bit.

    What a caller must know:

    - **The tag travels.** [row] enters beside its sums and leaves beside its activations
      [latency] cycles later, thus the caller keeps no count of the pipe's depth.
    - **A ragged group drains zeros, as it does at the array.** The elaboration pads the
      norm ROM to whole groups with zero words, thus no fetch walks past a layer's range
      or off the ROM's end. A padded lane states zero, except under a [join], where it
      states the ReLU of whatever [residual] carries. The caller ignores the lanes past
      the channel count.
    - **The multiply is LUTs and never a DSP.** The array owns all 240 of them at the
      fused rung; an epilogue that took DSPs would have to move where the design is
      tightest.
    - **No clamp counts.** The twin counts every clamp because a format election stands on
      the number; the board has no counter cell by standing rule. The drift report is
      where that number lives.
    - There is no clear on the datapath: what is real is what [drained] marks. The tag
      registers clear. *)

open Hardcaml

(** The geometry of the column array beside it. [lanes] sizes every datapath port; [rows]
    sizes the tag alone, and the unit never reads it. *)
module type Shape = sig
  val rows : int
  val lanes : int
end

(** cycles from a row at [drained] to that row at [O.valid]: the multiply, the shift and
    the two clamps do not stand in one cycle at 100 MHz *)
val latency : int

module Make (Shape : Shape) : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; drained : 'a
      (** a strobe: this cycle carries one drained row. It is the array's [valid], one
          wire across the seam. *)
      ; row : 'a (** the tag of that row; it travels and the unit does not read it *)
      ; sums : 'a
      (** the row's accumulators, straight from the array: [lanes] int32 sums, lane 0 in
          the low bits *)
      ; residual : 'a
      (** the pair input the join adds: [lanes] activations, lane 0 in the low bits. Only
          a [join] reads it. *)
      ; norms : 'a
      (** the folded norm of each lane's output channel: [lanes] words, lane 0 in the low
          bits, exactly as [Elaboration.norm_rom] holds them. The unit slices them with
          the elaboration's own widths, thus the format has one home. *)
      ; relu : 'a
      (** the convolution takes its ReLU: the stem, and the opening layer of a pair. It
          arrives at run time, thus one instantiation serves every layer. *)
      ; join : 'a
      (** the layer closes a pair, thus [residual] adds and the sum takes its own ReLU and
          its own clamp. Never beside [relu]: the join carries the ReLU of the sum. *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { valid : 'a (** a strobe: this cycle carries one row of activations *)
      ; activation_row : 'a
      (** the tag that entered with them. It is [row] under a second name because a
          circuit's ports share one namespace. *)
      ; activations : 'a
      (** the row's activations in the Q of the twin: [lanes] int16, lane 0 in the low
          bits *)
      }
    [@@deriving hardcaml]
  end

  (** [create i] is the block: [lanes] identical lanes and nothing between them. The norms
      and the two flags carry everything a layer states, thus one instantiation serves
      every layer. *)
  val create : Signal.t I.t -> Signal.t O.t
end
