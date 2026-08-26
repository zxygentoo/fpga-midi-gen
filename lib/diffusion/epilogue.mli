(** The epilogue: one drained row of accumulators into one row of activations.

    L3 of the diffusion source. The column array states one row of its sums each cycle;
    this unit turns each into the activations that the tail of the twin's [layer_forward]
    states, [lanes] at a time, and hands them back one row a cycle. It holds no memory and
    no walk: the caller states the rows and the layer they belong to, and the unit states
    the activations.

    One lane is the tail of the twin, operation for operation:

    {v
      value = ((sums * gain) asr shift) + bias
      value = max value 0                        when [relu]
      value = clamp16 value
      value = clamp16 (max 0 (residual + value))     when [join]
    v}

    **A LAYER THAT CLOSES A PAIR CLAMPS TWICE.** The twin writes the convolution through
    its counted clamp and then writes the sum through it again, thus the second clamp is
    not a tidy repetition of the first: a value that rides the first one and then meets
    the residual row gives a different answer under one clamp than under two. Gate B is
    bit for bit, thus this is arithmetic and not hygiene.

    **THE WIDE BUFFERS ARE NOT HERE.** The residual row arrives as a row and the
    activations leave as a row; the residual columns and the output columns stand with the
    store, at L4. A residual column is a read cache and an output column is a write
    buffer, thus both are the memory's business — the same cut that put the three-column
    window with the store rather than inside the array. What this unit holds is arithmetic
    and a pipeline.

    **THE INPUT CARRIES TWO TIMESCALES.** [norms], [relu] and [join] are the layer's facts
    — the layer table states them, and they stand still from the first row of a drain to
    the last. Everything beside them is the stream: one drained row each [drained] cycle,
    exactly as the array hands it over, with the residual row beside it.

    What a caller must know:

    - **The tag travels.** [row] enters beside its sums and leaves beside its activations
      [latency] cycles later, thus the caller keeps no count of the pipe's depth. This is
      era four's rule for its MAC and this unit does not move it.
    - **A ragged group drains zeros, as it does at the array.** The elaboration pads the
      norm ROM to whole groups with zero words, as it pads the weight image, thus NO FETCH
      WALKS PAST A LAYER'S RANGE, or off the ROM's end at the last layer — a head of four
      channels in a group of five really reaches that end. A padded lane then states zero,
      except under a [join], where it states the ReLU of whatever [residual] carries on
      it. The caller ignores the lanes past the channel count, as it does at the array.
    - **The multiply is LUTs and never a DSP.** The array owns the DSPs: G follows H, thus
      the climb to the second rung costs none, but the fused rung at G 5 is 48 by 5, which
      is the device's whole 240. An epilogue that took DSPs would have to move at the
      moment the design is tightest.
    - **No clamp counts.** The twin counts every clamp because a format election stands on
      the number; the board has no status cell and no counter cell by standing rule, thus
      the circuit clamps and says nothing. The drift report is where that number lives.
    - There is no clear on the datapath: what is real is what [drained] marks. The tag
      registers clear. *)

open Hardcaml

(** The shape one instantiation is built for: the geometry of the column array beside it.
    [lanes] sizes every datapath port; [rows] sizes the tag alone, and the unit never
    reads it. Every width of the interface follows from the two. *)
module type Shape = sig
  val rows : int
  val lanes : int
end

(** Cycles from a row at [drained] to that row at [O.valid]. The multiply, the shift and
    the two clamps do not stand in one cycle at 100 MHz, thus the unit is a pipeline and
    the caller reads its depth here — or ignores it and follows the tag. *)
val latency : int

module Make (Shape : Shape) : sig
  (** [Shape.lanes], re-exported: a caller slices the ports on it *)
  val lanes : int

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
      (** the residual row — the pair input the join adds, under the name the twin gives
          it. [lanes] activations, lane 0 in the low bits; only a [join] reads it. *)
      ; norms : 'a
      (** the folded norm of each lane's output channel: [lanes] words, lane 0 in the low
          bits, each one exactly as [Elaboration.norm_rom] holds it — the bias in the low
          [Elaboration.bias_bits], the shift above it, then the gain. The unit slices them
          with the elaboration's own widths, thus the format has one home. *)
      ; relu : 'a
      (** the convolution takes its ReLU — the twin's own flag, [layer_forward ~relu]: the
          stem, and the opening layer of a pair. The gate is one wire; the fact is the
          layer's, thus it arrives at run time — one instantiation serves every layer. *)
      ; join : 'a
      (** the layer closes a pair, thus [residual] adds and the sum takes its own ReLU and
          its own clamp. Never beside [relu]: the pair-closing convolution runs with no
          ReLU of its own — the join carries the ReLU of the sum — and the head takes
          neither. *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { valid : 'a (** a strobe: this cycle carries one row of activations *)
      ; activation_row : 'a
      (** the tag that entered with them. It is [row] under another name only because a
          circuit's ports share one namespace and [row] stands on the input side, where it
          matches the array's own field and the seam wires across straight. *)
      ; activations : 'a
      (** the row's activations in the Q of the twin: [lanes] int16, lane 0 in the low
          bits *)
      }
    [@@deriving hardcaml]
  end

  (** [create i] is the block: [lanes] identical lanes and nothing between them. One
      instantiation serves every layer — the norms and the two flags carry everything a
      layer states — thus the layer table decides only what the caller feeds it. *)
  val create : Signal.t I.t -> Signal.t O.t
end
