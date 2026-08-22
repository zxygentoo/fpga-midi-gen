(** The state-space model as a circuit: the note source of era five.

    The block computes the integers of [Quantized], operation for operation. [Quantized]
    is the reference, and the frame comparison in this module proves the match: the frames
    of the circuit must equal the frames of [Quantized.Engine], step for step.

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
      [Mamba.elected_ring] of them.
    - There is one reset, and it is [rewind]: it loads the PRNG and returns the walk to
      its origin. It runs no pass, thus [idle] never falls for it. No memory is cleared
      and none needs to be: at step 0 the state reads as zero through one mux, a tap reads
      zero by its age rule, and the key and value rings are masked by the fill count. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [create ~model ~seed i] is the block. [model] is the quantized model of the
    elaboration: the ROM image, the tensor bases, every exponent and every per-head
    constant come from it, and the shape it carries decides the width of every counter and
    address. The elaboration checks the shape rules of the packing loudly. [seed] is the
    32-bit seed of the walk, read at [rewind], thus one seed names one sequence in the
    simulation and on the board. *)
val create : model:Quantized.Model.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
