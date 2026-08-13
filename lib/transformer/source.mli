(** The transformer as a circuit: the note source of era three.

    The block computes the integers of [Quantized], operation for operation. [Quantized]
    is the reference, and the stream comparison in this module proves the match: the
    events of the circuit must equal the events of [Quantized.Engine], event for event.

    The block sits behind [Source_intf] and states its own releases. An On takes the
    highest free seat, an Off names the seat that holds its pitch, and
    [Sounding_state.Rtl] guarantees both: no draw crosses the socket that the grammar of
    the instrument refuses.

    The shape, in five layers. L0 is the primitives — [Divider], [Isqrt], [Exp2] here, and
    [Sounding_state.Rtl] and [Prng.Rtl] in the core. L1 is the datapath: the RAMs, the KV
    rings, the banked weight ROM, and [Mac] behind the one multiplier. L2 is the schedule,
    a value: the forward pass and the sampler are lists of operations, and one operation
    holds the facts of one step. L3 compiles that list into cases of a program counter. L4
    is a small outer machine that keeps the token walk, the seats and the handshake.
    source.ml holds the design and its reasons.

    What a caller must know:

    - The weights arrive at elaboration — the bitstream carries them — thus [create] takes
      the whole model. The seed arrives live from the SEED cell, and a [rewind] captures
      it, as the pink era does.
    - Cycle counts are not a contract. The socket is latency-insensitive, and the PRNG
      steps only on command, thus a draw cannot move when the timing of the circuit
      changes. One step costs of the order of a million cycles at the shape of the era;
      the cost model beside the schedule states it exactly, and a bench pins the model to
      the circuit.
    - One step gives the events of one sentence, and a step where nothing speaks gives no
      note. The block returns to [idle] between commands, as the socket demands. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [create ~model ~seed i] is the block. [model] is the quantized model of the
    elaboration: the ROM image, the tensor bases and every exponent come from it, and the
    shape it carries decides the width of every counter and address. The elaboration
    checks the shape rules of the packing loudly. [seed] is the 32-bit seed of the walk,
    read at [rewind], thus one seed names one sequence in the simulation and on the board. *)
val create : model:Quantized.Model.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
