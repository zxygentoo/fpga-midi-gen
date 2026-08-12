(** The transformer prototype as a circuit: the note source of era three.

    The block computes the integers of [Fixed], operation for operation — the twin is the
    reference, and the stream comparison proves the match. One multiplier, one divider,
    one isqrt and one big walk; the budget of the step makes speed worthless, thus every
    loop runs at three cycles for each term. [docs/transformer_rtl_proto.md] holds the
    design: the memories, the operations and the walk of the source FSM.

    The block sits behind [Source_intf] with the type bit: it states its own releases. An
    On takes the highest free seat, an Off names the seat that holds its pitch, and the
    legality mask guarantees both. The sequencer of this era elaborates [~gated:false].

    The weights arrive at elaboration — the bitstream carries them — and the SEED cell
    arrives as the live view; a rewind captures it, as the pink era does. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [model] is the quantized model of the elaboration: the ROM and its addresses come from
    it. The prototype accepts one shape — its address packing — and the elaboration checks
    it loudly. *)
val create : model:Fixed.Model.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
