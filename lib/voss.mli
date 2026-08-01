(** The Voss-McCartney core as a circuit: the rows, the walk and the note mapping.

    The block is the note source of this era, on the [Source_intf] interface. It holds the
    row bytes and the step count, and it owns the PRNG: the draw stream has exactly one
    consumer, thus no other block can move the sequence. The reference is [Pink]: [rewind]
    is [Pink.create], and each [step] is [Pink.next_note]. The stream-comparison test
    drives the two side by side.

    The walk is sequential, with one draw in two cycles. A walk takes at most
    [2 * rows + 2] cycles, and the step budget of the sequencer is at least one
    millisecond.

    The parameters come from [Pink.Params]. The elaboration requires: each note inside 0
    to 127, [rows] at least 2, and the stretch window [rows * 256 / stretch] a power of
    two. With these, the mapping is shifts, adds and one constant multiply — no divider. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [seed] is the live view of the SEED cell; [rewind] captures its value at the run
    start. *)
val create : params:Pink.Params.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
