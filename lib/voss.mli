(** The Voss-McCartney core as a circuit: the rows, the walk and the voices.

    The block is the note source of this era, on the [Source_intf] interface. It holds the
    row bytes and the step count, and it owns the PRNG: the draw stream has exactly one
    consumer, thus no other block can move the sequence. The reference is [Pink] with the
    same model: [rewind] starts the model at the seed, and each [step] gives the notes
    that speak at that step, one at a time and from the lowest voice upward.

    The voices share one row set, thus the decomposition does not change the walk. The
    voices take the rows in order: the head of the voice list takes the first rows, thus
    it re-rolls at every step and it is the fastest voice. Each voice sums its own rows
    and maps the sum with its own constants. A voice speaks when the walk re-rolls a row
    of its group, and either its policy re-strikes a held pitch or the pitch moved. Every
    voice speaks at the first step of a run.

    At the end of the walk the block latches the note of each voice and a mask of the
    voices that speak, and then it reports them against the [ready] of the sequencer. A
    voice that does not speak takes no cycle on the socket.

    The walk is sequential, with one draw in two cycles. A walk takes at most
    [2 * rows + 2] cycles, and the step budget of the sequencer is at least one
    millisecond.

    The parameters come from [Pink]. The elaboration requires: at least one voice and not
    more than [Source_intf.voices], at least 2 rows in total, at least 2 degrees in each
    voice, each note inside 0 to 127, and a stretch window [rows * 256 / stretch] that is
    a power of two in each voice. With these, the mapping is shifts, adds and one constant
    multiply — no divider. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [seed] is the live view of the SEED cell; [rewind] captures its value at the run
    start. *)
val create : model:Pink.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
