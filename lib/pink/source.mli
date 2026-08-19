(** The Voss-McCartney core as a circuit: the rows, the walk and the frame.

    The block is a note source on the [Source_intf] interface. It holds the row bytes and
    the step count, and it owns the PRNG: the draw stream has exactly one consumer, thus
    no other block can move the sequence. The reference is [Pink] with the same model:
    [rewind] starts the model at the seed, and each [step] gives the frame that
    [Pink.next_frame] states.

    The voices share one row set, thus the decomposition does not change the walk. The
    voices take the rows in order: the head of the voice list takes the first rows, thus
    it re-rolls at every step and it is the fastest voice. Each voice sums its own rows
    and maps the sum with its own constants. The row order puts the highest voice first
    and seat 0 of a frame is the lowest, thus the seats turn the list around and a model
    with fewer voices leaves the low seats silent.

    **A frame states which pitch a voice holds, and no voice of this model rests.**
    Therefore the block keeps no note register and no speak decision: the note of a voice
    is combinational from the rows, the pitch of the step before decides nothing, and a
    voice that re-rolls onto the pitch it already holds states the same code again. Era
    one re-struck such a pitch and the frame socket cannot; [Pink] holds what the
    smoothing costs.

    The walk is sequential, with one draw in two cycles. A walk takes at most
    [2 * rows + 3] cycles, and the step budget of the sequencer is at least one
    millisecond.

    The parameters come from [Pink]. The elaboration requires: at least one voice and not
    more than [Frame.voices], at least 2 rows in total, at least 2 degrees in each voice,
    each note inside 0 to 127, and a stretch window [rows * 256 / stretch] that is a power
    of two in each voice. With these, the mapping is shifts, adds and one constant
    multiply — no divider. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [seed] is the live view of the SEED cell; [rewind] captures its value at the run
    start. *)
val create : model:Pink.t -> seed:Signal.t -> Signal.t I.t -> Signal.t O.t
