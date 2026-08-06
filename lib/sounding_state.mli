(** The sounding state: which pitches ring, and what the sentence has done so far. The
    guards of the sampler derive from it; the training loss does not see it — the model
    learns the instrument from the data, and the sampler holds the line at the draw.

    [legal_mask] is the full grammar of the corpus encoding: an [Off] when its pitch
    sounds and the sentence holds no ON yet; an [On] when its pitch does not sound, a seat
    of the four is open, and its pitch is above the last ON of the sentence; [End] always.
    A model whose training loss carried this mask needs it at the draw: its raw mass on
    the codes outside is untrained.

    [safe_mask] is the safety floor alone — the two rules whose violation damages the
    instrument state: no [On] of a sounding pitch (the cross-kill of the S-1), and no
    fifth voice. An [Off] of a silent pitch passes: the synthesizer ignores it. A model
    that learned the grammar from data needs only this.

    This state is the register set of the future circuit: the sounding vector, the last ON
    and the seat count. *)

type t

(** nothing sounds, and the sentence holds no ON yet *)
val silence : t

(** [step t token] walks one token. It does not test legality. *)
val step : t -> Token.t -> t

(** [legal_mask t] is the grammar flag of each byte code, indexed by the code. *)
val legal_mask : t -> bool array

(** [safe_mask t] is the safety flag of each byte code, indexed by the code. *)
val safe_mask : t -> bool array
