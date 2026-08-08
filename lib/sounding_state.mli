(** The sounding state: which pitches ring, and what the sentence has done so far. The
    mask of the training loss and the guard of the sampler both derive from it, thus the
    model spends no mass on a code that the draw would refuse.

    [legal_mask] is the full grammar of the corpus encoding: [Start] never — it is input
    only; an [On] when its pitch does not sound, a seat of the four is open, and its pitch
    is above the last ON of the sentence; an [Off] when its pitch sounds and the sentence
    holds no ON yet; [End] always. The training loss carries this mask, thus the model
    needs it at the draw: its raw mass on the codes outside is untrained.

    Two of these rules protect the instrument: no [On] of a sounding pitch (the cross-kill
    of the S-1), and no fifth voice. The rest hold the sentence order of the encoding.

    This state is the register set of the future circuit: the sounding vector, the last ON
    and the seat count. *)

type t

(** nothing sounds, and the sentence holds no ON yet *)
val silence : t

(** [step t token] walks one token. It does not test legality. *)
val step : t -> Token.t -> t

(** [legal_mask t] is the grammar flag of every code, indexed by the code. *)
val legal_mask : t -> bool array
