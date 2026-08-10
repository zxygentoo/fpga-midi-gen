(** The sounding state: which pitches ring, and what the sentence has done so far. The
    mask of the training loss and the guard of the sampler both derive from it, thus the
    model spends no mass on a code that the draw would refuse.

    [legal_mask] is the full grammar of the corpus encoding: [Start] never — it is input
    only; an [On] when its pitch does not sound, a seat of the four is open, and its pitch
    is below the last ON of the sentence; an [Off] when its pitch sounds, the sentence
    holds no ON yet, and its pitch is above the last OFF; [End] always. The training loss
    carries this mask, thus the model needs it at the draw: its raw mass on the codes
    outside is untrained.

    The ONs fall and the OFFs climb, and each direction earns its place. The fall is the
    melody leading: the top voice is chosen before the voices under it and conditions on
    none of them, as the music is written. The climb then makes the two runs meet in the
    middle, so the release of the top moving voice sits beside its attack and one melodic
    step is two adjacent tokens.

    Both directions are rules of the instrument and not conventions of the tokenizer. A
    convention leaves the permutations of a chord inside the softmax, where the model must
    spend mass to learn an order the mask could refuse for nothing.

    Two of these rules protect the instrument itself: no [On] of a sounding pitch (the
    cross-kill of the S-1), and no fifth voice. The rest hold the sentence order.

    This state is the register set of the future circuit: the sounding vector, the last
    ON, the last OFF and the seat count. *)

type t

(** nothing sounds, and the sentence holds no ON and no OFF yet *)
val silence : t

(** [step t token] walks one token. It does not test legality. *)
val step : t -> Token.t -> t

(** [legal_mask t] is the grammar flag of every code, indexed by the code. *)
val legal_mask : t -> bool array
