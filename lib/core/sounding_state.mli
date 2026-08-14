(** The sounding state: which pitches ring, and what the sentence has done so far. The
    mask of the training loss and the guard of the sampler both derive from it, thus the
    model spends no mass on a code that the draw would refuse.

    [legal_mask] is the full grammar of the corpus encoding: [Start] never — it is input
    only; an [On] when its pitch is not 0, its pitch does not sound, a seat of the four is
    open, and its pitch is below the last ON of the sentence; an [Off] when its pitch
    sounds, the sentence holds no ON yet, and its pitch is above the last OFF; [End]
    always. The training loss carries this mask, thus the model needs it at the draw: its
    raw mass on the codes outside is untrained.

    The ONs fall and the OFFs climb, and each direction earns its place. The fall is the
    melody leading: the top voice is chosen before the voices under it and conditions on
    none of them, as the music is written. The climb then makes the two runs meet in the
    middle, so the release of the top moving voice sits beside its attack and one melodic
    step is two adjacent tokens.

    Both directions are rules of the instrument and not conventions of the tokenizer. A
    convention leaves the permutations of a chord inside the softmax, where the model must
    spend mass to learn an order the mask could refuse for nothing.

    Two of these rules protect the instrument itself: no [On] of a sounding pitch (the
    cross-kill of the S-1), and no fifth voice. One protects the encoding: [Off 0] has no
    code, because code 0 is End, thus a sounding pitch 0 could never be released and pitch
    0 never starts. The rest hold the sentence order.

    This state is the register set of the circuit below: the sounding vector, the last ON,
    the last OFF and the seat count. *)

open Hardcaml

type t

(** nothing sounds, and the sentence holds no ON and no OFF yet *)
val silence : t

(** [step t token] walks one token. It does not test legality. *)
val step : t -> Token.t -> t

(** [legal_mask t] is the grammar flag of every code, indexed by the code. *)
val legal_mask : t -> bool array

(** [sounding t] is the pitches that sound, ascending. A release walks them in this order,
    thus its OFFs climb as the grammar states, and the order needs no seat: the software
    model, the reference and the circuit can all state it the same way. *)
val sounding : t -> int list

(** The circuit: the same grammar, in registers. The state above is the reference, and the
    block test in this module drives the two side by side over drawn walks. Therefore the
    definition of the grammar has one home, and a change to the rules must move both.

    [land_] walks one token into the state, as [step] does; [query] asks the legality of
    one code, as [legal_mask] answers it. The two ports are independent: a source queries
    every code of the vocabulary while the state stands, and lands one token at the end of
    the step.

    Every width comes from the encoding of [Token]. A code is one byte, its top bit is the
    type and the rest is the pitch, thus [Token.vocab] and [Token.seats] shape every
    register and the reserved codes come from [Token.to_code].

    The contract:

    - [clear] and [boot] both put the state at silence. [clear] is the reset of the
      circuit, and [boot] is the rewind of a source. [boot] wins over [land_] in the same
      cycle.
    - [land_] is a strobe: it applies the token in [code] one time. It does not test
      legality, as [step] does not. The caller lands only a token the grammar allows: the
      seat count is a register and not a set, thus an illegal token moves it where [step]
      would hold it, and the two states part.
    - [query] is combinational into [legal]. It costs no cycle and it changes no state,
      thus the unit puts no wait and no handshake on its caller. *)
module Rtl : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; boot : 'a (** a strobe: the state takes silence *)
      ; land_ : 'a (** a strobe: apply the token in [code] *)
      ; code : 'a (** the token to apply, one byte *)
      ; query : 'a (** the code to test, one byte *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { legal : 'a (** the grammar allows [query] in this state *) }
    [@@deriving hardcaml]
  end

  val create : Signal.t I.t -> Signal.t O.t
end
