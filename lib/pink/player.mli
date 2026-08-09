(** The player: the rule that makes note events from the steps of a model.

    The rule has one definition, thus the audition tool and the circuit play the same
    piece. [bin/play_pink.ml] sends the events to the synthesizer, and the RTL tests
    compare the message stream of the circuit against them. The block that the circuit
    compares against is [Sequencer]: this module is its reference, as [Pink] is the
    reference of [Voss].

    A voice speaks when it is due and it holds no note, or its pitch moved, or its policy
    re-strikes a held pitch. A voice that speaks closes its note before it opens the new
    one, thus the four voices of the synthesizer are never exceeded.

    The player holds no time and no MIDI: the caller decides when a step and a gate come,
    and it gives each event a channel and a velocity. *)

(** A note event of the performance. *)
module Event : sig
  type t =
    | On of int (** the MIDI note *)
    | Off of int (** the MIDI note *)
  [@@deriving sexp_of]
end

type t

(** [create ~model ~seed] is the player at the origin, with no note open. The rules of
    [Pink.create] apply to the model and the seed. *)
val create : model:Pink.t -> seed:int -> t

(** [step t] takes one step of the model and gives the events of the voices that speak,
    from the lowest voice upward — the order of the wire. *)
val step : t -> t * Event.t list

(** [gate t] closes the note of the highest voice. The lower voices sustain to their next
    articulation, thus the gate does not touch them. The caller decides when the gate
    comes: it is the gate time, and only when that time is less than the step. *)
val gate : t -> t * Event.t list

(** [stop t] closes the note of each voice, from the lowest upward. *)
val stop : t -> t * Event.t list
