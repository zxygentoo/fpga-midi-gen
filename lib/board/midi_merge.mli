(** The merge of the message sources: which source gives the next message.

    The block has no state, therefore it has no clock and no clear. It is a priority mux,
    and the [ready] of [Midi_out] goes back to the source that has the grant.

    The doorbell has the priority. The host waits in a poll loop, and the model accepts a
    delay of one millisecond. A manual debug action is too rare to stop the model.

    [Midi_out] holds its [ready] at 0 for the full length of a send. Therefore no other
    source can put a message between the bytes of the message that goes out, and this
    block needs no state to make this true. *)

open Hardcaml

module I : sig
  type 'a t =
    { doorbell : 'a Midi.Rtl.Message.t (** the test-message source; it has the priority *)
    ; model : 'a Midi.Rtl.Message.t (** the model source *)
    ; out_ready : 'a (** from [Midi_out]: 1 when it can take a message *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { out : 'a Midi.Rtl.Message.t (** the message of the source that has the grant *)
    ; doorbell_ready : 'a (** 1 when [Midi_out] takes the doorbell message *)
    ; model_ready : 'a (** 1 when [Midi_out] takes the model message *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
