(** The note-source socket: the interface between a model core and the [Sequencer].

    A source answers [step] with one note. [rewind] puts it at the origin of its sequence
    — the run start — thus the same run plays the same sequence. The handshake covers any
    walk length: [ready] is 0 while the source works, and [valid] strobes when [note]
    answers the last [step].

    The socket carries no configuration. A source takes what it needs by closure at
    elaboration: [Voss] takes the model parameters and the live view of the SEED cell, and
    a later core takes its weights port the same way. The top level seats a model core
    with one line: the [source] argument of [Model.create].

    The module holds only the interface records, thus it has no [.mli]: the definitions
    are their own signature. *)

open Hardcaml

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; rewind : 'a (** a strobe: go to the origin of the sequence — the run start *)
    ; step : 'a (** a strobe: give the note of one step *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { note : 'a [@bits 8] (** the MIDI note of the step *)
    ; valid : 'a (** a strobe: [note] answers the last [step] *)
    ; ready : 'a (** 1 when the source can take [rewind] or [step] *)
    }
  [@@deriving hardcaml]
end
