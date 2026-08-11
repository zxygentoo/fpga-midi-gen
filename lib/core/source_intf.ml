(** The note-source socket: the interface between a model core and the [Sequencer].

    A source answers [step] with the notes that speak at that step, one at a time and from
    the highest voice downward: the melody leads, and a model that chooses the top voice
    first reports in the order of its choices. A note that does not speak never crosses,
    thus the sequencer makes no selection and plays what it receives. [rewind] puts the
    source at the origin of its sequence — the run start — thus the same run plays the
    same sequence.

    The synthesizer has four voices, thus [voices] is 4: a fact of the hardware and not a
    parameter of the model. Voice 0 is the lowest, and a source with fewer voices takes
    the high numbers, because the sequencer gates the highest voice only. Each note
    carries its voice, because the sequencer keeps one open note for each of them and the
    voice number is the key of that state.

    The handshake:

    - [idle] is 1 when the source is at rest. The sequencer strobes [rewind] or [step]
      only then.
    - The source holds a note while [valid] is 1, and the transfer is the one cycle in
      which [valid] and [ready] are both 1 — the rule of [Midi.Rtl.Message]. After it the
      source gives the next note or goes back to rest.
    - [idle] rising ends the command: after [step] the step has no more notes, and after
      [rewind] the source is at the origin. A step where nothing speaks gives no [valid].

    No configuration crosses the socket: a source takes what it needs by closure at
    elaboration. [docs/pink_rtl.md] gives the design and its reasons.

    The module holds only the interface records, thus it has no [.mli]: the definitions
    are their own signature. *)

open Hardcaml

(** the number of voices of the synthesizer *)
let voices = 4

module Note = struct
  type 'a t =
    { voice : 'a [@bits Signal.address_bits_for voices]
    (** the voice that sounds it; 0 is the lowest *)
    ; pitch : 'a [@bits 8] (** the MIDI note number *)
    }
  [@@deriving hardcaml]
end

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; rewind : 'a (** a strobe: go to the origin of the sequence — the run start *)
    ; step : 'a (** a strobe: take one step and give the notes that speak *)
    ; ready : 'a (** 1 when the sequencer can take the note *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { note : 'a Note.t (** the note that speaks; it holds while [valid] is 1 *)
    ; valid : 'a (** 1 while [note] holds a note that the sequencer must play *)
    ; idle : 'a (** 1 when the source is at rest and can take a command *)
    }
  [@@deriving hardcaml]
end
