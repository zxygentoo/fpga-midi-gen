(** The note-source socket: the contract between a model core and the [Sequencer].

    A source states one step of music as one frame — the four voice codes of [Frame] in
    one word, seat 0 the low byte. A frame says which voice holds which pitch and nothing
    else, thus the socket carries no note, no seat and no release: the sequencer holds the
    set of pitches that sound and [Frame.events_of_frames] states what a new frame does to
    it.

    Every step gives a frame, and a step in which nothing sounds gives four zero bytes and
    not silence on the socket.

    The handshake:

    - [idle] is 1 when the source is at rest. The sequencer strobes [rewind] or [step]
      only then.
    - [valid] answers [step] one time for each step, and [frame] holds the frame of that
      step while it is 1. There is no [ready]: the sequencer strobes and waits, thus it is
      always ready and no transfer rule is necessary.
    - [idle] rising ends the command.

    **A source answers [step] from a frame it has drawn already.** It draws the frame of
    the next step while the sequencer sends the messages of this one, thus [valid] comes
    at once and the wire is the only thing the tempo waits for. If a step ever arrives
    before the draw finishes, [valid] comes late and the sequencer waits: the socket is
    latency-insensitive, and the rule is a rule of correctness and not of timing.

    No configuration crosses the socket: a source takes what it needs by closure at
    elaboration. [docs/transformer_rtl.md] gives the design and its reasons.

    The module holds only the interface records, thus it has no [.mli]: the definitions
    are their own signature. *)

open Hardcaml

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; rewind : 'a (** a strobe: go to the origin of the sequence — the run start *)
    ; step : 'a (** a strobe: state one step of music *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { frame : 'a [@bits Frame.code_bits * Frame.voices]
    (** the voice codes of the step; seat 0 is the low byte *)
    ; valid : 'a
    (** a strobe: [frame] holds the frame of the step that [step] asked for *)
    ; idle : 'a (** 1 when the source is at rest and can take a command *)
    }
  [@@deriving hardcaml]
end
