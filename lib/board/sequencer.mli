(** The sequencer: the run engine of the model.

    The block reads the parameter views, drives the note source, and gives the model
    messages on the [Midi.Rtl.Message] interface. The rules are the ones of
    [docs/pink_rtl.md]:

    - A prescaler divides the clock by [clocks_per_ms]. The run start resets it, thus a
      run is the same cycle for cycle in each simulation and on the board.
    - The source goes to its origin at the run start and at no other time.
    - At each step boundary the block samples RUN and STEP_MS: a change applies at the
      next step. A sampled STEP_MS of 0 counts as 1.
    - At each step the source gives the notes that speak, one at a time. For a note with
      [on] at 1 the block closes the note of that voice, if it holds one, and then opens
      the new note. Therefore the number of open notes is never more than the number of
      voices. For a note with [on] at 0 the block sends one Note Off from the stored pair
      of that voice and frees the seat — the path of a source that states its own
      releases.
    - The block takes a note with [source_ready] in the cycle that completes its Note On,
      thus the source holds a stable pitch for both messages.
    - Each voice has an open-note register. It keeps the note and the channel of the Note
      On, and the Note Off takes the stored pair. When the run stops, each open voice gets
      its Note Off, from the highest voice downward.
    - The block sends a Note Off only to keep its state true — the steal and the stop. It
      does not shape the music: a musical release is the source's, stated with [on] at 0.
    - The millisecond count does not pause while the merge stalls a message, thus the beat
      does not drift. One step sends at most two messages for each voice, which is about
      7.7 ms of line time for four voices. A step that is shorter than its messages
      stretches. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t (** the named views; each one is stable *)
    ; source : 'a Source_intf.O.t (** the outputs of the note source *)
    ; midi_ready : 'a (** from [Midi_merge]: 1 when the MIDI path takes the message *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { midi : 'a Midi.Rtl.Message.t (** the model source *)
    ; source_rewind : 'a (** a strobe: the source goes to its origin — the run start *)
    ; source_step : 'a (** a strobe: take one step and give the notes that speak *)
    ; source_ready : 'a (** 1 in the cycle that takes the note of the source *)
    }
  [@@deriving hardcaml]
end

val create : clocks_per_ms:int -> Signal.t I.t -> Signal.t O.t
