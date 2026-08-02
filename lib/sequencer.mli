(** The sequencer: the run engine of the model.

    The block reads the parameter views, drives the note source, and gives the model
    messages on the [Midi.Rtl.Message] interface. The rules are the ones of
    [docs/pink_rtl.md]:

    - A prescaler divides the clock by [clocks_per_ms]. The run start resets it, thus a
      run is the same cycle for cycle in each simulation and on the board.
    - The source goes to its origin at the run start and at no other time.
    - At each step boundary the block samples RUN, STEP_MS and GATE_MS: a change applies
      at the next step. A sampled STEP_MS of 0 counts as 1.
    - At each step the source gives the notes that speak, one at a time. For each note the
      block closes the note of that voice, if it holds one, and then opens the new note.
      Therefore the number of open notes is never more than the number of voices.
    - The block takes a note with [source_ready] in the cycle that completes its Note On,
      thus the source holds a stable pitch for both messages.
    - Each voice has an open-note register. It keeps the note and the channel of the Note
      On, and the Note Off takes the stored pair. When the run stops, each open voice gets
      its Note Off, from the lowest voice upward.
    - The gate closes the highest voice, and no other, when the sampled GATE_MS is less
      than the sampled STEP_MS. If it is not less, the gate never comes, and that voice
      sends its Note Off immediately before its next Note On.
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
