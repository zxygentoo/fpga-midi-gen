(** The sequencer: the run engine of the model.

    The block reads the parameter views, drives the note source, and gives the model
    messages on the [Midi.Message] interface. The rules are the ones of
    [docs/pink_rtl.md]:

    - A prescaler divides the clock by [clocks_per_ms]. The run start resets it, thus a
      run is the same cycle for cycle in each simulation and on the board.
    - The source goes to its origin at the run start and at no other time.
    - At each step boundary the block samples RUN, STEP_MS and GATE_MS: a change applies
      at the next step. A sampled STEP_MS of 0 counts as 1.
    - The open-note register keeps the note and the channel of each Note On, and the Note
      Off takes the stored pair. When the run stops, the open note gets its Note Off.
    - The gate closes the note when the sampled GATE_MS is less than the sampled STEP_MS;
      otherwise the Note Off goes immediately before the next Note On.
    - The millisecond count does not pause while the merge stalls a message, thus the beat
      does not drift. *)

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
    { midi : 'a Midi.Message.t (** the model source *)
    ; source_rewind : 'a (** a strobe: the source goes to its origin — the run start *)
    ; source_step : 'a (** a strobe: give the note of one step *)
    }
  [@@deriving hardcaml]
end

val create : clocks_per_ms:int -> Signal.t I.t -> Signal.t O.t
