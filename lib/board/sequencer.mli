(** The sequencer: the run engine of the model.

    The block reads the parameter views, drives the note source, and gives the model
    messages on the [Midi.Rtl.Message] interface. The rules are the ones of
    [docs/pink_rtl.md] and [docs/transformer_rtl.md]:

    - A prescaler divides the clock by [clocks_per_ms]. The run start resets it, thus a
      run is the same cycle for cycle in each simulation and on the board.
    - The source goes to its origin at the run start and at no other time.
    - At each step boundary the block samples RUN and STEP_MS: a change applies at the
      next step. A sampled STEP_MS of 0 counts as 1.
    - **At each step the source states one frame, and the block decodes it.** It holds the
      set of pitches that sound; the frame states the set that must sound. It sends the
      releases — the first set less the second — and then the strikes — the second less
      the first — each in ascending pitch order. This is the rule of
      [Frame.events_of_frames], and that document states why a seat walk breaks on the
      exchange and the unison.
    - **The run stop is a silent frame**, thus the stop needs no rule of its own: the
      release walk closes every pitch that sounds, ascending, and the block returns to
      rest.
    - Each held pitch keeps the channel of its Note On, and its Note Off takes that pair.
      Therefore a change of CHANNEL inside a run cannot leave a note open.
    - The block strobes [step] only while the source is idle, thus a source that has not
      finished its draw holds the boundary; the millisecond count does not pause while it
      waits, thus the beat does not drift. One step sends at most two messages for each
      voice, which is about 7.7 ms of line time for four voices. A step that is shorter
      than its messages stretches.

    The block sends no Note Off of its own. Every message follows from the frames: there
    is no steal, because a frame states the whole sonority, and there is no gate. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t (** the named views; each one is stable *)
    ; source : 'a Source_intf.O.t (** the outputs of the note source *)
    ; midi_ready : 'a (** from [Midi_out]: 1 when the line takes the message *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { midi : 'a Midi.Rtl.Message.t (** the model source *)
    ; source_rewind : 'a (** a strobe: the source goes to its origin — the run start *)
    ; source_step : 'a (** a strobe: state one step of music *)
    }
  [@@deriving hardcaml]
end

val create : clocks_per_ms:int -> Signal.t I.t -> Signal.t O.t
