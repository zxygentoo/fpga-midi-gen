(** The control cells: the storage, the write decode, the named views, the value that each
    cell reads as, and the doorbell source.

    A control cell is not a memory location. Many blocks look at a cell continuously, and
    one block writes it. Therefore the block gives named views with the natural width of
    each value, and no consumer needs an address.

    Each cell carries its power-on value in the bitstream, and a clear gives the same
    value. Thus the cells are correct at cycle 0 and the section needs no init walk.

    A write fills a shadow copy one byte in each cycle, and one [commit] strobe moves all
    the bytes into the live cells. Therefore a view never shows a part of one write and a
    part of the next, and a value of more than one byte does not tear.

    The doorbell is a cell behavior, thus it stays here. The block latches MIDI_MSG and
    MIDI_LEN at the ring and gives an ordinary message source. Nothing in the MIDI path
    knows that the doorbell is special. *)

open Hardcaml

(** The named views of the control cells: the parameters of the model. MIDI_MSG, MIDI_LEN
    and MIDI_GO have no view, because the block gives the doorbell as a message source. *)
module Params : sig
  type 'a t =
    { run : 'a (** the run state, in bit 0 *)
    ; channel : 'a (** the MIDI channel of the model, 0 to 15 *)
    ; step_ms : 'a (** the step period in ms *)
    ; gate_ms : 'a (** the gate time in ms *)
    ; velocity : 'a (** the note velocity *)
    ; seed : 'a (** the PRNG seed *)
    }
  [@@deriving hardcaml]
end

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; write_enable : 'a (** writes [write_data] into the shadow cell *)
    ; write_address : 'a (** the cell index of the write *)
    ; write_data : 'a (** the byte to write *)
    ; commit : 'a (** a strobe: copy the shadow into the live cells *)
    ; read_address : 'a (** the cell index that [read_data] answers *)
    ; run_toggle : 'a (** a strobe from the board button: invert bit 0 of RUN *)
    ; doorbell_ready : 'a
    (** from [Midi_merge]: 1 when the MIDI path takes the doorbell message *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { params : 'a Params.t (** the named views; each one is stable *)
    ; read_data : 'a
    (** the byte at [read_address]; the read is combinational. MIDI_GO gives 1 while a
        doorbell message waits, and each other cell gives its stored byte *)
    ; doorbell : 'a Midi.Message.t (** the test message, as a message source *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
