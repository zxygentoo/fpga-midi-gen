(** The source seat: the one block between a model and the sequencer.

    The block connects a note source to the [Sequencer]; it holds no logic and it does not
    name a model. The top level seats a model core with the [source] argument — one line
    names the model of the era, and a later era changes only that line.

    The integration tests drive the parameter views and compare the message stream against
    the messages that the reference composes: the exactness proof of the whole model path. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t (** the named views of [Control_regs] *)
    ; midi_ready : 'a (** from [Midi_merge]: 1 when the MIDI path takes the message *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { midi : 'a Midi.Rtl.Message.t (** the model source *) }
  [@@deriving hardcaml]
end

val create
  :  clocks_per_ms:int
  -> source:(Signal.t Source_intf.I.t -> Signal.t Source_intf.O.t)
  -> Signal.t I.t
  -> Signal.t O.t
