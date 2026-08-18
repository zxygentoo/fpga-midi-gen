(** The source seat: the model side of the board, whole.

    The block holds a note source, the [Sequencer] that drives it, and the [Midi_out] that
    puts the messages on the line. It holds no logic of its own and it names no model: the
    top level seats a model core with the [source] argument — one line names the model of
    the era, and a later era changes only that line.

    The seat takes the parameter views and gives the MIDI line. The message interface
    between the sequencer and the transmitter stays inside, thus the top level wires no
    handshake: there is one producer of MIDI on this board and it ends at a pin.

    The integration tests drive the parameter views and compare the bytes of the line
    against the messages that the reference composes: the exactness proof of the whole
    model path. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t (** the named views of [Control_regs] *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { serial : 'a (** the MIDI line; it idles at 1 *) } [@@deriving hardcaml]
end

(** [clocks_per_ms] is the step clock of the sequencer and [clocks_per_bit] the baud
    divisor of the MIDI line. Both are board facts, thus the top level states them. *)
val create
  :  clocks_per_ms:int
  -> clocks_per_bit:int
  -> source:(Signal.t Source_intf.I.t -> Signal.t Source_intf.O.t)
  -> Signal.t I.t
  -> Signal.t O.t
