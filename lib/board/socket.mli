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

(** The mounting every socket simulation stands in: the block over a source, the line
    sampled cycle by cycle, and the bytes decoded back off it.

    It stands here and not in one test, because the three integration tests of the
    repository — era one's, era four's and era six's — mounted this same simulation three
    times, and the era that copied it would be the era whose mounting drifted from the one
    before it.

    WHAT IS NOT HERE IS THE RUN LENGTH. Each test states how long its own run is: era four
    counts steps times the period, era six covers a draw budget, era one takes the pink
    step. That rule is the test's, thus [run_for] takes the cycles and states none. *)
module For_test : sig
  type t =
    { inputs : Bits.t ref I.t (** the parameter views, to write with [Harness.set] *)
    ; clear_line : unit -> unit
    (** forget what the line has carried so far — a run opens with this *)
    ; run_for : cycles:int -> unit (** cycle the simulation, sampling the line *)
    ; messages : unit -> int list list
    (** the bytes the line has carried since [clear_line], as messages. Every message of
        the sequencer is three bytes, thus the byte stream divides into messages with no
        ambiguity. *)
    }

  (** [harness ~clocks_per_ms ~clocks_per_bit ~source ()] mounts [source] in the seat.
      [source] reads the seed off the parameter views, thus a test writes the seed like
      any other parameter and the simulation needs no rebuild. *)
  val harness
    :  clocks_per_ms:int
    -> clocks_per_bit:int
    -> source:
         (Signal.t Control_regs.Params.t
          -> Signal.t Source_intf.I.t
          -> Signal.t Source_intf.O.t)
    -> unit
    -> t
end
