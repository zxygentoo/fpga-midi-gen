(** The control port: the wire-protocol engine and the control register file.

    The behavior is the one of [docs/abi.md]: a frame that does not decode gets no
    response, a write applies its bytes in the sequence of increasing addresses, and a
    rejected access changes no cell.

    After power-on and after clear, the port writes the control defaults into the register
    file (state [Init]); then it serves one transaction at a time. While the port serves a
    transaction, it ignores the input stream: a frame in that interval gets no response,
    and the sender must repeat it.

    A write applies one byte each cycle: a multi-byte value is torn between these cycles.
    The wire protocol cannot observe this, because the response comes after the last byte.
    A hardware block that consumes a multi-byte cell must take the value on the write of
    its last byte, as the SEED rule of the ABI does, or sample the cells only when [state]
    is [State.Ready].

    The MSG cells are the test-message doorbell of the ABI, and the port holds its send
    machine. A write of a value with bit 0 = 1 to MSG_GO sends MSG_LEN bytes of MSG on the
    [midi_data] stream, and a read of MSG_GO is 1 while the message waits. The cells are
    the one storage of the message: the sender reads them as it sends. Thus the poll rule
    of the ABI — read MSG_GO as 0 before a write that covers a cell of MSG, MSG_LEN or
    MSG_GO — is the rule that keeps each message complete on the stream. The port ignores
    the send bit while a message waits, and also when MSG_LEN is not in 1 to
    [Abi.Limits.max_msg_len]. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a (** the byte stream from the host UART *)
    ; in_valid : 'a (** a strobe: [in_data] holds one stream byte *)
    ; hold : 'a (** from the consumer: 1 stalls the stream *)
    ; midi_hold : 'a (** from the MIDI transmitter: 1 stalls the message stream *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { out_data : 'a (** the response byte stream: COBS frames with their delimiters *)
    ; out_valid : 'a (** the transmitter takes [out_data] when [hold] is 0 *)
    ; midi_data : 'a (** the test-message byte stream to the MIDI transmitter *)
    ; midi_valid : 'a (** the transmitter takes [midi_data] when [midi_hold] is 0 *)
    ; state : 'a (** [State.init], [State.ready] or [State.busy] *)
    }
  [@@deriving hardcaml]
end

(** The meaning of [O.state]. *)
module State : sig
  type t =
    | Init (** the port loads the control defaults; the cells are not valid *)
    | Ready (** idle; the cells are valid and stable *)
    | Busy (** a transaction is in progress; a write can tear a multi-byte cell *)

  (** the encoding on the [O.state] wires *)
  val to_code : t -> int
end

val create : Signal.t I.t -> Signal.t O.t
