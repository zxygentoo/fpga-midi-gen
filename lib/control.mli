(** The host control: the contract between the host drivers and the FPGA.

    This module is the single definition of the host-control constants and of the
    wire-frame codec. The drivers use it directly. The RTL reads the constants at
    elaboration time. The normative description is [docs/host_control.md]. *)

(** The sizes of the host control, in bytes. A wire payload is a header and then DATA. *)
module Constants : sig
  (** The number of bytes in OP, ADDR and LEN. *)
  val request_header_bytes : int

  (** The number of bytes in OP and STATUS. *)
  val response_header_bytes : int

  (** The maximum data bytes in one read or write. *)
  val max_data_len : int

  (** The largest payload: the request header and the largest DATA. The FPGA discards a
      frame with a longer payload. *)
  val max_payload_bytes : int
end

module Op : sig
  val read : int
  val write : int

  (** The response OP is the request OP plus this flag. *)
  val response_flag : int
end

module Status : sig
  type t =
    | Ok
    | Bad_op
    | Bad_address
    | Bad_length

  val to_code : t -> int
  val of_code : int -> t option

  (** The name of the status in the diagnostic output. *)
  val to_string : t -> string
end

(** The power-on values of the control cells. [Reg.fields] gives the value of each cell,
    and these are the names that a driver reads. *)
module Default : sig
  (** MIDI channel 3, the S-1 factory default. *)
  val channel : int

  val step_ms : int
  val gate_ms : int
  val velocity : int

  (** Not 0: the PRNG state must never be all zeros. *)
  val seed : int
end

(** The control registers. They are the local storage of the control unit, thus one byte
    of address names each one. *)
module Reg : sig
  val base : int
  val size : int

  (** Bit 0 holds the run state; the board button also toggles it. *)
  val run : int

  val channel : int
  val step_ms : int
  val gate_ms : int
  val velocity : int

  (** The PRNG loads it at the run start. *)
  val seed : int

  (** A write with bit 0 = 1 sends the test message; a read is 1 while a message waits. *)
  val midi_go : int

  val midi_len : int
  val midi_msg : int

  (** One register: its name, the address of its first byte, the number of bytes, and its
      power-on value. A value of more than one byte is little-endian. *)
  type field =
    { name : string
    ; address : int
    ; width : int
    ; default : int
    }

  (** Each register, from the first address upward. The RTL cells, their power-on values
      and the driver dump all read this one table, thus a width has one definition. *)
  val fields : field list

  (** [width_of address] is the number of bytes of the register at [address]. It raises
      [Not_found_s] when no register starts there. *)
  val width_of : int -> int
end

type request =
  | Read of
      { addr : int
      ; len : int
      }
  | Write of
      { addr : int
      ; data : Bytes.t
      }

type response =
  { op : int
  ; status : Status.t
  ; data : Bytes.t
  }

(** [doorbell_write message] is the address and the bytes of the one ascending write that
    sends [message] to the MIDI output: MIDI_MSG, then MIDI_LEN, then MIDI_GO with bit 0
    at 1. A write applies at one time, thus this one burst does the whole operation, and
    the layout of the burst has one definition. It raises [Invalid_argument] when
    [message] is not 1 to [Midi.max_message_bytes] bytes. *)
val doorbell_write : int list -> int * Bytes.t

(** [encode_request r] is the complete wire frame, with the COBS encoding and the
    delimiter. It raises [Invalid_argument] when a field does not fit the wire format. *)
val encode_request : request -> Bytes.t

(** [decode_response frame] parses one delimited wire frame. *)
val decode_response : Bytes.t -> (response, string) result

(** The board side of the codec. The hardware encodes each response, thus only a test that
    fakes a board needs this. *)
module For_test : sig
  val encode_response : response -> Bytes.t
end
