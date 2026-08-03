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

(** The control registers. They are the local storage of the control unit, thus one byte
    of address names each one. *)
module Reg : sig
  val base : int
  val size : int

  (** Bit 0 holds the run state; the board button also toggles it. *)
  val run : int

  val channel : int

  (** 2 bytes, little-endian. *)
  val step_ms : int

  (** 2 bytes, little-endian. *)
  val gate_ms : int

  val velocity : int

  (** 4 bytes, little-endian; the PRNG loads it at the run start. *)
  val seed : int

  (** A write with bit 0 = 1 sends the test message; a read is 1 while a message waits. *)
  val midi_go : int

  val midi_len : int

  (** 3 bytes. *)
  val midi_msg : int
end

(** The power-on values of the control cells. *)
module Default : sig
  (** MIDI channel 3, the S-1 factory default. *)
  val channel : int

  val step_ms : int
  val gate_ms : int
  val velocity : int

  (** Not 0: the PRNG state must never be all zeros. *)
  val seed : int
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
