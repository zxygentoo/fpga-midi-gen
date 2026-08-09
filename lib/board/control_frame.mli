(** The wire codec of the host control: one request frame out, one response frame back.

    The framing is COBS; see [Cobs] in [lib/cobs.ml]. All values of more than one byte are
    little-endian, on the wire and in the cells; [Bytes_util] holds the byte accessors.
    [Control_intf] gives the constants that the frames carry, and the normative
    description is [docs/host_control.md].

    The host and the board hold one half of the codec each: the drivers encode a request
    and decode a response, and the hardware does the other two. Therefore [encode_request]
    and [decode_response] are the driver operations, and the board half is here only so
    that a test can play the board. *)

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
  ; status : Control_intf.Status.t
  ; data : Bytes.t
  }

(** [encode_request r] is the complete wire frame, with the COBS encoding and the
    delimiter. It raises [Invalid_argument] when a field does not fit the wire format. *)
val encode_request : request -> Bytes.t

(** [decode_response frame] parses one delimited wire frame. The frame comes off a serial
    line, thus a frame that does not parse is an ordinary result and not an error of the
    program. *)
val decode_response : Bytes.t -> (response, string) result

(** The board side of the codec. The hardware encodes each response, thus only a test that
    fakes a board needs this. *)
module For_test : sig
  val encode_response : response -> Bytes.t
end
