(** The host side of the ABI: transactions over the transport to the control port.

    A [t] carries bytes to the control port and back: the mgt tool supplies the serial
    port, and a test can supply a script or a simulation. A transaction sends one request
    frame and collects one response frame.

    The layer holds no clock: [receive] blocks until a byte arrives, and a transaction
    with no response does not return. Bound it outside the process when a bound is
    necessary, for example with timeout(1). Each operation is idempotent, thus the caller
    can simply run a failed command again. *)

type t =
  { send : Bytes.t -> unit (** writes a complete frame to the line *)
  ; receive : unit -> char (** the next byte of the stream; blocks until it arrives *)
  }

type error =
  | Garbled
  (** the response does not decode or does not answer the request; run the command again *)
  | Nak of Abi.Status.t (** the port rejected the access; no cell changed *)

(** [resync t] sends one delimiter: whatever partial frame an earlier session left in the
    port, the delimiter ends it. Call it one time after the port opens. The transport
    provider discards stale input on its own, for example with tcflush. *)
val resync : t -> unit

val read : t -> address:int -> length:int -> (Bytes.t, error) result
val write : t -> address:int -> data:Bytes.t -> (unit, error) result
