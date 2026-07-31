(** The driver side of the host control: transactions with the control port.

    A [t] is the transport: it carries request frames to the control port and response
    frames back. [serial] makes the transport over an open serial port; the tests inside
    the module script the byte level directly.

    The layer holds no clock. A transaction blocks until the response arrives, and a
    transaction with no response does not return: when a bound is necessary, set it
    outside the process, for example with timeout(1). Each operation is idempotent, thus
    after an error the caller can simply run it again. *)

type t

type error =
  | Garbled
  (** the response does not decode or does not answer the request; run the command again *)
  | Nak of Control.Status.t (** the port rejected the access; no cell changed *)

(** [serial ~baud fd] is the transport over an open serial port: raw 8N1 at [baud] and a
    blocking read. The caller opens [fd] and owns its lifetime. *)
val serial : baud:int -> Core_unix.File_descr.t -> t

(** [read t ~address ~length] is the value of the [length] cells at [address]. *)
val read : t -> address:int -> length:int -> (Bytes.t, error) result

(** [write t ~address ~data] writes [data] to the cells at [address]. *)
val write : t -> address:int -> data:Bytes.t -> (unit, error) result

(** [resync t] discards stale input and ends any partial frame in the port: whatever an
    earlier session left behind, one delimiter ends it. Call it one time after the port
    opens. *)
val resync : t -> unit
