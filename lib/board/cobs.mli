(** COBS: Consistent Overhead Byte Stuffing.

    The delimiter is fixed at zero: COBS group headers are the counts 1 to 255, thus any
    other delimiter can collide with a header byte.

    The encoder makes the minimal form: no empty group after a final 254-byte block. The
    decoder accepts both the minimal form and the phantom-zero form. *)

(** The frame delimiter, [' \000']. A receiver splits the stream after each delimiter
    byte. *)
val delimiter : char

(** [encode src] is the COBS body of [src] plus the zero delimiter. *)
val encode : Bytes.t -> Bytes.t

(** [decode frame] is the decoded body of [frame]. The delimiter must be the last byte of
    [frame], and the only zero byte in it. *)
val decode : Bytes.t -> (Bytes.t, string) result
