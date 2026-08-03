(** The byte level of a [Bytes.t]: read and write integers, and show the bytes.

    [Base.Bytes] gives [get] and [set] on characters only. These functions add the
    operations that a binary protocol needs. *)

(** [byte b i] is the value of the byte at [i]. *)
val byte : Bytes.t -> int -> int

(** [set_byte b i v] writes [v] to the byte at [i]. It raises [Invalid_argument] if [v] is
    not in the range 0 to 255. *)
val set_byte : Bytes.t -> int -> int -> unit

(** [uint_le b ~pos ~width] is the value of the [width] bytes at [pos], in the
    little-endian order. *)
val uint_le : Bytes.t -> pos:int -> width:int -> int

(** [hex b] is the bytes of [b] as hexadecimal text: two lowercase digits for each byte,
    and one space between the bytes. *)
val hex : Bytes.t -> string
