(** Serial receiver, 8N1, LSB first, with a two-flop synchronizer on the input. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; serial : 'a (** the serial line *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { data : 'a (** the received byte; read it when [valid] is 1 *)
    ; valid : 'a (** high for one cycle when a byte is complete *)
    }
  [@@deriving hardcaml]
end

(** [clocks_per_bit] selects the baud rate at elaboration time. The block samples each bit
    at its center, verifies the start bit, and discards a frame with a bad stop bit. *)
val create : clocks_per_bit:int -> Signal.t I.t -> Signal.t O.t

module For_test : sig
  (** [decode_line wave ~clocks_per_bit] is the bytes of an 8N1 waveform that is sampled
      one time in each cycle: [wave] holds one character for each cycle, '1' or '0'. It is
      the software mirror of this block, thus a test of any serial line reads it with this
      one definition. It discards a frame with a bad start or stop bit. *)
  val decode_line : string -> clocks_per_bit:int -> Bytes.t
end
