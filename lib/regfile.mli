(** The control register file: [Abi.Reg.Ctl.size] cells with their power-on values from
    [Abi.Default]. The cell semantics beyond plain memory (the doorbell, the run button)
    arrive with the blocks that own them. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; write_enable : 'a (** writes [write_data] to the cell [address] *)
    ; address : 'a (** the cell index; 0 is the cell at [Abi.Reg.Ctl.base] *)
    ; write_data : 'a (** the byte to write *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { read_data : 'a
    (** the value of the cell [address]; the read is combinational, and during a write it
        is the old value *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
