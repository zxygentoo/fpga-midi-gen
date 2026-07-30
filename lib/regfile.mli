(** The control register file: [Abi.Reg.Ctl.size] cells with their power-on values from
    [Abi.Default]. The cell semantics beyond plain memory (the doorbell, the run button)
    arrive with the blocks that own them. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; wr_en : 'a (** writes [wr_data] to the cell [wr_idx] *)
    ; wr_idx : 'a (** a cell index; 0 is the cell at [Abi.Reg.Ctl.base] *)
    ; wr_data : 'a (** the byte to write *)
    ; rd_idx : 'a (** the cell index for [rd_data] *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { rd_data : 'a (** the value of the cell [rd_idx]; the read is combinational *)
    ; cells : 'a
    (** all cells as one vector; the cell at [Abi.Reg.Ctl.base] is the low byte *)
    }
  [@@deriving hardcaml]
end

val create : Signal.t I.t -> Signal.t O.t
