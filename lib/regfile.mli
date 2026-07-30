(** The control register file: [Abi.Reg.Ctl.size] cells with their power-on values from
    [Abi.Default]. The read is combinational. The cell semantics beyond plain memory (the
    doorbell, the run button) arrive with the blocks that own them. *)

open Hardcaml

type t =
  { rd_data : Signal.t
  ; cells : Signal.t
  (** all cells as one vector; the cell at [Abi.Reg.Ctl.base] is the low byte *)
  }

val create
  :  clock:Signal.t
  -> clear:Signal.t
  -> wr_en:Signal.t
  -> wr_idx:Signal.t
  -> wr_data:Signal.t
  -> rd_idx:Signal.t
  -> t
