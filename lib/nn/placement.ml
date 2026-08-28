(* The placement family — see placement.mli. The rules stand HERE, one time for the whole
   repository, rather than once in each unit that obeys them. *)

open Hardcaml

let no_dsp product = Signal.add_attribute product (Rtl_attribute.Vivado.use_dsp false)
let replica copy = Signal.add_attribute copy (Rtl_attribute.Vivado.dont_touch true)
let slice_rows = 8
let slices_for ~rows = (rows + slice_rows - 1) / slice_rows
let block_ram = Rtl_attribute.Vivado.Ram_style.block
