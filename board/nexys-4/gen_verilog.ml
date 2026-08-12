(* Writes the Verilog of the board top level. Usage: dune exec
   board/nexys-4/gen_verilog.exe [output-directory] Run from the repository root; the
   default output directory is board/_generated. The elaboration quantizes the king
   checkpoint of the era — the bitstream carries the weights. *)

open Core

let checkpoint = "_train/d64-mk-do01-48k-s4-prog.ckpt"

let () =
  let argv = Sys.get_argv () in
  let dir = if Array.length argv > 1 then argv.(1) else "board/_generated" in
  Core_unix.mkdir_p dir ~perm:0o755;
  let model =
    Mgen_transformer.Quantized.Model.of_checkpoint
      Mgen_transformer.Transformer.Config.baseline
      checkpoint
  in
  let rtl =
    Hardcaml.Rtl.create Verilog [ Mgen_nexys4.Top.create ~model () ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl
;;
