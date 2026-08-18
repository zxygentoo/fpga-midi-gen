(* Writes the Verilog of the board top level. Usage: dune exec
   board/nexys-4/gen_verilog.exe [output-directory] Run from the repository root; the
   default output directory is board/_generated.

   The elaboration takes the model of the seat whole — the bitstream carries the weights,
   thus the checkpoint is read here and quantized here, and the path is a constant of this
   file and not a flag: one checkpoint, one bitstream, and no runtime configuration. The
   draw is the one the ear elected, which [Quantized.Model.of_checkpoint] defaults to. *)

open Core

(* the model the ear elected on 2026-08-18 *)
let checkpoint = "_train/d64-frame-do03-96k-s6-l6-nopos-span4.ckpt"

let () =
  let argv = Sys.get_argv () in
  let dir = if Array.length argv > 1 then argv.(1) else "board/_generated" in
  Core_unix.mkdir_p dir ~perm:0o755;
  (* the file states the width and the layer count; the heads, the context and the slope
     span are the draw of the era and stand in [Transformer.Config.baseline] *)
  let { Mgen_transformer.Transformer.Config.heads; context; slope_span; _ } =
    Mgen_transformer.Transformer.Config.baseline
  in
  let config =
    Mgen_transformer.Transformer.Config.of_checkpoint
      checkpoint
      ~heads
      ~context
      ~slope_span
  in
  let model = Mgen_transformer.Quantized.Model.of_checkpoint config checkpoint in
  let rtl =
    Hardcaml.Rtl.create Verilog [ Mgen_nexys4.Top.create ~model () ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl
;;
