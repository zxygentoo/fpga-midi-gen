(* Writes the Verilog of the board top level. Usage: dune exec
   board/nexys-4/gen_verilog.exe [output-directory] Run from the repository root; the
   default output directory is board/_generated.

   The elaboration takes the model of the seat whole — the bitstream carries the weights,
   thus the checkpoint is read here and quantized here, and the path is a constant of this
   file and not a flag: one checkpoint, one bitstream, and no runtime configuration. The
   draw is the one [Quantized.Model.of_checkpoint] defaults to.

   Every width, the PLAN and the ALiBi span come from the file. Era four had to be told
   its heads, its context and its span; here the head count and the state width size
   tensors, the shape of each layer group names its kind, and the span stands after the
   last one. The one number the file cannot hold is the depth of the ring at inference,
   and [Mamba.elected_ring] states it. *)

open Core

(* the elected model of era five, docs/mamba.md: six blocks, the Zamba head, the
   feed-forward *)
let checkpoint = "_train/mamba/d64-mamba-k4-n16-zamba-ff-do03-48k-s7.ckpt"

let () =
  let argv = Sys.get_argv () in
  let dir = if Array.length argv > 1 then argv.(1) else "board/_generated" in
  Core_unix.mkdir_p dir ~perm:0o755;
  let config = Mgen_mamba.Mamba.Config.of_checkpoint checkpoint in
  let model = Mgen_mamba.Quantized.Model.of_checkpoint config checkpoint in
  let rtl =
    Hardcaml.Rtl.create Verilog [ Mgen_nexys4.Top.create ~model () ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl
;;
