(* Writes the Verilog of the board top level. Usage: dune exec
   board/nexys-4/gen_verilog.exe [output-directory] Run from the repository root; the
   default output directory is board/_generated.

   The elaboration takes the model of the seat whole — the bitstream carries the weights,
   thus the checkpoint is read here and quantized here, and the path is a constant of this
   file and not a flag: one checkpoint, one bitstream, and no runtime configuration. The
   draw is the one [Quantized.Model.of_checkpoint] defaults to.

   THE SHAPE OF THE MODEL COMES FROM THE FILE and the GEOMETRY comes from this one call.
   [Config.of_checkpoint] reads L and H out of the tensor shapes, as the twin does; the
   four numbers below are the ones no checkpoint can hold — the steps of a canvas, the
   lanes of a group, and the passes of the walk. Every width, depth and base of the
   circuit follows from the elaboration they make. *)

open Core

(* Rung 2 of the climb, docs/diffusion_rtl.md: L 64 by H 16, elected through the listening
   gate 2026-08-27. The golden candidate waits behind the fused pair. *)
let checkpoint = "_train/diffusion/coconet/l64-h16-100k.ckpt"

(* T 128 steps of a canvas, G 4 lanes in a group — 192 of the device's 240 DSPs — and N
   512 passes of the walk. At STEP_MS 200 the canvas plays for 25.6 s, and the cost model
   states the pass inside that window. *)
let steps = 128
let lanes = 4
let walk = 512

let () =
  let argv = Sys.get_argv () in
  let dir = if Array.length argv > 1 then argv.(1) else "board/_generated" in
  Core_unix.mkdir_p dir ~perm:0o755;
  let config = Mgen_diffusion.Diffusion.Config.of_checkpoint checkpoint in
  let model = Mgen_diffusion.Quantized.Model.of_checkpoint config checkpoint in
  let e = Mgen_diffusion.Elaboration.create model ~steps ~lanes ~walk in
  print_endline (Mgen_diffusion.Elaboration.to_string e);
  let rtl =
    Hardcaml.Rtl.create Verilog [ Mgen_nexys4.Top.create ~e () ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl
;;
