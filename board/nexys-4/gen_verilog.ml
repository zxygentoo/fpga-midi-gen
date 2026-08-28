(* Writes the Verilog of the board top level. Usage: dune exec
   board/nexys-4/gen_verilog.exe -- [-int8 PATH] [output-directory] Run from the
   repository root; the default output directory is board/_generated.

   The elaboration takes the model of the seat whole — the bitstream carries the weights,
   thus the model is read here and the path is a default of this file and not a runtime
   configuration: one contract file, one bitstream.

   THE MODEL IS A CONTRACT FILE AND THE QUANTIZATION IS NOT HERE.
   jax/diffusion/quantized.py folds the norm and states the int8 image, one time, and
   writes the file this reads: `uv run python -m diffusion.infer quantize --ckpt C --out
   C.int8`. The quantization happens above the seam, one time, thus the file is the only
   thing that crosses it for a build; the gate that holds it is
   `test_g1_the_quantizer_states_the_golden_netlist` — the netlist of this program must
   stay the netlist the flash carries.

   THE SHAPE OF THE MODEL COMES FROM THE FILE and the GEOMETRY comes from this one call. A
   contract file states L and H in its own tensor shapes. The three numbers below are the
   ones no file can hold — the steps of a sheet, the lanes of a group, and the passes of
   the walk. Every width, depth and base of the circuit follows from the elaboration they
   make. *)

open Core

(* Rung 3 of the climb, docs/diffusion_rtl.md: the golden candidate, L 48 by H 20, elected
   2026-08-25 and on the board through the fused pair and the timing cuts 2026-08-28. *)
let contract = "_train/diffusion/coconet/l48-h20-100k.int8"

(* T 128 steps of a sheet, G 5 lanes in a group — all 240 of the device's DSPs — and N 512
   passes of the walk. At STEP_MS 200 the sheet plays for 25.6 s, and the cost model
   states the pass inside that window: 4.30 M cycles, 22.0 s of draw. G 4 would need 215
   ms a step and miss the window. *)
let steps = 128
let lanes = 5
let walk = 512

let command =
  Command.basic
    ~summary:"write the Verilog of the board top level"
    (let%map_open.Command int8 =
       flag
         "-int8"
         (optional_with_default contract string)
         ~doc:"PATH the contract file of the quantized model"
     and dir =
       anon (maybe_with_default "board/_generated" ("output-directory" %: string))
     in
     fun () ->
       Core_unix.mkdir_p dir ~perm:0o755;
       let model = Mgen_diffusion.Model.of_int8_checkpoint int8 in
       let e = Mgen_diffusion.Elaboration.create model ~steps ~lanes ~walk in
       print_endline (Mgen_diffusion.Elaboration.to_string e);
       let rtl =
         Hardcaml.Rtl.create Verilog [ Mgen_nexys4.Top.create ~e () ]
         |> Hardcaml.Rtl.full_hierarchy
         |> Rope.to_string
       in
       Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl)
;;

let () = Command_unix.run command
