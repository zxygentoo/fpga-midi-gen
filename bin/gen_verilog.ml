(* Writes the Verilog of the board top level over ONE ERA'S SOURCE. Usage: dune exec
   bin/gen_verilog.exe -- ERA [flags] [output-directory] Run from the repository root; the
   default output directory is board/_generated.

   THE TOP LEVEL NAMES NO ERA -- [Top] takes the source as an argument -- thus one program
   elaborates every era and this is the only place that chooses. Era six holds the board;
   the others keep their netlists alive, and `jax/tests/test_parity.py` pins the md5 of
   each. Neither an elaboration nor a quantizer can move without that gate saying so.

   THE MODEL IS A CONTRACT FILE AND THE QUANTIZATION IS NOT HERE. `jax/<era>/quantized.py`
   folds the norm and states the int8 image, one time, and writes the file this reads. The
   default of every era is `weights/<era>.int8`, which `make verilog-<era>` writes from
   the checkpoint beside it; `weights/README.md` holds that command and why the contract
   file is derived and not committed.

   ERA ONE READS NO FILE. Pink noise is parameters and not weights, thus `Pink.default` is
   the model and this is the one era that elaborates on a bare clone.

   THE SHAPE OF A MODEL COMES FROM ITS FILE and the GEOMETRY comes from a flag. A contract
   file states every width and depth in its own tensor shapes; era six's three numbers --
   the steps of a sheet, the lanes of a group and the passes of the walk -- are the ones
   no file can hold, and every base of the circuit follows from the elaboration they make. *)

open Core

(* what `make verilog-<era>` writes beside the committed checkpoint; weights/README.md
   holds why the contract file is derived and not committed *)
let elected era = sprintf "weights/%s.int8" era

let int8_flag era =
  let%map_open.Command path =
    flag
      "-int8"
      (optional_with_default (elected era) string)
      ~doc:"PATH the contract file of the quantized model"
  in
  path
;;

(* One era's top level, written as top.v. The model arrives as a THUNK, thus a file is
   read when the command runs and never while the command line is parsed. *)
let top ~summary ~model ~source =
  Command.basic
    ~summary
    (let%map_open.Command model
     and dir =
       anon (maybe_with_default "board/_generated" ("output-directory" %: string))
     in
     fun () ->
       Core_unix.mkdir_p dir ~perm:0o755;
       let rtl =
         Hardcaml.Rtl.create
           Verilog
           [ Mgen_nexys4.Top.create ~source:(source (model ())) () ]
         |> Hardcaml.Rtl.full_hierarchy
         |> Rope.to_string
       in
       Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl)
;;

let pink =
  top
    ~summary:"era one: the pink noise source, which reads no file"
    ~model:(Command.Param.return (fun () -> Mgen_pink.Pink.default))
    ~source:(fun model -> Mgen_pink.Source.create ~model)
;;

let transformer =
  top
    ~summary:"era four: the step-frame transformer"
    ~model:
      (let%map_open.Command path = int8_flag "transformer" in
       fun () -> Mgen_transformer.Model.of_int8_checkpoint path)
    ~source:(fun model -> Mgen_transformer.Source.create ~model)
;;

let mamba =
  top
    ~summary:"era five: the state-space model"
    ~model:
      (let%map_open.Command path = int8_flag "mamba" in
       fun () -> Mgen_mamba.Model.of_int8_checkpoint path)
    ~source:(fun model -> Mgen_mamba.Source.create ~model)
;;

(* Rung 3 of the climb, docs/diffusion_rtl.md: the golden candidate, L 48 by H 20, elected
   2026-08-25 and on the board through the fused pair and the timing cuts 2026-08-28.

   T 128 steps of a sheet, G 5 lanes in a group -- all 240 of the device's DSPs -- and N
   512 passes of the walk. At STEP_MS 200 the sheet plays for 25.6 s, and the cost model
   states the pass inside that window: 4.30 M cycles, 22.0 s of draw. G 4 would need 215
   ms a step and miss the window. *)
let diffusion =
  top
    ~summary:"era six: the masked sheet, which holds the board"
    ~model:
      (let%map_open.Command path = int8_flag "diffusion"
       and steps =
         flag "-steps" (optional_with_default 128 int) ~doc:"T the steps of a sheet"
       and lanes =
         flag "-lanes" (optional_with_default 5 int) ~doc:"G the lanes of a group"
       and walk =
         flag "-walk" (optional_with_default 512 int) ~doc:"N the passes of the walk"
       in
       fun () ->
         let model = Mgen_diffusion.Model.of_int8_checkpoint path in
         let e = Mgen_diffusion.Elaboration.create model ~steps ~lanes ~walk in
         (* the geometry every base of the circuit follows from: a build's log wants it *)
         print_endline (Mgen_diffusion.Elaboration.to_string e);
         e)
    ~source:(fun e -> Mgen_diffusion.Source.create ~e)
;;

let command =
  Command.group
    ~summary:"write the Verilog of the board top level over one era's source"
    [ "pink", pink; "transformer", transformer; "mamba", mamba; "diffusion", diffusion ]
;;

let () = Command_unix.run command
