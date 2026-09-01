(* Writes the Verilog of ONE UNIT for an out-of-context reading. Usage: dune exec
   bin/gen_probe.exe -- UNIT [lanes]

   The units of era six take their weights, their columns and their logits as ports, thus
   a probe needs no checkpoint and no drawn weights: the shape is the whole input, and P
   is the board's 48. [lanes] is G where a unit has one, and the sweep the round wants is
   1, 4 and 5 — the fallback rung, the elected rung, and the geometry the fused pair would
   need at the top of the ladder.

   The array is the round's timing risk. The epilogue and the draw are the round's DSP
   rule: THE ARRAY OWNS THE DSPS AND EVERYTHING ELSE IS LUTS, because the fused rung needs
   all 240 for the lanes. Both units pin their products with the Vivado attribute, and
   these probes are what say the tools honoured it.

   The circuit is always named [probe], thus probe.tcl states one top. Out of context,
   because no device carries a 768-bit port. *)

open Core

let rows = Mgen_diffusion.Model.rows

let circuit unit lanes =
  match unit with
  | "array" ->
    let module Unit =
      Mgen_diffusion.Column_array.Make (struct
        let rows = rows
        let lanes = lanes
      end)
    in
    let module Probe = Hardcaml.Circuit.With_interface (Unit.I) (Unit.O) in
    Probe.create_exn ~name:"probe" Unit.create
  | "epilogue" ->
    let module Unit =
      Mgen_diffusion.Epilogue.Make (struct
        let rows = rows
        let lanes = lanes
      end)
    in
    let module Probe = Hardcaml.Circuit.With_interface (Unit.I) (Unit.O) in
    Probe.create_exn ~name:"probe" Unit.create
  | "draw" ->
    let module Unit =
      Mgen_diffusion.Draw.Make (struct
        let classes = rows
      end)
    in
    let module Probe = Hardcaml.Circuit.With_interface (Unit.I) (Unit.O) in
    let temper = Mgen_nn.Quantized.Constants.temper_at_one in
    Probe.create_exn ~name:"probe" (Unit.create ~temper)
  | "forward" ->
    (* RING 2: the whole column engine — the weight ROM through the broadcast trees into
       the array, and the store's read into the window. It is the ring where era four's
       retiming trap lives on home ground, thus the synthesis log is read for the absorbed
       DSP registers as well as the numbers.

       The weights are DRAWN and not a checkpoint's: a timing reading needs no correct
       data, and the widths of this design follow the RULES and not a model's own peak —
       [Elaboration.shift_bits] states that argument — thus a drawn model elaborates the
       netlist a trained one does. *)
    let model = Mgen_diffusion.Model.For_test.drawn ~layers:16 ~width:16 ~seed:1 in
    let module Unit =
      Mgen_diffusion.Forward.Make (struct
        let e = Mgen_diffusion.Elaboration.create model ~steps:128 ~lanes ~walk:512
      end)
    in
    let module Probe = Hardcaml.Circuit.With_interface (Unit.I) (Unit.O) in
    Probe.create_exn ~name:"probe" Unit.create
  | other -> invalid_argf "no unit is named %s: array, epilogue, draw or forward" other ()
;;

let () =
  let argv = Sys.get_argv () in
  let unit = if Array.length argv > 1 then argv.(1) else "array" in
  let lanes = if Array.length argv > 2 then Int.of_string argv.(2) else 4 in
  let dir = "board/_generated" in
  Core_unix.mkdir_p dir ~perm:0o755;
  let rtl =
    Hardcaml.Rtl.create Verilog [ circuit unit lanes ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.write_all (Filename.concat dir "probe.v") ~data:rtl;
  printf "%s at P %d by G %d\n" unit rows lanes
;;
