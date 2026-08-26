(* Writes the Verilog of ONE UNIT for an out-of-context reading. Usage: dune exec
   board/nexys-4/gen_probe.exe [lanes] [output-directory]

   The column array is the whole timing risk of era six — docs/diffusion_rtl.md says so —
   and it is the first thing this round can measure. The unit takes its weights and its
   columns as ports, thus a probe needs no checkpoint and no drawn weights: the shape is
   the whole input, and P is the board's 48. [lanes] is G, and the sweep the round wants
   is 1, 4 and 5 — the fallback rung, the elected rung, and the geometry the fused pair
   would need at the top of the ladder.

   Out of context, because no device carries a 768-bit port. The reading that follows is
   the array's own logic and routing; what the ports hide is the store and the window on
   the other side of them, and probe.tcl states the delay it charges for that. *)

open Core

let () =
  let argv = Sys.get_argv () in
  let lanes = if Array.length argv > 1 then Int.of_string argv.(1) else 4 in
  let dir = if Array.length argv > 2 then argv.(2) else "board/_generated" in
  Core_unix.mkdir_p dir ~perm:0o755;
  let module Array_ =
    Mgen_diffusion.Column_array.Make (struct
      let rows = Mgen_diffusion.Diffusion.rows
      let lanes = lanes
    end)
  in
  let module Probe = Hardcaml.Circuit.With_interface (Array_.I) (Array_.O) in
  let rtl =
    Hardcaml.Rtl.create Verilog [ Probe.create_exn ~name:"column_array" Array_.create ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.write_all (Filename.concat dir "probe.v") ~data:rtl;
  printf
    "column_array at P %d by G %d: %d lanes\n"
    Mgen_diffusion.Diffusion.rows
    lanes
    (Mgen_diffusion.Diffusion.rows * lanes)
;;
