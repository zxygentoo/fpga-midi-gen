(* Writes the Verilog of the board top level. Usage: dune exec
   board/nexys-4/gen_verilog.exe [output-directory] Run from the repository root; the
   default output directory is board/_generated. The elaboration takes the model of the
   seat whole — the bitstream carries it — and [Pink] sits there while era four is
   written. *)

open Core

let () =
  let argv = Sys.get_argv () in
  let dir = if Array.length argv > 1 then argv.(1) else "board/_generated" in
  Core_unix.mkdir_p dir ~perm:0o755;
  let rtl =
    Hardcaml.Rtl.create
      Verilog
      [ Mgen_nexys4.Top.create ~model:Mgen_pink.Pink.default () ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl
;;
