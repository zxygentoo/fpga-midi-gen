(* Writes the Verilog of the board top level. Usage: dune exec bin/gen_verilog.exe
   [output-directory] Run from the repository root; the default output directory is
   board/_generated. *)

let () =
  let dir = if Array.length Sys.argv > 1 then Sys.argv.(1) else "board/_generated" in
  if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
  let rtl =
    Hardcaml.Rtl.create Verilog [ Mgen.Top.create () ]
    |> Hardcaml.Rtl.full_hierarchy
    |> Rope.to_string
  in
  Out_channel.with_open_text (Filename.concat dir "top.v") (fun oc ->
    Out_channel.output_string oc rtl)
;;
