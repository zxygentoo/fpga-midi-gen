(* The parts every RTL gate driver mounts — see gate_common.mli for the contract. It knows
   no era: the model reader, the source and the walk all arrive as arguments. *)
open Core

let int8_param of_int8_checkpoint =
  let%map_open.Command path =
    flag "-int8" (required string) ~doc:"PATH the contract file of the model"
  in
  fun () -> of_int8_checkpoint path
;;

let seed_param =
  let open Command.Param in
  flag "-seed" (required int) ~doc:"N the seed of the walk"
;;

let steps_param =
  let open Command.Param in
  flag "-steps" (required int) ~doc:"N the steps of the walk"
;;

let verilog_command ~summary ~model ~source =
  Command.basic
    ~summary
    (let%map_open.Command model
     and dir = anon ("output-directory" %: string) in
     fun () ->
       Core_unix.mkdir_p dir ~perm:0o755;
       let model = model () in
       let rtl =
         Hardcaml.Rtl.create Verilog [ Mgen_nexys4.Top.create ~source:(source model) () ]
         |> Hardcaml.Rtl.full_hierarchy
         |> Rope.to_string
       in
       Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl)
;;

let walk_command ~summary ~model ~walk =
  Command.basic
    ~summary
    (let%map_open.Command model
     and seed = seed_param
     and steps = steps_param in
     fun () ->
       let play = walk ~model:(model ()) ~seed in
       (* the for loop and not [List.init]: a simulation must be stepped in the true
          order, and [List.init] applies its function in the reverse index order *)
       for step = 0 to steps - 1 do
         let frame = play () in
         printf
           "step %d %08x %s\n"
           step
           frame
           (String.concat
              ~sep:" "
              (List.map (Mgen_corpus.Vocab.classes_of_frame frame) ~f:Int.to_string))
       done)
;;
