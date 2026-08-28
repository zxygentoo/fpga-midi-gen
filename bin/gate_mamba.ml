(* gate_mamba: the driver of the RTL gates of era five.

   THE ORACLE IS THE JAX TWIN AND IT IS NOT HERE. This tool runs the circuit in Cyclesim
   and prints WHAT THE CIRCUIT DID, line by line; jax/tests/test_rtl_mamba.py states what
   it must have done, from its own engine over the same model, and compares. Nothing in
   this file states an expectation, thus a gate cannot pass by the driver agreeing with
   itself.

   The model is a CONTRACT FILE and never a draw of this side: jax/mamba/quantized.py
   draws the tiny model, quantizes it and writes the file, and both sides then read one
   model. Every width and the plan travel in that file — the image states them — thus no
   flag here states a shape.

   [verilog] is the netlist gate of the era. Era six holds the board and [Top] takes its
   source as an argument, thus this program elaborates ERA FIVE'S top level, and
   `test_g1_the_mamba_quantizer_states_its_netlist` holds its md5 against the pin. Neither
   the elaboration nor the quantizer can move without that gate saying so. *)

open Core
module Model = Mgen_mamba.Model
module Source = Mgen_mamba.Source

let model_param =
  let%map_open.Command path =
    flag "-int8" (required string) ~doc:"PATH the contract file of the model"
  in
  fun () -> Model.of_int8_checkpoint path
;;

let verilog_command =
  Command.basic
    ~summary:"write the Verilog of era five's board top level into a directory"
    (let%map_open.Command model = model_param
     and dir = anon ("output-directory" %: string) in
     fun () ->
       Core_unix.mkdir_p dir ~perm:0o755;
       let model = model () in
       let rtl =
         Hardcaml.Rtl.create
           Verilog
           [ Mgen_nexys4.Top.create ~source:(Source.create ~model) () ]
         |> Hardcaml.Rtl.full_hierarchy
         |> Rope.to_string
       in
       Out_channel.write_all (Filename.concat dir "top.v") ~data:rtl)
;;

let command =
  Command.group
    ~summary:
      "drive the circuit of era five and state what it did; jax/tests/test_rtl_mamba.py \
       states what it must have done"
    [ "verilog", verilog_command ]
;;

let () = Command_unix.run command
