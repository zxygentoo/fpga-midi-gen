(* gate_transformer: the driver of the RTL gates of era four.

   THE ORACLE IS THE JAX TWIN AND IT IS NOT HERE. This tool runs the circuit in Cyclesim
   and prints WHAT THE CIRCUIT DID, line by line; jax/tests/test_rtl_transformer.py states
   what it must have done, from its own engine over the same model, and compares. Nothing
   in this file states an expectation, thus a gate cannot pass by the driver agreeing with
   itself.

   The model is a CONTRACT FILE and never a draw of this side:
   jax/transformer/quantized.py draws the tiny model, quantizes it and writes the file,
   and both sides then read one model. Every shape number of era four travels in that file
   — the width and the layers in the tensors, the heads, the context and the ALiBi span
   beside them — thus no flag here states a shape.

   [verilog] is the netlist gate of the era. Era six holds the board and [Top] takes its
   source as an argument, thus this program elaborates ERA FOUR'S top level, and
   `test_g1_the_transformer_quantizer_states_its_netlist` holds its md5 against the pin.
   Neither the elaboration nor the quantizer can move without that gate saying so. *)

open Core
module Model = Mgen_transformer.Model
module Source = Mgen_transformer.Source

let model_param =
  let%map_open.Command path =
    flag "-int8" (required string) ~doc:"PATH the contract file of the model"
  in
  fun () -> Model.of_int8_checkpoint path
;;

let verilog_command =
  Command.basic
    ~summary:"write the Verilog of era four's board top level into a directory"
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

let seed_param =
  let open Command.Param in
  flag "-seed" (required int) ~doc:"N the seed of the walk"
;;

(* The walk, step by step. It prints the FRAME the socket face answered and the CLASSES
   that frame states, through [Vocab]'s own decode: the vocabulary is the corpus library's
   rule and it stays on this side, thus the twin holds no format of its own and states
   classes alone. *)
let walk_command =
  Command.basic
    ~summary:"one walk: the frame of each step, and the classes it states"
    (let%map_open.Command model = model_param
     and seed = seed_param
     and steps = flag "-steps" (required int) ~doc:"N the steps of the walk" in
     fun () ->
       let h = Source.For_test.Bench.harness ~model:(model ()) ~seed () in
       h.rewind ();
       for step = 0 to steps - 1 do
         let frame = h.play () in
         printf
           "step %d %08x %s\n"
           step
           frame
           (String.concat
              ~sep:" "
              (List.map (Mgen_corpus.Vocab.classes_of_frame frame) ~f:Int.to_string))
       done)
;;

let command =
  Command.group
    ~summary:
      "drive the circuit of era four and state what it did; \
       jax/tests/test_rtl_transformer.py states what it must have done"
    [ "walk", walk_command; "verilog", verilog_command ]
;;

let () = Command_unix.run command
