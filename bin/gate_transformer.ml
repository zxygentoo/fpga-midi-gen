(* gate_transformer: the driver of the RTL gates of era four.

   THE ORACLE IS THE JAX TWIN AND IT IS NOT HERE. This tool runs the circuit in Cyclesim
   and prints WHAT THE CIRCUIT DID, line by line; jax/tests/test_rtl_transformer.py states
   what it must have done, from its own engine over the same model, and compares. Nothing
   in this file states an expectation, thus a gate cannot pass by the driver agreeing with
   itself.

   The flags, the netlist command and the walk print are [Gate_common], which the three
   drivers share; this file states era four's model reader, its source and its bench, and
   nothing else. Every shape number of era four travels in the contract file — the width
   and the layers in the tensors, the heads, the context and the ALiBi span beside them —
   thus no flag here states a shape.

   [verilog] is the netlist gate of the era, and
   `test_g1_the_transformer_quantizer_states_its_netlist` holds its md5 against the pin.
   Neither the elaboration nor the quantizer can move without that gate saying so. *)

open Core
module Model = Mgen_transformer.Model
module Source = Mgen_transformer.Source

let model_param = Gate_common.int8_param Model.of_int8_checkpoint

let command =
  Command.group
    ~summary:
      "drive the circuit of era four and state what it did; \
       jax/tests/test_rtl_transformer.py states what it must have done"
    [ ( "walk"
      , Gate_common.walk_command
          ~summary:"one walk: the frame of each step, and the classes it states"
          ~model:model_param
          ~walk:(fun ~model ~seed ->
            let h = Source.For_test.Bench.harness ~model ~seed () in
            h.rewind ();
            h.play) )
    ; ( "verilog"
      , Gate_common.verilog_command
          ~summary:"write the Verilog of era four's board top level into a directory"
          ~model:model_param
          ~source:(fun model -> Source.create ~model) )
    ]
;;

let () = Command_unix.run command
