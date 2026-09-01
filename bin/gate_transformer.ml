(* gate_transformer: the driver of the RTL gates of era four.

   THE ORACLE IS THE JAX TWIN AND IT IS NOT HERE. This tool runs the circuit in Cyclesim
   and prints WHAT THE CIRCUIT DID, line by line; jax/tests/test_rtl_transformer.py states
   what it must have done, from its own engine over the same model, and compares. Nothing
   in this file states an expectation, thus a gate cannot pass by the driver agreeing with
   itself.

   The flags and the walk print are [Gate_common], which the three drivers share; this
   file states era four's model reader, its source and its bench, and nothing else. Every
   shape number of era four travels in the contract file — the width and the layers in the
   tensors, the heads, the context and the ALiBi span beside them — thus no flag here
   states a shape.

   THE NETLIST GATE OF THE ERA IS NOT HERE. [bin/gen_verilog transformer] elaborates the
   board top level over this era's source and
   `test_g1_the_transformer_quantizer_states_its_netlist` holds its md5 against the pin;
   neither the elaboration nor the quantizer can move without that gate saying so. A gate
   driver states what a circuit DID, thus it writes no netlist for a build. *)

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
    ]
;;

let () = Command_unix.run command
