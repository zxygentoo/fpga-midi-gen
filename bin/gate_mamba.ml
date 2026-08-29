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

let model_param = Gate_common.int8_param Model.of_int8_checkpoint

(* Every write of the whole residual stream, in the order the machine made them: the
   embed, then the join of each layer. ERA FIVE'S FOUR FAULTS WERE ALL FAULTS OF THE
   COMPOSITION LAYER — a weight address whose stride was not the tensor's, a channel block
   read at the gate's offset, an operand taken on the address side of a two-cycle read,
   and a ring run off its end — and none of them moved a frame.

   IT STAYS IN THIS FILE: it reads [streams], which era four's bench has not, thus it is
   not a thing the drivers share. *)
let stream_command =
  Command.basic
    ~summary:
      "one walk: every write of the whole stream, in the order the machine made them"
    (let%map_open.Command model = model_param
     and seed = Gate_common.seed_param
     and steps = Gate_common.steps_param in
     fun () ->
       let h = Source.For_test.Bench.harness ~model:(model ()) ~seed () in
       h.rewind ();
       for step = 0 to steps - 1 do
         let (_ : int) = h.play () in
         List.iteri (h.streams ()) ~f:(fun at row ->
           printf
             "write %d %d %s\n"
             step
             at
             (String.concat ~sep:" " (Array.to_list (Array.map row ~f:Int.to_string))))
       done)
;;

let command =
  Command.group
    ~summary:
      "drive the circuit of era five and state what it did; jax/tests/test_rtl_mamba.py \
       states what it must have done"
    [ ( "walk"
      , Gate_common.walk_command
          ~summary:"one walk: the frame of each step, and the classes it states"
          ~model:model_param
          ~walk:(fun ~model ~seed ->
            let h = Source.For_test.Bench.harness ~model ~seed () in
            h.rewind ();
            h.play) )
    ; "stream", stream_command
    ; ( "verilog"
      , Gate_common.verilog_command
          ~summary:"write the Verilog of era five's board top level into a directory"
          ~model:model_param
          ~source:(fun model -> Source.create ~model) )
    ]
;;

let () = Command_unix.run command
