(** What every RTL gate driver mounts: the three flags, the netlist command and the walk.

    A GATE DRIVER STATES WHAT THE CIRCUIT DID AND NEVER WHAT IT SHOULD HAVE DONE. The
    oracle is the era's integer twin above the seam, in [jax/<era>/quantized.py], and the
    gate that compares them is [jax/tests/test_rtl_<era>.py]. Nothing here states an
    expectation, thus a gate cannot pass by the driver agreeing with itself, and this
    module holds only the part that is neither side of that comparison.

    IT KNOWS NO ERA. The model reader, the source and the walk all arrive as arguments,
    thus this library depends on no era library — it could not, because the board library
    it elaborates through already depends on all of them. *)

open Core

(** [int8_param of_int8_checkpoint] is the [-int8] flag: the path of a contract file, read
    with the era's own reader.

    THE MODEL IS A CONTRACT FILE AND NEVER A DRAW OF THIS SIDE. The twin draws the tiny
    model, quantizes it and writes the file, and both sides then read one model. Every
    shape number travels in that file, thus no flag of a driver states a shape. *)
val int8_param : (string -> 'model) -> (unit -> 'model) Command.Param.t

(** the [-seed] flag: the seed of the walk *)
val seed_param : int Command.Param.t

(** the [-steps] flag: the steps of the walk *)
val steps_param : int Command.Param.t

(** [verilog_command ~summary ~model ~source] writes the Verilog of an era's board top
    level into a directory named on the command line.

    Era six holds the board and [Top] takes its source as an argument, thus a driver
    elaborates ITS OWN era's top level and [test_parity.py] holds the md5 of it against
    the pin. Neither the elaboration nor the quantizer can move without that gate saying
    so. *)
val verilog_command
  :  summary:string
  -> model:(unit -> 'model) Command.Param.t
  -> source:
       ('model
        -> seed:Hardcaml.Signal.t
        -> Hardcaml.Signal.t Mgen_core.Source_intf.I.t
        -> Hardcaml.Signal.t Mgen_core.Source_intf.O.t)
  -> Command.t

(** [walk_command ~summary ~model ~walk] prints one walk, step by step: the FRAME the
    socket face answered and the CLASSES that frame states, through [Vocab]'s own decode.
    The vocabulary is the corpus library's rule and it stays on this side, thus the twin
    holds no format of its own and states classes alone.

    [walk ~model ~seed] rewinds the era's own bench and gives back the function that plays
    ONE step. The bench itself does not cross this interface: the two frozen eras carry
    different ones — era five's also reports the stream writes — and the walk is what they
    share. *)
val walk_command
  :  summary:string
  -> model:(unit -> 'model) Command.Param.t
  -> walk:(model:'model -> seed:int -> unit -> int)
  -> Command.t
