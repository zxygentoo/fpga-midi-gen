(** What every RTL gate driver mounts: the three flags and the walk.

    A GATE DRIVER STATES WHAT THE CIRCUIT DID AND NEVER WHAT IT SHOULD HAVE DONE. The
    oracle is the era's integer twin above the seam, in [jax/<era>/quantized/infer.py],
    and the gate that compares them is [jax/tests/test_rtl_<era>.py]. Nothing here states
    an expectation, thus a gate cannot pass by the driver agreeing with itself, and this
    module holds only the part that is neither side of that comparison.

    IT KNOWS NO ERA AND NO BOARD. The model reader and the walk arrive as arguments, thus
    this library names no era library; and a board top level is not a gate driver's work,
    thus it names no board library either. [bin/gen_verilog] elaborates every era. *)

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
