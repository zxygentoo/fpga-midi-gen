(* gate_diffusion: the driver of the RTL gates of era six.

   THE ORACLE IS THE JAX TWIN AND IT IS NOT HERE. This tool runs the circuit in Cyclesim
   and prints WHAT THE CIRCUIT DID, line by line; [jax/tests/test_rtl_diffusion.py] states
   what it must have done, from its own engine at the same seed and its own [layer_writes]
   over the same model, and compares. Nothing in this file states an expectation, thus a
   gate cannot pass by the driver agreeing with itself.

   The model is a CONTRACT FILE and never a draw of this side:
   [jax/diffusion/quantized.py] draws the tiny model, quantizes it and writes the file,
   and both sides then read one model. The geometry cannot travel in a file — the steps of
   a sheet, the lanes of a group and the passes of a walk are the elaboration's — thus it
   travels in the flags, and the Python side states the same numbers it passed.

   Two gates, two subcommands:

   - [walk] is INSTRUMENT 2: the walk beside the engine, PHASE FOR PHASE. It prints every
     write of the cell port in the order the walk made them — the opening's classes, each
     pass's mask bits, each pass's redraws — and then the frames the transfer face
     answers, beside the frames the sheet it drew states. The finished sheet alone would
     pass a walk whose masks are one pass out of phase, or one that spends a uniform on a
     standing cell: both draw a sheet, and both draw the WRONG one with no local symptom.

   - [stream] is INSTRUMENT 3: every column the engine writes, against the twin's own
     [layer_writes]. Era five's four faults — a weight address whose stride was not the
     tensor's, a channel block read at the gate's offset, an operand taken on the address
     side of a two-cycle read, and a ring run off its end — were all faults of the
     composition layer, and none of them moved a frame. *)

open Core
module Model = Mgen_diffusion.Model
module Sheet = Mgen_diffusion.Sheet
module Elaboration = Mgen_diffusion.Elaboration
module Forward = Mgen_diffusion.Forward
module Generator = Mgen_diffusion.Generator
module Source = Mgen_diffusion.Source
module Frame = Mgen_core.Frame

(* THE GEOMETRY IS THE ONLY THING THE FLAGS CARRY. The shape of the model — L, H and every
   width — comes out of the contract file, as it does for a build; a gate that stated a
   width of its own could go on matching a default while one side moved. *)
let elaboration_param =
  let%map_open.Command path =
    flag "-int8" (required string) ~doc:"PATH the contract file of the model"
  and steps = flag "-steps" (required int) ~doc:"T the steps of one sheet"
  and lanes = flag "-lanes" (required int) ~doc:"G the output channels of one group"
  and walk = flag "-walk" (required int) ~doc:"N the passes of the walk"
  and rows =
    flag
      "-rows"
      (optional int)
      ~doc:"P the pitch rows of a column (default: the classes of the vocabulary)"
  in
  fun () -> Elaboration.create ?rows (Model.of_int8_checkpoint path) ~steps ~lanes ~walk
;;

(* the one flag era six shares with the frozen drivers; its geometry flags below are the
   sheet's own and no other era has them *)
let seed_param = Gate_common.seed_param

(* one column of activations, as a line states it *)
let row values = String.concat ~sep:" " (List.map (Array.to_list values) ~f:Int.to_string)

(* ==================================================================== *)
(* The walk *)
(* ==================================================================== *)

(* THE SHEET THE CIRCUIT DREW, READ OUT OF ITS OWN CLASS WRITES: the last class a cell
   took is the class it holds. The frames are then [Model.frames_of_sheet] of that sheet,
   and the two comparisons of the Python side compose — the writes ARE the twin's, thus
   the sheet is the twin's, thus a frame face that agrees with this sheet agrees with the
   twin. The driver never reads a model to state a frame. *)
let sheet_of_writes (writes : Generator.For_test.Bench.write list) ~steps =
  let sheet = Array.make_matrix ~dimx:steps ~dimy:Frame.voices 0 in
  List.iter writes ~f:(fun { Generator.For_test.Bench.mask; step; seat; value } ->
    if not mask then sheet.(step).(seat) <- value);
  sheet
;;

(* the steps the transfer face is played PAST the end of the sheet. The face is cyclic —
   the counter wraps at T — thus these read the sheet's own first frames again, and the
   gate holds the wrap the scheduler's copy rests on. *)
let wrapped_steps = 2

let run_walk e ~seed =
  let steps = e.Elaboration.steps in
  let h = Generator.For_test.Bench.harness ~e ~seed () in
  h.start ();
  let writes = h.writes () in
  List.iter writes ~f:(fun { Generator.For_test.Bench.mask; step; seat; value } ->
    printf "write %s %d %d %d\n" (if mask then "MASK" else "CLASS") step seat value);
  let frames = Model.frames_of_sheet (sheet_of_writes writes ~steps) in
  List.iter
    (List.range 0 (steps + wrapped_steps))
    ~f:(fun step -> printf "want_frame %d %d\n" step frames.(step % steps));
  List.iter
    (List.range 0 (steps + wrapped_steps))
    ~f:(fun step ->
      let frame = h.play () in
      printf "frame %d %d\n" step frame)
;;

let walk_command =
  Command.basic
    ~summary:"the walk beside the engine: every write of the cell port, then the frames"
    (let%map_open.Command elaborate = elaboration_param
     and seed = seed_param in
     fun () -> run_walk (elaborate ()) ~seed)
;;

(* ==================================================================== *)
(* The stream *)
(* ==================================================================== *)

let run_stream e ~seed =
  let steps = e.Elaboration.steps in
  let rows = e.rows in
  (* THE STEM'S INPUT AT ONE SEED: [Sheet.For_test.stem_input]'s rule, which fits P by
     construction. The sheet and the mask are printed, thus the Python side builds the
     same input from the same two facts and never redraws them. *)
  let sheet, hidden = Sheet.For_test.stem_input ~steps ~rows ~walk:e.walk ~seed in
  Array.iteri sheet ~f:(fun step seats ->
    Array.iteri seats ~f:(fun seat cell ->
      printf "sheet %d %d %d\n" step seat cell;
      printf "hidden %d %d %d\n" step seat (Bool.to_int hidden.(step).(seat))));
  let stem = Sheet.For_test.plane_activations sheet hidden ~steps ~rows in
  let module Bench =
    Forward.For_test.Bench (struct
      let e = e
    end)
  in
  let pass =
    Bench.run
      ~planes:(fun ~step ~plane -> Sheet.For_test.plane_column stem ~step ~plane ~rows)
      ()
  in
  (* THE WRITES OF A TURN COME OUT INTERLEAVED, thus a turn is what the cursor takes.
     Inside a pair the blocks run A0, A1, A2 B0, A3 B1, ...: A's columns go to Y and B's
     to X in one stream, and each write says which store took it. The driver takes the
     stream by TURN and then reads each layer's tensor whole out of the store that holds
     it. *)
  let depth = Elaboration.store_depth e in
  let store () = Array.init depth ~f:(fun (_ : int) -> Array.create ~len:rows 0) in
  let x = store () in
  let y = store () in
  let cursor = ref pass.written in
  let misplaced = ref 0 in
  let print_layer at (layer : Elaboration.layer) destination =
    List.iter (List.range 0 steps) ~f:(fun step ->
      List.iter (List.range 0 layer.outputs) ~f:(fun channel ->
        let address = Elaboration.column_address e ~step ~channel in
        printf "write %d %d %d %s\n" at step channel (row destination.(address))))
  in
  let turn_columns (turn : Elaboration.turn) =
    let ats = turn.first :: Option.to_list turn.second in
    let count = List.sum (module Int) ats ~f:(fun at -> steps * e.layers.(at).outputs) in
    let mine = List.take !cursor count in
    cursor := List.drop !cursor count;
    List.iter mine ~f:(fun (w : Bench.write) ->
      (if w.to_y then y else x).(w.address) <- w.column);
    (* every destination column written exactly one time, and each store taking exactly
       the layer's own count: the destination and the address as one key, X at [address]
       and Y above them *)
    let placed =
      List.fold
        mine
        ~init:(Set.empty (module Int))
        ~f:(fun seen w -> Set.add seen (if w.to_y then depth + w.address else w.address))
    in
    if List.length mine <> count || Set.length placed <> count then Int.incr misplaced;
    List.iter ats ~f:(fun at ->
      let layer = e.layers.(at) in
      let to_y =
        match layer.role with
        | Pair_open -> true
        | Stem | Pair_close | Head -> false
      in
      if List.count mine ~f:(fun w -> Bool.equal w.to_y to_y) <> steps * layer.outputs
      then Int.incr misplaced;
      print_layer at layer (if to_y then y else x))
  in
  let head_columns () =
    List.iteri pass.offered ~f:(fun step columns ->
      Array.iteri columns ~f:(fun seat column ->
        printf "logits %d %d %s\n" step seat (row column)))
  in
  Array.iter e.turns ~f:(fun (turn : Elaboration.turn) ->
    match e.layers.(turn.first).role with
    | Head -> head_columns ()
    | Stem | Pair_open | Pair_close -> turn_columns turn);
  printf "misplaced %d\n" !misplaced
;;

let stream_command =
  Command.basic
    ~summary:"one forward pass: every column the stores took, and the head's logits"
    (let%map_open.Command elaborate = elaboration_param
     and seed = seed_param in
     fun () -> run_stream (elaborate ()) ~seed)
;;

let command =
  Command.group
    ~summary:
      "drive the circuit of era six and state what it did; \
       jax/tests/test_rtl_diffusion.py states what it must have done"
    [ "walk", walk_command; "stream", stream_command ]
;;

let () = Command_unix.run command
