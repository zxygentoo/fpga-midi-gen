(* checkpoint_tool: operations on one checkpoint. The eval subcommand is the referee of
   the election run standalone: the fixed valid windows of Evaluation. With -out it writes
   the gate file of the JAX seam — the batch, the loss and the config that made them —
   thus the parity gates compare against the same rows and the same model the election
   reads. The config flags must equal the flags of the training run; the checkpoint holds
   only tensors. *)

open Core
module Evaluation = Mgen_transformer.Evaluation
module Fixed = Mgen_transformer.Fixed
module Jsb = Mgen_corpus.Jsb
module Token = Mgen_core.Token
module Transformer = Mgen_transformer.Transformer

(* The gate carries the heads and the ALiBi span, because no other file holds them: a
   tensor shape gives the width and the layer count, and the two tables give the rest, but
   the heads only split the width at run time and ALiBi holds no position table. The piece
   positions ride along as "progress". Without them the JAX side must keep its own copy,
   and a run at another span then passes the compare against a different model. *)
let gate_entries rows ~(config : Transformer.Config.t) ~loss =
  let rows = Evaluation.batch_of_rows rows in
  let codes = rows.codes in
  let batch = Array.length codes in
  let need = Array.length codes.(0) in
  let context = need - 1 in
  let packed = Array.map rows.masks ~f:(Array.map ~f:Evaluation.mask_words) in
  let i32 shape read = Nx.init Nx.int32 shape (fun i -> Int32.of_int_exn (read i)) in
  let scalar value = Nx_io.P (i32 [| 1 |] (fun (_ : int array) -> value)) in
  [ "codes", Nx_io.P (i32 [| batch; need |] (fun i -> codes.(i.(0)).(i.(1))))
  ; "phases", Nx_io.P (i32 [| batch; context |] (fun i -> rows.phases.(i.(0)).(i.(1))))
  ; ( "progress"
    , Nx_io.P (i32 [| batch; context |] (fun i -> rows.progress.(i.(0)).(i.(1)))) )
  ; ( "masks"
    , Nx_io.P
        (i32 [| batch; context; Evaluation.words_per_mask |] (fun i ->
           packed.(i.(0)).(i.(1)).(i.(2)))) )
  ; "loss", Nx_io.P (Nx.init Nx.float32 [| 1 |] (fun (_ : int array) -> loss))
  ; "heads", scalar config.heads
  ; "span", scalar config.slope_span
  ]
;;

let eval ~checkpoint ~corpus ~heads ~context ~slope_span ~rows ~batch ~out =
  let config = Transformer.Config.of_checkpoint checkpoint ~heads ~context ~slope_span in
  let params = Transformer.Params.load config ~path:checkpoint in
  let data = Jsb.load ~path:corpus in
  let eval_rows = Evaluation.rows data.valid ~context ~limit:rows in
  let loss = Evaluation.loss config params eval_rows ~batch in
  printf "%d valid rows  loss %.4f\n%!" (List.length eval_rows) loss;
  Option.iter out ~f:(fun path ->
    Core_unix.mkdir_p (Filename.dirname path);
    Nx_io.save_safetensors ~overwrite:true path (gate_entries eval_rows ~config ~loss);
    printf "wrote %s\n%!" path)
;;

let eval_command =
  Command.basic
    ~summary:"the election's own eval of one checkpoint; -out writes the gate file"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
     and corpus =
       flag
         "-corpus"
         (optional_with_default Jsb.default_path string)
         ~doc:"PATH the voice-separated corpus file"
     and slope_span =
       flag
         "-alibi-span"
         (optional_with_default Transformer.Config.(baseline.slope_span) int)
         ~doc:"N the ALiBi exponent span; it must equal the span of the training run"
     and heads =
       flag
         "-heads"
         (optional_with_default Transformer.Config.(baseline.heads) int)
         ~doc:"N the heads"
     and context =
       flag
         "-context"
         (optional_with_default Transformer.Config.(baseline.context) int)
         ~doc:"N the window, in tokens"
     and rows =
       flag "-rows" (optional_with_default 16 int) ~doc:"N the widest gate batch, in rows"
     and batch = flag "-batch" (optional_with_default 16 int) ~doc:"N the compute batch"
     and out = flag "-out" (optional string) ~doc:"PATH write the gate file here" in
     fun () -> eval ~checkpoint ~corpus ~heads ~context ~slope_span ~rows ~batch ~out)
;;

(* The calibration of the integer twin: run the quantized walk, print the peak of each
   signal class — the circuit widths read them — and measure the drift against the float
   model: the top-1 agreement and the cosine of the logits at every draw, on the twin's
   own walk. *)
let ranges ~checkpoint ~steps ~seed =
  let config =
    Transformer.Config.of_checkpoint
      checkpoint
      ~heads:Transformer.Config.baseline.heads
      ~context:Transformer.Config.baseline.context
      ~slope_span:Transformer.Config.baseline.slope_span
  in
  let quantized = Fixed.Model.of_checkpoint config checkpoint in
  let params = Transformer.Params.load config ~path:checkpoint in
  let engine = ref (Fixed.Engine.init quantized ~seed) in
  (* the histories of the float pass, newest first, as the float sampler keeps them *)
  let codes = ref [ Token.to_code Token.Start ] in
  let phases = ref [ 0 ] in
  let progress = ref [ 0 ] in
  let window history = List.take history 256 |> List.rev |> Array.of_list in
  let float_logits () =
    let codes = window !codes in
    Transformer.logits
      config
      params
      ~codes:[| codes |]
      ~phases:[| window !phases |]
      ~progress:[| window !progress |]
      ~dropout:Transformer.Dropout.none
    |> Nx.get [ 0; Array.length codes - 1 ]
    |> Nx.to_array
  in
  let argmax values ~value =
    let best = ref 0 in
    Array.iteri values ~f:(fun index v ->
      if Float.(value v > value values.(!best)) then best := index);
    !best
  in
  let cosine q f =
    let dot = ref 0.0
    and qq = ref 0.0
    and ff = ref 0.0 in
    Array.iteri q ~f:(fun i qi ->
      let qi = Float.of_int qi in
      dot := !dot +. (qi *. f.(i));
      qq := !qq +. (qi *. qi);
      ff := !ff +. (f.(i) *. f.(i)));
    !dot /. Float.sqrt (!qq *. !ff)
  in
  let agree = ref 0 in
  let cosine_sum = ref 0.0 in
  let draws = ref 0 in
  let events = ref 0 in
  let step_index = ref 0 in
  while !step_index < steps do
    let quantized = Fixed.Engine.logits !engine in
    let floated = float_logits () in
    if argmax quantized ~value:Float.of_int = argmax floated ~value:Fn.id then incr agree;
    cosine_sum := !cosine_sum +. cosine quantized floated;
    incr draws;
    let engine', code = Fixed.Engine.next_code !engine in
    engine := engine';
    let phase = !step_index mod Transformer.phase_buckets in
    let bucket =
      !step_index / Transformer.progress_stride mod Transformer.progress_buckets
    in
    engine := Fixed.Engine.forward !engine ~code ~phase ~bucket;
    codes := code :: !codes;
    phases := phase :: !phases;
    progress := bucket :: !progress;
    match Token.of_code code with
    | Start -> assert false
    | On (_ : int) | Off (_ : int) -> incr events
    | End -> incr step_index
  done;
  printf "%d steps  %d events  %d draws\n" steps !events !draws;
  printf
    "against the float model: top-1 %.1f%%  cosine %.4f\n"
    (100.0 *. Float.of_int !agree /. Float.of_int !draws)
    (!cosine_sum /. Float.of_int !draws);
  printf "the peaks, before any clamp:\n";
  List.iter (Fixed.Engine.peaks !engine) ~f:(fun (name, peak) ->
    printf "  %-8s %d\n" name peak)
;;

let ranges_command =
  Command.basic
    ~summary:"calibrate the integer twin: the signal peaks and the float drift"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
     and steps = flag "-steps" (optional_with_default 96 int) ~doc:"N the steps to draw"
     and seed = flag "-seed" (optional_with_default 42 int) ~doc:"N the seed" in
     fun () -> ranges ~checkpoint ~steps ~seed)
;;

(* The twin's socket stream: what the board must send, event for event. The comparison
   script reads these lines against the amidi capture of the S-1 thru. *)
let twin ~checkpoint ~steps ~seed =
  let config =
    Transformer.Config.of_checkpoint
      checkpoint
      ~heads:Transformer.Config.baseline.heads
      ~context:Transformer.Config.baseline.context
      ~slope_span:Transformer.Config.baseline.slope_span
  in
  let model = Fixed.Model.of_checkpoint config checkpoint in
  let engine = ref (Fixed.Engine.init model ~seed) in
  for step = 1 to steps do
    let engine', events = Fixed.Engine.next_step !engine in
    engine := engine';
    printf
      "step %d:%s\n"
      step
      (String.concat
         (List.map events ~f:(fun { Fixed.Engine.voice; pitch; on } ->
            sprintf " %s:%d@%d" (if on then "on" else "off") pitch voice)))
  done
;;

let twin_command =
  Command.basic
    ~summary:"the integer twin's event stream: the reference of the board capture"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
     and steps = flag "-steps" (optional_with_default 64 int) ~doc:"N the steps to draw"
     and seed = flag "-seed" (optional_with_default 42 int) ~doc:"N the seed" in
     fun () -> twin ~checkpoint ~steps ~seed)
;;

let command =
  Command.group
    ~summary:"operations on one checkpoint"
    [ "eval", eval_command; "ranges", ranges_command; "twin", twin_command ]
;;

let () = Command_unix.run command
