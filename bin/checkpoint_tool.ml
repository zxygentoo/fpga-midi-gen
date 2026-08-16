(* checkpoint_tool: operations on one checkpoint. The eval subcommand is the referee of
   the election run standalone: the fixed valid windows of Evaluation. With -out it writes
   the gate file of the JAX seam — the batch, the loss and the config that made them —
   thus the parity gates compare against the same rows and the same model the election
   reads. The config flags must equal the flags of the training run; the checkpoint holds
   only tensors. *)

open Core
module Evaluation = Mgen_transformer.Evaluation
module Quantized = Mgen_transformer.Quantized
module Jsb = Mgen_corpus.Jsb
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
  let eval_rows = Evaluation.rows (Jsb.pack data.valid) ~context ~limit:rows in
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

(* The drift of the reference against the float model: the top-1 agreement, the cosine of
   the logits and the same-draw share at every draw, on the quantized walk. *)
let drift ~checkpoint ~steps ~seed =
  let config =
    Transformer.Config.of_checkpoint
      checkpoint
      ~heads:Transformer.Config.baseline.heads
      ~context:Transformer.Config.baseline.context
      ~slope_span:Transformer.Config.baseline.slope_span
  in
  let params = Transformer.Params.load config ~path:checkpoint in
  let { Quantized.Drift.draws; events; same_peak; same_draw; mean_cosine } =
    Quantized.Drift.walk config params ~steps ~seed
  in
  let share count = 100.0 *. Float.of_int count /. Float.of_int draws in
  printf "%d steps  %d events  %d draws\n" steps events draws;
  printf
    "against the float model: top-1 %.1f%%  cosine %.4f  same draw %.1f%%\n"
    (share same_peak)
    mean_cosine
    (share same_draw)
;;

let drift_command =
  Command.basic
    ~summary:
      "the drift of the reference against the float model: top-1, cosine and the \
       same-draw share"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
     and steps = flag "-steps" (optional_with_default 96 int) ~doc:"N the steps to draw"
     and seed = flag "-seed" (optional_with_default 42 int) ~doc:"N the seed" in
     fun () -> drift ~checkpoint ~steps ~seed)
;;

(* The reference socket stream: what the board must send, event for event. The comparison
   script reads these lines against the amidi capture of the S-1 thru. *)
let stream ~checkpoint ~steps ~seed =
  let config =
    Transformer.Config.of_checkpoint
      checkpoint
      ~heads:Transformer.Config.baseline.heads
      ~context:Transformer.Config.baseline.context
      ~slope_span:Transformer.Config.baseline.slope_span
  in
  let model = Quantized.Model.of_checkpoint config checkpoint in
  let engine = ref (Quantized.Engine.init model ~seed) in
  for step = 1 to steps do
    let engine', events = Quantized.Engine.next_step !engine in
    engine := engine';
    printf
      "step %d:%s\n"
      step
      (String.concat
         (List.map events ~f:(fun { Quantized.Engine.voice; pitch; on } ->
            sprintf " %s:%d@%d" (if on then "on" else "off") pitch voice)))
  done
;;

let stream_command =
  Command.basic
    ~summary:"the reference event stream: what the board must send, event for event"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
     and steps = flag "-steps" (optional_with_default 64 int) ~doc:"N the steps to draw"
     and seed = flag "-seed" (optional_with_default 42 int) ~doc:"N the seed" in
     fun () -> stream ~checkpoint ~steps ~seed)
;;

let command =
  Command.group
    ~summary:"operations on one checkpoint"
    [ "drift", drift_command; "eval", eval_command; "stream", stream_command ]
;;

let () = Command_unix.run command
