(* checkpoint_tool: operations on one checkpoint — the drift of the quantized reference
   against the float model and the reference event stream. The config flags must equal the
   flags of the training run; the checkpoint holds only tensors. *)

open Core
module Quantized = Mgen_transformer.Quantized
module Transformer = Mgen_transformer.Transformer

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
    [ "drift", drift_command; "stream", stream_command ]
;;

let () = Command_unix.run command
