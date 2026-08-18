(* checkpoint_tool: what the integer twin of a checkpoint does — the drift of the twin
   against the float model, and the event stream the board must send.

   The configuration flags must equal the flags of the training run; the checkpoint holds
   only tensors. *)

open Core
module Frame = Mgen_core.Frame
module Jsb = Mgen_corpus.Jsb
module Quantized = Mgen_transformer.Quantized
module Transformer = Mgen_transformer.Transformer

let config_of checkpoint ~heads ~context ~slope_span =
  Transformer.Config.of_checkpoint checkpoint ~heads ~context ~slope_span
;;

(* the flags that must equal the training run, in one place: three commands would
   otherwise state them three times *)
let config_flags =
  let%map_open.Command checkpoint =
    flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
  and heads =
    flag
      "-heads"
      (optional_with_default Transformer.Config.(baseline.heads) int)
      ~doc:"N the heads; they must equal the heads of the training run"
  and context =
    flag
      "-context"
      (optional_with_default Transformer.Config.(baseline.context) int)
      ~doc:"N the window, in steps"
  and slope_span =
    flag
      "-alibi-span"
      (optional_with_default Transformer.Config.(baseline.slope_span) int)
      ~doc:"N the ALiBi exponent span; it must equal the span of the training run"
  in
  checkpoint, config_of checkpoint ~heads ~context ~slope_span
;;

(* The drift of the twin against the float model: the top-1 agreement, the cosine of the
   logits and the same-draw share, over every draw of the chain. *)
let drift ~checkpoint ~config ~steps ~seed =
  let params = Transformer.Params.load config ~path:checkpoint in
  let { Quantized.Drift.steps = (_ : int); draws; same_peak; same_draw; mean_cosine } =
    Quantized.Drift.walk config params ~steps ~seed
  in
  let share count = 100.0 *. Float.of_int count /. Float.of_int (max 1 draws) in
  printf "%d steps  %d draws of the chain\n" steps draws;
  printf
    "against the float model: top-1 %.1f%%  cosine %.4f  same draw %.1f%%\n"
    (share same_peak)
    mean_cosine
    (share same_draw)
;;

let drift_command =
  Command.basic
    ~summary:
      "the drift of the integer twin against the float model: top-1, cosine and the \
       same-draw share"
    (let%map_open.Command checkpoint, config = config_flags
     and steps = flag "-steps" (optional_with_default 64 int) ~doc:"N the steps to draw"
     and seed = flag "-seed" (optional_with_default 42 int) ~doc:"N the seed" in
     fun () -> drift ~checkpoint ~config ~steps ~seed)
;;

(* The reference stream: what the board must send, event for event. The line format is the
   one of play_transformer and of the JAX twin, thus the three compare with `diff` — and
   the difference between this stream and the float player's is the quantization. *)
let stream ~checkpoint ~config ~steps ~seed ~temperature ~min_p =
  let model = Quantized.Model.of_checkpoint ~temperature ~min_p config checkpoint in
  let engine = ref (Quantized.Engine.init model ~seed) in
  let frames =
    Array.init steps ~f:(fun (_ : int) ->
      let next, (step : Quantized.Engine.step) = Quantized.Engine.next_step !engine in
      engine := next;
      step.frame)
  in
  List.iteri (Frame.events_of_frames frames) ~f:(fun index events ->
    let text =
      List.map events ~f:(function
        | Frame.Event.On pitch -> sprintf "on:%d" pitch
        | Frame.Event.Off pitch -> sprintf "off:%d" pitch)
    in
    printf
      "step %3d  %s\n"
      index
      (if List.is_empty text then "-" else String.concat ~sep:" " text))
;;

(* Gate A of the JAX seam: the loss of the float model over the canonical valid windows.

   A referee reads the canonical stream alone — every piece at shift zero, in the order
   given — thus this number is deterministic and two referees that read one checkpoint
   must agree. [Jsb.windows] cuts the windows and [jax/data.py] states the same cut, thus
   the trainer and the reference measure one thing. [jax/tests/test_parity.py] demands the
   same number from the JAX forward, because a training run on the other side of that seam
   only means something if the two forwards agree everywhere the loss can see.

   The shape travels in the output and this side states no number of its own: held apart,
   the heads and the span would go on matching a default while one side moved, and the
   gate would pass two different models. *)
let loss ~checkpoint ~config ~corpus =
  let params = Transformer.Params.load config ~path:checkpoint in
  let data = Jsb.load ~path:corpus in
  (* the first stream of a split is the canonical one, whatever the lane *)
  let random_state = Random.State.make [| 0 |] in
  let stream = List.hd_exn (Jsb.streams data.valid ~count:1 ~random_state) in
  let { Transformer.Config.context; heads; slope_span; _ } = config in
  let windows = Jsb.windows stream ~context in
  printf
    "windows %d  context %d  heads %d  span %d  loss %.6f\n"
    (List.length windows)
    context
    heads
    slope_span
    (Transformer.loss config params ~windows)
;;

let loss_command =
  Command.basic
    ~summary:
      "the loss over the canonical valid windows: Gate A of the JAX seam, which the \
       parity test demands of the JAX forward"
    (let%map_open.Command checkpoint, config = config_flags
     and corpus =
       flag
         "-corpus"
         (optional_with_default Jsb.default_path string)
         ~doc:"PATH the corpus that states the windows"
     in
     fun () -> loss ~checkpoint ~config ~corpus)
;;

let stream_command =
  Command.basic
    ~summary:"the reference event stream: what the board must send, event for event"
    (let%map_open.Command checkpoint, config = config_flags
     and steps = flag "-steps" (optional_with_default 64 int) ~doc:"N the steps to draw"
     and seed =
       flag
         "-seed"
         (optional_with_default 1 int)
         ~doc:"N the seed, under the rule of the SEED cell: 1 up to 0xFFFFFFFF"
     and temperature =
       flag
         "-temperature"
         (optional_with_default Transformer.elected_temperature float)
         ~doc:"F"
     and min_p =
       flag "-min-p" (optional_with_default Transformer.elected_min_p float) ~doc:"F"
     in
     fun () -> stream ~checkpoint ~config ~steps ~seed ~temperature ~min_p)
;;

let command =
  Command.group
    ~summary:"the integer twin of one checkpoint"
    [ "drift", drift_command; "loss", loss_command; "stream", stream_command ]
;;

let () = Command_unix.run command
