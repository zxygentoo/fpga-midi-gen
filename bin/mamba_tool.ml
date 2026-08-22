(* mamba_tool: what one checkpoint of era five states — the loss over the canonical
   windows, the drift of the integer twin against the float model, and the event stream
   the board must send.

   It states no shape of its own. Era four's tool had to be told the heads, the context
   and the ALiBi span, because none of the three sized a tensor there; the recurrence has
   no window, and the head count and the state width both size tensors, thus every width
   comes out of the file and a wrong flag cannot pass a different model. *)

open Core
module Frame = Mgen_core.Frame
module Jsb = Mgen_corpus.Jsb
module Mamba = Mgen_mamba.Mamba
module Quantized = Mgen_mamba.Quantized

let config_flags =
  let%map_open.Command checkpoint =
    flag "-ckpt" (required string) ~doc:"PATH the checkpoint; it states every width"
  in
  checkpoint, Mamba.Config.of_checkpoint checkpoint
;;

(* The drift of the twin against the float model: the top-1 agreement, the cosine of the
   logits, the same-draw share, and the clamps.

   The clamps are here because the formats of this era were chosen with margin and not
   metered on a trained checkpoint: a share above zero is the finding that says which
   format is wrong, and an error in the state carries forward where era four's died with
   its window. Give it a LONG walk — the drift of a few steps proves less here than it did
   there. *)
let drift ~checkpoint ~config ~steps ~seed =
  let params = Mamba.Params.load config ~path:checkpoint in
  let { Quantized.Drift.steps = (_ : int)
      ; draws
      ; same_peak
      ; same_draw
      ; mean_cosine
      ; dt_clamped
      ; beta_clamped
      ; state_clamped
      }
    =
    Quantized.Drift.walk config params ~steps ~seed
  in
  let share count = 100.0 *. Float.of_int count /. Float.of_int (max 1 draws) in
  printf "%d steps  %d draws of the chain\n" steps draws;
  printf
    "against the float model: top-1 %.1f%%  cosine %.4f  same draw %.1f%%\n"
    (share same_peak)
    mean_cosine
    (share same_draw);
  printf
    "clamped: dt %.4f%%  beta %.4f%%  state %.4f%%\n"
    (100.0 *. dt_clamped)
    (100.0 *. beta_clamped)
    (100.0 *. state_clamped)
;;

let drift_command =
  Command.basic
    ~summary:
      "the drift of the integer twin against the float model, and the clamps the walk met"
    (let%map_open.Command checkpoint, config = config_flags
     and steps =
       flag
         "-steps"
         (optional_with_default 128 int)
         ~doc:"N the steps to draw; the state carries error forward, thus give it many"
     and seed = flag "-seed" (optional_with_default 42 int) ~doc:"N the seed" in
     fun () -> drift ~checkpoint ~config ~steps ~seed)
;;

(* The reference stream: what the board must send, event for event. The line format is the
   one of play_mamba and of the JAX twin, thus the three compare with `diff`. *)
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
   must agree. [Jsb.windows] cuts the windows and [jax/data.py] states the same cut.

   The window is a choice of the referee here and not a property of the model: the
   recurrence has no context length, and each window opens on a zero state. It travels in
   the output so that the two sides cut the same count. *)
let loss ~checkpoint ~config ~corpus ~context =
  let params = Mamba.Params.load config ~path:checkpoint in
  let data = Jsb.load ~path:corpus in
  let random_state = Random.State.make [| 0 |] in
  let stream = List.hd_exn (Jsb.streams data.valid ~count:1 ~random_state) in
  let windows = Jsb.windows stream ~context in
  let { Mamba.Config.d; d_in; heads; state; plan; span; ring; taps = (_ : int) } =
    config
  in
  printf
    "windows %d  context %d  d %d  d_in %d  heads %d  state %d  span %d  ring %d  plan \
     %s  loss %.6f\n"
    (List.length windows)
    context
    d
    d_in
    heads
    state
    span
    ring
    (Mamba.Kind.spell plan)
    (Mamba.loss config params ~windows)
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
     and context =
       flag
         "-context"
         (optional_with_default 256 int)
         ~doc:"N the window the referee cuts; a recurrence has no context of its own"
     in
     fun () -> loss ~checkpoint ~config ~corpus ~context)
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
         (optional_with_default Mamba.elected_temperature float)
         ~doc:"F"
     and min_p =
       flag "-min-p" (optional_with_default Mamba.elected_min_p float) ~doc:"F"
     in
     fun () -> stream ~checkpoint ~config ~steps ~seed ~temperature ~min_p)
;;

let () =
  Command_unix.run
    (Command.group
       ~summary:"one checkpoint of the state-space model"
       [ "drift", drift_command; "loss", loss_command; "stream", stream_command ])
;;
