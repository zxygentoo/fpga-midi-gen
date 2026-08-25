(* check_diffusion: the referee numbers of one canvas checkpoint — the loss of the float
   model (Gate A of the JAX seam), and the drift of the integer twin against it.

   The tool holds no walk of its own: play_diffusion draws both models, and its step lines
   without -play ARE the reference stream, thus the float walk, the twin and the JAX walk
   all compare with `diff`.

   The checkpoint states its own shape, thus no flag restates the training run. *)

open Core
module Jsb = Mgen_corpus.Jsb
module Diffusion = Mgen_diffusion.Diffusion
module Quantized = Mgen_diffusion.Quantized

let load checkpoint =
  let config = Diffusion.Config.of_checkpoint checkpoint in
  config, Diffusion.Params.load config ~path:checkpoint
;;

let checkpoint_flag =
  let open Command.Param in
  flag "-ckpt" (required string) ~doc:"PATH the checkpoint; it states its own shape"
;;

(* Gate A of the JAX seam: the masked loss of the float model over the deterministic
   canvases — the first [crop] steps of every valid piece that holds them, each masked by
   the Bernoulli half of [Diffusion.gate_mask]. The number is the mean over the canvases
   of each canvas's mean over its hidden cells. [jax/tests/test_parity.py] demands the
   same number of the JAX forward on the same canvases and the same masks, thus a pass
   proves the two forwards agree everywhere the loss can see.

   The shape travels in the output and this side states no number of its own: held apart,
   a width would go on matching a default while one side moved, and the gate would pass
   two different models. *)
let loss ~checkpoint ~corpus ~crop =
  let config, params = load checkpoint in
  let data = Jsb.load ~path:corpus in
  let canvases = Diffusion.gate_canvases data.valid ~steps:crop in
  let total =
    Array.foldi canvases ~init:0.0 ~f:(fun index total classes ->
      let hidden = Diffusion.gate_mask ~index ~steps:crop in
      total +. Diffusion.masked_nll params ~classes ~hidden)
  in
  printf
    "canvases %d  crop %d  layers %d  width %d  loss %.6f\n"
    (Array.length canvases)
    crop
    config.layers
    config.width
    (total /. Float.of_int (Array.length canvases))
;;

let loss_command =
  Command.basic
    ~summary:
      "the masked loss of the float model over the deterministic valid canvases: Gate A \
       of the JAX seam, which the parity test demands of the JAX forward"
    (let%map_open.Command checkpoint = checkpoint_flag
     and corpus =
       flag
         "-corpus"
         (optional_with_default Jsb.default_path string)
         ~doc:"PATH the voice-separated corpus file"
     and crop =
       flag "-crop" (optional_with_default 128 int) ~doc:"T the steps of one canvas"
     in
     fun () -> loss ~checkpoint ~corpus ~crop)
;;

(* the drift of the twin against the float model, on the walk the board takes *)
let drift ~checkpoint ~steps ~walk ~seed =
  let (_ : Diffusion.Config.t), params = load checkpoint in
  let { Quantized.Drift.passes
      ; cells
      ; same_peak
      ; same_draw
      ; mean_cosine
      ; activations_clamped
      }
    =
    Quantized.Drift.walk params ~steps ~walk ~seed
  in
  let share count = 100.0 *. Float.of_int count /. Float.of_int (max 1 cells) in
  printf "%d passes over %d steps redrew %d cells\n" passes steps cells;
  printf
    "against the float model: top-1 %.1f%%  cosine %.4f  same draw %.1f%%\n"
    (share same_peak)
    mean_cosine
    (share same_draw);
  printf "activations on the clamp: %.4f%%\n" (100.0 *. activations_clamped)
;;

let drift_command =
  Command.basic
    ~summary:
      "the drift of the integer twin against the float model: top-1, cosine, the \
       same-draw share and the clamp counters"
    (let%map_open.Command checkpoint = checkpoint_flag
     and steps =
       flag "-steps" (optional_with_default 128 int) ~doc:"T the steps of the canvas"
     and walk =
       flag "-walk" (optional_with_default 32 int) ~doc:"N the Gibbs passes to compare"
     and seed = flag "-seed" (optional_with_default 42 int) ~doc:"N the seed" in
     fun () -> drift ~checkpoint ~steps ~walk ~seed)
;;

let command =
  Command.group
    ~summary:"the referee numbers of one canvas checkpoint"
    [ "loss", loss_command; "drift", drift_command ]
;;

let () = Command_unix.run command
