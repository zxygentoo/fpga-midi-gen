(* checkpoint_tool: operations on one checkpoint — the drift of the quantized reference
   against the float model, the reference event stream, and the windowed texture of the
   endless walk. The config flags must equal the flags of the training run; the checkpoint
   holds only tensors. *)

open Core
module Quantized = Mgen_transformer.Quantized
module Jsb = Mgen_corpus.Jsb
module Texture = Mgen_transformer.Texture
module Token = Mgen_core.Token
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

(* The walks of a dump: the line format that [play_transformer] prints and
   [jax/transformer/infer.py] writes, one block for each seed. Gate C holds the two
   writers to that one format, thus a dump measures whatever drew it. *)
let walks_of_lines lines =
  let step_of_line line =
    match String.split line ~on:' ' |> List.filter ~f:(Fn.non String.is_empty) with
    | "step" :: (_ : string) :: events ->
      Some
        (List.filter_map events ~f:(fun text ->
           match String.lsplit2 text ~on:':' with
           | Some ("on", pitch) -> Some (Token.On (Int.of_string pitch))
           | Some ("off", pitch) -> Some (Token.Off (Int.of_string pitch))
           | Some (_ : string * string) | None -> None))
    | (_ : string list) -> None
  in
  let close walks current = if List.is_empty current then walks else current :: walks in
  let aux (walks, current) line =
    if String.is_prefix line ~prefix:"# seed"
    then close walks (List.rev current), []
    else (
      match step_of_line line with
      | Some step -> walks, step :: current
      | None -> walks, current)
  in
  let walks, current = List.fold lines ~init:([], []) ~f:aux in
  List.rev (close walks (List.rev current))
;;

let drawn_walks ~checkpoint ~heads ~context ~slope_span ~steps ~seeds ~temperature ~min_p =
  let config = Transformer.Config.of_checkpoint checkpoint ~heads ~context ~slope_span in
  let params = Transformer.Params.load config ~path:checkpoint in
  List.map
    (List.range 1 (seeds + 1))
    ~f:(fun seed ->
      let ~music, ~stats:(_ : Transformer.sample_stats) =
        Transformer.sample config params ~seed ~steps ~temperature ~min_p
      in
      music)
;;

(* The measurement of step 2 of the plan of docs/transformer_model.md: does the texture of
   the endless walk hold? One number over a whole draw cannot answer it, because a good
   opening hides a bad end. Therefore each seed gives a row of windows, and the report
   quotes the spread over the seeds window by window: the walk holds when the last window
   reads like the first and like the corpus. *)
let texture ~label ~span ~music =
  let rows = List.map music ~f:(Texture.windows ~span) in
  let reference =
    Texture.windows
      (Texture.steps_of_frames (Jsb.pack (Jsb.load ~path:Jsb.default_path).train).frames)
      ~span:Int.max_value
  in
  (* the mean of the seeds, and the two ends of their spread: one seed says nothing *)
  let quote values =
    let sorted = List.sort values ~compare:Float.compare in
    printf
      "  %5.2f [%.2f %.2f]"
      (List.sum (module Float) values ~f:Fn.id /. Float.of_int (List.length values))
      (List.hd_exn sorted)
      (List.last_exn sorted)
  in
  printf
    "%s: %d walks, %d steps, windows of %d\n"
    label
    (List.length music)
    (List.fold music ~init:0 ~f:(fun most walk -> max most (List.length walk)))
    span;
  List.iter reference ~f:(fun (w : Texture.window) ->
    printf
      "the packed corpus:  onsets/step %.2f   single-ON %.2f   median dur %.1f   under a \
       quarter %.2f\n\n"
      w.onsets
      w.single_on
      w.median_duration
      w.under_a_quarter);
  printf
    "step        onsets/step         single-ON         median dur       under a quarter\n";
  List.iteri (List.transpose_exn rows) ~f:(fun index seeds_of_window ->
    printf "%5d" (index * span);
    let of_each f = quote (List.map seeds_of_window ~f) in
    of_each (fun (w : Texture.window) -> w.onsets);
    of_each (fun (w : Texture.window) -> w.single_on);
    of_each (fun (w : Texture.window) -> w.median_duration);
    of_each (fun (w : Texture.window) -> w.under_a_quarter);
    printf "\n%!")
;;

let texture_command =
  Command.basic
    ~summary:"the windowed texture of the endless walk, over many seeds"
    (let%map_open.Command checkpoint =
       flag
         "-ckpt"
         (optional string)
         ~doc:
           "PATH draw the walks from this checkpoint. The draw of one long walk costs \
            minutes here, thus a twelve-seed measurement takes -walks instead."
     and walks =
       flag
         "-walks"
         (optional string)
         ~doc:
           "PATH measure the walks of a dump: the step lines of play_transformer, or of \
            jax/transformer/infer.py, which draws the seeds in one batch"
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
     and slope_span =
       flag
         "-alibi-span"
         (optional_with_default Transformer.Config.(baseline.slope_span) int)
         ~doc:"N the ALiBi exponent span; it must equal the span of the training run"
     and steps =
       flag "-steps" (optional_with_default 8192 int) ~doc:"N the steps of each walk"
     and span =
       flag "-span" (optional_with_default 1024 int) ~doc:"N the steps of one window"
     and seeds =
       flag
         "-seeds"
         (optional_with_default 12 int)
         ~doc:"N the walks, seeded 1 up to N; the design document asks for twelve"
     and temperature =
       flag "-temperature" (optional_with_default 0.9 float) ~doc:"F the temperature"
     and min_p =
       flag
         "-min-p"
         (optional_with_default 0.00390625 float)
         ~doc:"F drop tokens under this share of the peak"
     in
     fun () ->
       match checkpoint, walks with
       | Some checkpoint, None ->
         texture
           ~label:(Filename.basename checkpoint)
           ~span
           ~music:
             (drawn_walks
                ~checkpoint
                ~heads
                ~context
                ~slope_span
                ~steps
                ~seeds
                ~temperature
                ~min_p)
       | None, Some path ->
         texture
           ~label:(Filename.basename path)
           ~span
           ~music:(walks_of_lines (In_channel.read_lines path))
       | None, None | Some (_ : string), Some (_ : string) ->
         Printf.eprintf "give -ckpt to draw the walks, or -walks to measure a dump\n";
         exit 2)
;;

let command =
  Command.group
    ~summary:"operations on one checkpoint"
    [ "drift", drift_command; "stream", stream_command; "texture", texture_command ]
;;

let () = Command_unix.run command
