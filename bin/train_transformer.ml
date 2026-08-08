(* train_transformer: trains the host transformer of docs/transformer_model.md on the JSB
   chorale corpus, and writes the checkpoint. The sweep of the design document runs
   through the flags: -d, -layers, -heads, -context. The trainer reports the loss of the
   train batches, and at each evaluation point the loss of fixed windows of the train and
   the valid split. The checkpoint keeps the parameters of the best valid loss. *)

open Core
module Evaluation = Mgen.Evaluation
module Jsb = Mgen.Jsb
module Transformer = Mgen.Transformer

(* The training pool. The test split has no job in this project — the valid split picks
   the checkpoints and the sweep winners, and the audition judges the music — thus
   [Train_test] folds it into training. [All] serves the final board model: no split is
   held out, the step budget comes from the sweep, and the checkpoint is the last step. *)
module Pool = struct
  type t =
    | Train
    | Train_test
    | All

  let arg =
    Command.Arg_type.of_alist_exn [ "train", Train; "train+test", Train_test; "all", All ]
  ;;

  let chorales t (data : Jsb.t) =
    match t with
    | Train -> data.train
    | Train_test -> data.train @ data.test
    | All -> data.train @ data.test @ data.valid
  ;;
end

(* One train row: a window of context + 1 codes from one chorale, with a fresh
   transposition — a draw from the legal shifts of the piece, or from the fixed window of
   the [-augment] control. A short chorale takes padding: the zero word, which is silence. *)
let train_row rng pool ~context ~augment =
  let chorale = pool.(Random.State.int rng (Array.length pool)) in
  let shift =
    match augment with
    | Some window -> Random.State.int rng ((2 * window) + 1) - window
    | None ->
      let shifts = Array.of_list chorale.Jsb.legal_shifts in
      shifts.(Random.State.int rng (Array.length shifts))
  in
  let ~codes, ~phases = Jsb.encode (Jsb.transpose ~by:shift chorale) in
  let masks = Evaluation.masks_after codes in
  let need = context + 1 in
  let length = Array.length codes in
  if length >= need
  then (
    let start = Random.State.int rng (length - need + 1) in
    ( Array.sub codes ~pos:start ~len:need
    , Array.sub phases ~pos:start ~len:context
    , Array.sub masks ~pos:start ~len:context
    , Array.create ~len:context 1.0 ))
  else (
    let steps = Array.count codes ~f:(fun code -> code = 0) in
    let padded_codes = Array.create ~len:need 0 in
    Array.blit ~src:codes ~src_pos:0 ~dst:padded_codes ~dst_pos:0 ~len:length;
    let padded_phases =
      Array.init context ~f:(fun i ->
        if i < length then phases.(i) else (steps + i - length) mod Jsb.bar_steps)
    in
    let padded_masks =
      Array.init context ~f:(fun i ->
        if i < length then masks.(i) else masks.(length - 1))
    in
    (* position [i] draws label [i + 1]: it is real while the label is inside the piece *)
    let weights = Array.init context ~f:(fun i -> if i + 1 < length then 1.0 else 0.0) in
    padded_codes, padded_phases, padded_masks, weights)
;;

let train_batch rng pool ~batch ~context ~augment =
  let rows = List.init batch ~f:(fun (_ : int) -> train_row rng pool ~context ~augment) in
  let codes, phases, masks =
    Evaluation.batch_of_rows
      (List.map rows ~f:(fun (codes, phases, masks, _) -> codes, phases, masks))
  in
  codes, phases, masks, Array.of_list_map rows ~f:(fun (_, _, _, weights) -> weights)
;;

(* The schedule story: a linear warmup to the peak, then a cosine decay to zero over the
   rest of the budget. Kaun's [warmup_cosine] warms up and stays; the decay is the part
   that pays, thus the composition lives here. A warmup of zero is the constant peak. *)
let schedule ~peak ~warmup ~total : Kaun.Optim.Schedule.t =
  fun step ->
  if warmup = 0
  then peak
  else if step <= warmup
  then peak *. Float.of_int step /. Float.of_int warmup
  else (
    let progress =
      Float.of_int (step - warmup) /. Float.of_int (max 1 (total - warmup))
    in
    peak *. 0.5 *. (1. +. Float.cos (Float.pi *. progress)))
;;

let parameter_count params =
  List.sum (module Int) (Transformer.Params.to_list params) ~f:(fun tensor ->
    Array.fold (Nx.shape tensor) ~init:1 ~f:( * ))
;;

let train
  ~corpus
  ~d
  ~layers
  ~heads
  ~context
  ~batch
  ~steps
  ~learning_rate
  ~seed
  ~augment
  ~log_every
  ~eval_every
  ~eval_limit
  ~checkpoint
  ~train_on
  ~warmup
  ~weight_decay
  ~clip
  ~dropout_rate
  ~eval_context
  ~slope_span
  =
  let config = { Transformer.Config.d; layers; heads; context; slope_span } in
  let data = Jsb.load ~path:corpus in
  let pool = Array.of_list (Pool.chorales train_on data) in
  (* The windows of the referee come from whole pieces, thus a long training context
     leaves almost none: 149 valid rows at 256, 56 at 512, 6 at 1024, none at 2048. ALiBi
     holds no position table, so a model trained long evaluates short, and one evaluation
     context makes every run of a sweep compare. *)
  let eval_context = Option.value eval_context ~default:context in
  let train_eval = Evaluation.rows data.train ~context:eval_context ~limit:eval_limit in
  let valid_eval = Evaluation.rows data.valid ~context:eval_context ~limit:eval_limit in
  (* the batch stream takes its own lane: the parameter draw reads [| seed |], the batches
     [| seed; 1 |] and the dropout [| seed; 2 |], thus the three stay distinct *)
  let rng = Random.State.make [| seed; 1 |] in
  let dropout_rng = Random.State.make [| seed; 2 |] in
  let params = ref (Transformer.Params.draw config ~seed) in
  printf
    "corpus: %d train chorales; eval rows: %d train, %d valid; parameters: %d\n%!"
    (Array.length pool)
    (List.length train_eval)
    (List.length valid_eval)
    (parameter_count !params);
  let started = Core_unix.gettimeofday () in
  let algorithm =
    Kaun.Optim.adamw
      ~lr:(schedule ~peak:learning_rate ~warmup ~total:steps)
      ~weight_decay
      ()
  in
  let opt_state = ref (Kaun.Optim.init algorithm (Transformer.Params.to_ptree !params)) in
  let tracker = Kaun.Metric.tracker () in
  let best = ref Float.infinity in
  let evaluate step =
    let train_loss = Evaluation.loss config !params train_eval ~batch in
    let valid_loss = Evaluation.loss config !params valid_eval ~batch in
    let mark =
      if Float.(valid_loss < !best)
      then (
        best := valid_loss;
        (match train_on with
         | Pool.All -> ()
         | Train | Train_test ->
           Option.iter checkpoint ~f:(fun path -> Transformer.Params.save !params ~path));
        "  *")
      else ""
    in
    printf "step %4d  eval  train %.4f  valid %.4f%s\n%!" step train_loss valid_loss mark
  in
  for step = 1 to steps do
    let codes, phases, masks, weights = train_batch rng pool ~batch ~context ~augment in
    (* the dropout takes its own lane, thus the draw and the batch streams stay put *)
    let dropout =
      Transformer.Dropout.draw
        config
        ~rate:dropout_rate
        ~batch
        ~length:context
        ~seed:(Random.State.int dropout_rng Int.max_value)
    in
    let value, grads =
      Rune.value_and_grads
        (fun tensors ->
          let params = Transformer.Params.of_list config tensors in
          Transformer.loss config params ~codes ~phases ~masks ~weights ~dropout)
        (Transformer.Params.to_list !params)
    in
    let grads_tree =
      Transformer.Params.to_ptree (Transformer.Params.of_list config grads)
    in
    let grads_tree =
      if Float.(clip > 0.0)
      then Kaun.Optim.clip_by_global_norm clip grads_tree
      else grads_tree
    in
    let updated, next_state =
      Kaun.Optim.update
        algorithm
        !opt_state
        (Transformer.Params.to_ptree !params)
        grads_tree
    in
    opt_state := next_state;
    params := Transformer.Params.of_ptree config updated;
    Kaun.Metric.observe tracker "loss" (Nx.item [] value);
    if step % log_every = 0 || step = 1
    then (
      printf "step %4d  loss %.4f\n%!" step (Kaun.Metric.mean tracker "loss");
      Kaun.Metric.reset tracker);
    if step % eval_every = 0 || step = steps then evaluate step
  done;
  let seconds = Core_unix.gettimeofday () -. started in
  printf
    "time: %.0f s, %.3f s each step, the evaluations inside\n%!"
    seconds
    (seconds /. Float.of_int steps);
  printf "best valid %.4f\n%!" !best;
  Option.iter checkpoint ~f:(fun path ->
    match train_on with
    | Pool.All ->
      (* the valid split is inside the pool: the budget ends the run, the last step is the
         model *)
      Transformer.Params.save !params ~path;
      printf "checkpoint of the last step: %s\n%!" path
    | Train | Train_test -> printf "checkpoint of the best: %s\n%!" path)
;;

let command =
  Command.basic
    ~summary:"train the transformer of docs/transformer_model.md on the JSB chorales"
    (let%map_open.Command corpus =
       flag
         "-corpus"
         (optional_with_default Jsb.default_path string)
         ~doc:"PATH the jsb-chorales-16th.json file"
     and d =
       flag
         "-d"
         (optional_with_default Transformer.Config.(baseline.d) int)
         ~doc:"N the residual width"
     and layers =
       flag
         "-layers"
         (optional_with_default Transformer.Config.(baseline.layers) int)
         ~doc:"N the layers"
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
     and batch = flag "-batch" (optional_with_default 16 int) ~doc:"N the batch rows"
     and steps = flag "-steps" (optional_with_default 200 int) ~doc:"N the train steps"
     and learning_rate =
       flag "-lr" (optional_with_default 3e-4 float) ~doc:"F the learning rate"
     and seed = flag "-seed" (optional_with_default 1 int) ~doc:"N the seed"
     and augment =
       flag
         "-augment"
         (optional int)
         ~doc:
           "N the fixed transposition window of the old policy, the control; absent \
            draws from the legal shifts of each piece"
     and log_every =
       flag "-log-every" (optional_with_default 10 int) ~doc:"N the log period"
     and eval_every =
       flag "-eval-every" (optional_with_default 100 int) ~doc:"N the evaluation period"
     and eval_limit =
       flag
         "-eval-limit"
         (optional_with_default 128 int)
         ~doc:"N the widest evaluation set, in rows"
     and checkpoint =
       flag "-ckpt" (optional string) ~doc:"PATH write the best checkpoint here"
     and train_on =
       flag
         "-train-on"
         (optional_with_default Pool.Train Pool.arg)
         ~doc:"POOL train, train+test, or all (the final board model)"
     and dropout_rate =
       flag
         "-dropout"
         (optional_with_default 0.0 float)
         ~doc:
           "F the dropout rate; the JAX sweep of 2026-08-07 wants 0.1 at d 64 and 0.2 at \
            d 128 or the long context"
     and slope_span =
       flag
         "-alibi-span"
         (optional_with_default Transformer.Config.(baseline.slope_span) int)
         ~doc:
           "N the ALiBi exponent span: the slope of head k is 2 ** -(span (k+1) / \
            heads). The paper's 8 leaves the gentlest head at -4 logits by 1024 tokens; \
            a wider span sees further. The draw must state the same span."
     and eval_context =
       flag
         "-eval-context"
         (optional int)
         ~doc:"N evaluate at this context; absent takes the training context"
     and warmup =
       flag
         "-warmup"
         (optional_with_default 0 int)
         ~doc:"N warmup steps; 0 keeps the constant rate, else cosine decay follows"
     and weight_decay =
       flag "-wd" (optional_with_default 0.01 float) ~doc:"F the AdamW weight decay"
     and clip =
       flag
         "-clip"
         (optional_with_default 1.0 float)
         ~doc:"F clip the global gradient norm; 0 turns clipping off"
     in
     fun () ->
       train
         ~corpus
         ~d
         ~layers
         ~heads
         ~context
         ~batch
         ~steps
         ~learning_rate
         ~seed
         ~augment
         ~log_every
         ~eval_every
         ~eval_limit
         ~checkpoint
         ~train_on
         ~warmup
         ~weight_decay
         ~clip
         ~dropout_rate
         ~eval_context
         ~slope_span)
;;

let () = Command_unix.run command
