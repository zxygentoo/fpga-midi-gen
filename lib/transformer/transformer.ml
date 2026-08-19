open Core
module Ptree = Kaun.Ptree

type tensor = (float, Nx.float32_elt) Nx.t

let numel shape = Array.fold shape ~init:1 ~f:( * )

module Config = struct
  type t =
    { d : int
    ; layers : int
    ; heads : int
    ; context : int
    ; slope_span : int
    }

  (* the model the ear elected on 2026-08-18, and the defaults of jax/transformer/train.py *)
  let baseline = { d = 64; layers = 6; heads = 4; context = 256; slope_span = 4 }

  let of_checkpoint path ~heads ~context ~slope_span =
    let archive = Nx_io.load_safetensors path in
    let seats =
      match Stdlib.Hashtbl.find_opt archive "0" with
      | Some packed -> Nx_io.to_typed Nx.float32 packed
      | None -> invalid_argf "%s holds no tensor named 0: not a checkpoint" path ()
    in
    let tensors = Stdlib.Hashtbl.length archive in
    if tensors < 8 || (tensors - 2) % 6 <> 0
    then
      invalid_argf
        "%s holds %d tensors: not two tables and six for each layer"
        path
        tensors
        ();
    let shape = Nx.shape seats in
    if Array.length shape <> 3 || shape.(0) <> Frame.voices || shape.(1) <> Vocab.classes
    then
      invalid_argf
        "the seat table of %s is %s, and not %d seats of %d classes"
        path
        (Sexp.to_string ([%sexp_of: int array] shape))
        Frame.voices
        Vocab.classes
        ();
    { d = shape.(2); layers = (tensors - 2) / 6; heads; context; slope_span }
  ;;
end

(* The structure of the parameters over any tensor type, and the flat order of the
   checkpoint with it: the two tables, then six tensors for each layer. The integer twin
   of a later step instantiates the same structure, thus the order has one definition. *)
module Params_data = struct
  type 'a t =
    { seats : 'a (** the four tied tables in one tensor, seat 0 first *)
    ; phase : 'a
    ; layers : 'a layer array
    }

  and 'a layer =
    { wq : 'a
    ; wk : 'a
    ; wv : 'a
    ; wo : 'a
    ; w1 : 'a
    ; w2 : 'a
    }

  let to_list { seats; phase; layers } =
    seats
    :: phase
    :: List.concat_map (Array.to_list layers) ~f:(fun { wq; wk; wv; wo; w1; w2 } ->
      [ wq; wk; wv; wo; w1; w2 ])
  ;;

  let of_list ~layers items =
    match items with
    | seats :: phase :: rest ->
      let groups =
        List.chunks_of rest ~length:6
        |> List.map ~f:(function
          | [ wq; wk; wv; wo; w1; w2 ] -> { wq; wk; wv; wo; w1; w2 }
          | _ -> invalid_arg "a layer takes six tensors")
        |> Array.of_list
      in
      if Array.length groups <> layers
      then
        invalid_argf
          "%d layer groups do not fit %d layers"
          (Array.length groups)
          layers
          ();
      { seats; phase; layers = groups }
    | _ -> invalid_arg "the parameters start with the two tables"
  ;;
end

module Params = struct
  type t = tensor Params_data.t
  type layer = tensor Params_data.layer

  let to_list = Params_data.to_list

  let of_list (config : Config.t) tensors =
    Params_data.of_list ~layers:config.layers tensors
  ;;

  (* the shapes in the flat order of [Params_data.to_list], which [of_list] reads back *)
  let shapes (config : Config.t) =
    let d = config.d in
    (* wq, wk, wv and wo, then w1 and w2 of the feed-forward *)
    let layer_shapes =
      [ [| d; d |]; [| d; d |]; [| d; d |]; [| d; d |]; [| d; 4 * d |]; [| 4 * d; d |] ]
    in
    let tables = [ [| Frame.voices; Vocab.classes; d |]; [| Jsb.bar_steps; d |] ] in
    tables @ List.concat (List.init config.layers ~f:(fun (_ : int) -> layer_shapes))
  ;;

  (* The draw is a walk, thus the order of the tensors is part of the result. A record
     literal and [Array.init] leave that order to the compiler; the fold over [shapes]
     states it. *)
  let init config ~seed =
    let open Prng in
    let normal shape =
      let+ draws = normals ~count:(numel shape) ~scale:0.02 in
      Nx.create Nx.float32 shape draws
    in
    let (_ : Prng.state), tensors =
      Prng.run (all (List.map (shapes config) ~f:normal)) (Prng.create_folded ~seed)
    in
    of_list config tensors
  ;;

  let of_ptree config ptree =
    let leaves, (_ : Ptree.tensor list -> Ptree.t) = Ptree.flatten ptree in
    of_list
      config
      (List.map leaves ~f:(fun leaf ->
         match Ptree.Tensor.to_typed Nx.float32 leaf with
         | Some tensor -> tensor
         | None -> invalid_arg "a checkpoint tensor is not float32"))
  ;;

  (* [Kaun.Checkpoint.load] reads the structure, the shapes and the dtype of the template
     and never its values, thus zeros serve and the load needs no draw. *)
  let load config ~path =
    let zeros shape = Ptree.tensor (Nx.zeros Nx.float32 shape) in
    let like = Ptree.list (List.map (shapes config) ~f:zeros) in
    of_ptree config (Kaun.Checkpoint.load path ~like)
  ;;
end

(* the table of one seat: [Params.seats] holds the four in one tensor, seat 0 first *)
let seat_table (params : Params.t) seat = Nx.contiguous (Nx.get [ seat ] params.seats)

(* float32, because the block goes into a matmul *)
let one_hot_rows ~num_classes rows =
  let batch = Array.length rows in
  let length = Array.length rows.(0) in
  let codes =
    Nx.init Nx.int32 [| batch; length |] (fun index ->
      Int32.of_int_exn rows.(index.(0)).(index.(1)))
  in
  Nx.astype Nx.float32 (Nx.one_hot ~num_classes codes)
;;

(* the table lookup as one-hot times table: small, and one definition serves every table *)
let embed_rows table ~num_classes rows = Nx.matmul (one_hot_rows ~num_classes rows) table

(* The classes of a batch of frames, apart by seat: the result at [seat] holds one row of
   classes for each walk. Four tables read four rows, thus the pass wants the seats apart
   and never the frame whole. *)
let seat_classes frames =
  let classes =
    Array.map frames ~f:(Array.map ~f:(fun frame -> Vocab.classes_of_frame frame))
  in
  Array.init Frame.voices ~f:(fun seat ->
    Array.map classes ~f:(Array.map ~f:(fun row -> List.nth_exn row seat)))
;;

(* RMSNorm with no scale: the trainer of the design document folds the scale away *)
let rms_norm x =
  let axis = Array.length (Nx.shape x) - 1 in
  let mean_square = Nx.mean (Nx.square x) ~axes:[ axis ] ~keepdims:true in
  Nx.mul x (Nx.rsqrt (Nx.add_s mean_square 1e-6))
;;

(* ALiBi and the causal wall, shape [1; heads; length; length]. The slope is a recency
   prior, and [Config.slope_span] of the interface holds its reasoning. *)
let attention_bias ~heads ~length ~span =
  let positions =
    Nx.reshape [| length; 1 |] (Nx.astype Nx.float32 (Nx.arange Nx.int32 0 length 1))
  in
  let distance = Nx.sub positions (Nx.transpose positions) in
  let slopes =
    Nx.init Nx.float32 [| 1; heads; 1; 1 |] (fun index ->
      -.(2. ** (-.Float.of_int span *. Float.of_int (index.(1) + 1) /. Float.of_int heads)))
  in
  let alibi = Nx.mul slopes (Nx.reshape [| 1; 1; length; length |] distance) in
  let wall = Nx.mul_s (Nx.triu ~k:1 (Nx.ones Nx.float32 [| length; length |])) (-1e9) in
  Nx.add alibi (Nx.reshape [| 1; 1; length; length |] wall)
;;

let softmax x =
  let axis = Array.length (Nx.shape x) - 1 in
  let shifted = Nx.sub x (Nx.max x ~axes:[ axis ] ~keepdims:true) in
  let exp = Nx.exp shifted in
  Nx.div exp (Nx.sum exp ~axes:[ axis ] ~keepdims:true)
;;

(* the branch alone: the residual sum happens in [hidden] *)
let attention (config : Config.t) (layer : Params.layer) ~bias h =
  let d = config.d in
  let heads = config.heads in
  let shape = Nx.shape h in
  let batch = shape.(0) in
  let length = shape.(1) in
  let head_d = d / heads in
  let split x =
    Nx.transpose ~axes:[ 0; 2; 1; 3 ] (Nx.reshape [| batch; length; heads; head_d |] x)
  in
  let normed = rms_norm h in
  let q = split (Nx.matmul normed layer.wq) in
  let k = split (Nx.matmul normed layer.wk) in
  let v = split (Nx.matmul normed layer.wv) in
  let scores =
    Nx.add
      (Nx.mul_s
         (Nx.matmul q (Nx.transpose ~axes:[ 0; 1; 3; 2 ] k))
         (1. /. Float.sqrt (Float.of_int head_d)))
      bias
  in
  let context = Nx.matmul (softmax scores) v in
  let merged =
    (* the transpose leaves a strided view; the reshape needs one piece of memory *)
    Nx.reshape
      [| batch; length; d |]
      (Nx.contiguous (Nx.transpose ~axes:[ 0; 2; 1; 3 ] context))
  in
  Nx.matmul merged layer.wo
;;

let feed_forward (layer : Params.layer) h =
  let normed = rms_norm h in
  let hidden = Nx.maximum_s (Nx.matmul normed layer.w1) 0.0 in
  Nx.matmul hidden layer.w2
;;

(* The input of one position: the four seat rows sum, and the bar phase adds to them.

   A shared table with a voice tag cannot work here, and the reason is arithmetic and not
   capacity. Every step carries all four seats, thus the sum of four tags is the same
   vector at every position — a bias, which carries nothing — and what remains is
   symmetric in the four classes. A soprano on 72 over a bass on 48 would give the vector
   of a soprano on 48 under a bass on 72, and the voices would be thrown away on the way
   in. *)
let embedding params ~classes ~phases =
  let phase = embed_rows params.Params_data.phase ~num_classes:Jsb.bar_steps phases in
  Array.foldi classes ~init:phase ~f:(fun seat h rows ->
    Nx.add h (embed_rows (seat_table params seat) ~num_classes:Vocab.classes rows))
;;

(* the residual stream after the last layer and before any readout *)
let hidden (config : Config.t) params ~classes ~phases =
  let length = Array.length phases.(0) in
  let bias = attention_bias ~heads:config.heads ~length ~span:config.slope_span in
  let h = embedding params ~classes ~phases in
  Array.fold params.Params_data.layers ~init:h ~f:(fun h weights ->
    let h = Nx.add h (attention config weights ~bias h) in
    Nx.add h (feed_forward weights h))
;;

(* The seats of the chain, soprano first: the head draws seat 3 and then walks down.

   The order keeps the one decision the ear accepted in era three — the top voice is
   chosen first, and it conditions on no voice under it, as the music is written. *)
let chain_seats = List.rev (List.range 0 Frame.voices)

(* The chained head: the logits of each seat, paired with its seat.

   Each seat reads the stream that the seats above it have written:

   {v
     h3 = h                   logits(seat 3) = E[3] . rms(h3)
     h2 = h3 + E[3][c3]       logits(seat 2) = E[2] . rms(h2)
     h1 = h2 + E[2][c2]       logits(seat 1) = E[1] . rms(h1)
     h0 = h1 + E[1][c1]       logits(seat 0) = E[0] . rms(h0)
   v}

   [drawn] holds the classes the chain conditions on — the true frame in training, where
   the four heads then run in one pass with no sampling. Only seats 3, 2 and 1 are read.

   Four heads that drew in parallel would make the voices conditionally independent, and a
   chord is a joint choice: measured on this model, that costs 0.3157 nats for each step.
   The chain removes the cost for no parameters at all and three adds of a vector. *)
let seat_logits params h ~drawn =
  let (_ : tensor), rows =
    List.fold_map chain_seats ~init:h ~f:(fun stream seat ->
      let table = seat_table params seat in
      let raw = Nx.matmul (rms_norm stream) (Nx.transpose table) in
      let stream =
        if seat = 0
        then stream
        else Nx.add stream (embed_rows table ~num_classes:Vocab.classes drawn.(seat))
      in
      stream, (seat, raw))
  in
  rows
;;

(* the negative log likelihood of one seat: [raw] is [batch; length; classes] *)
let class_nll raw labels =
  let axis = 2 in
  let shifted = Nx.sub raw (Nx.max raw ~axes:[ axis ] ~keepdims:true) in
  let total = Nx.log (Nx.sum (Nx.exp shifted) ~axes:[ axis ] ~keepdims:true) in
  let hot = one_hot_rows ~num_classes:Vocab.classes labels in
  Nx.neg (Nx.sum (Nx.mul (Nx.sub shifted total) hot) ~axes:[ axis ])
;;

(* the inputs and the targets of one window: [context] positions state [context] targets,
   and the last frame of the window is a target alone *)
let inputs rows =
  Array.map rows ~f:(fun row -> Array.subo row ~len:(Array.length row - 1))
;;

let targets rows = Array.map rows ~f:(fun row -> Array.subo row ~pos:1)

let loss (config : Config.t) params ~windows =
  let frames = Array.of_list_map windows ~f:(fun (w : Jsb.stream) -> w.frames) in
  let positions = Array.of_list_map windows ~f:(fun (w : Jsb.stream) -> w.positions) in
  if Array.is_empty frames then invalid_arg "the loss takes one window or more";
  (* the bar phase is the low four bits of the rolling coordinate; the high four were the
     window position, which the ear dropped and the corpus still carries *)
  let phases =
    Array.map (inputs positions) ~f:(Array.map ~f:(fun at -> at % Jsb.bar_steps))
  in
  let classes = seat_classes (inputs frames) in
  let labels = seat_classes (targets frames) in
  let h = hidden config params ~classes ~phases in
  let nll =
    List.fold (seat_logits params h ~drawn:labels) ~init:None ~f:(fun total (seat, raw) ->
      let seat_nll = class_nll raw labels.(seat) in
      Some (Option.value_map total ~default:seat_nll ~f:(Nx.add seat_nll)))
  in
  (* the sum over the seats is the loss of one step, and the mean is over the steps: a
     mean over the predictions would divide by a count that changes with the encoding *)
  Nx.item [] (Nx.mean (Option.value_exn nll))
;;

(* The tempered weight of each class against the peak: the peak weighs one, thus [min_p]
   is a share of the peak and one compare holds the filter. *)
let tempered raw ~temperature =
  let peak = Array.fold raw ~init:Float.neg_infinity ~f:Float.max in
  Array.map raw ~f:(fun value -> Float.exp ((value -. peak) /. temperature))
;;

let above_min_p weights ~min_p =
  if Float.(min_p <= 0.0)
  then weights
  else
    Array.map weights ~f:(fun weight -> if Float.(weight >= min_p) then weight else 0.0)
;;

(* the running totals of the weights, left to right; the last of them is the total *)
let running_totals weights =
  Array.folding_map weights ~init:0.0 ~f:(fun total weight ->
    let total = total +. weight in
    total, total)
;;

(* The class whose running total passes the draw.

   It takes the uniform and not a draw, thus one function owns both sums and the total is
   the last running total — never a second sum of the same weights. Two sums of one array
   differ in the last bits, and a draw made against the other sum can land above every
   running total, where no class passes at all. That case is real in the twin, which adds
   pairwise in [sum] and left to right in [cumsum].

   Against this total the draw is strictly below it: the uniform falls under 1 by 2 ** -24
   at the least, thus the exact product falls short by about 2 ** 29 units in the last
   place, where rounding moves a result by half of one. Therefore the walk always ends on
   a class, and that class always holds weight — to reach the last index is to know that
   no earlier total passed, thus the weight there is the difference of two totals across
   the draw. No fallback is necessary, and none is written. *)
let pick weights ~uniform =
  let running = running_totals weights in
  let last = Array.length running - 1 in
  let draw = uniform *. running.(last) in
  let rec walk index =
    if index = last || Float.(running.(index) > draw) then index else walk (index + 1)
  in
  walk 0
;;

(* The draw of one seat as one function: the tempered weights, the min-p floor, and the
   class whose running total passes the draw. The sampler below and the drift report of
   [Quantized] both take it, thus the two pipelines are comparable pick for pick. *)
let draw_class raw ~temperature ~min_p ~uniform =
  pick (above_min_p (tempered raw ~temperature) ~min_p) ~uniform
;;

(* the row of a table as a stream of one position, thus the chain can add it *)
let table_row table index ~d = Nx.reshape [| 1; d |] (Nx.get [ index ] table)

(* One step of the chained draw, on the host and between two passes of the network: the
   soprano first, and each seat under it reading the stream the seats above have written.
   It gives the classes of the frame, seat 0 first. *)
let draw_frame (config : Config.t) params ~temperature ~min_p ~rng stream =
  let (rng, (_ : tensor)), classes =
    List.fold_map chain_seats ~init:(rng, stream) ~f:(fun (rng, stream) seat ->
      let table = seat_table params seat in
      let raw = Nx.to_array (Nx.matmul (rms_norm stream) (Nx.transpose table)) in
      let rng, uniform = Prng.run Prng.uniform rng in
      let index = draw_class raw ~temperature ~min_p ~uniform in
      let stream =
        if seat = 0 then stream else Nx.add stream (table_row table index ~d:config.d)
      in
      (rng, stream), index)
  in
  (* the chain runs from the soprano down, and a frame reads from seat 0 up *)
  rng, List.rev classes
;;

(* The logits of every seat at the last position of one window, over the classes the chain
   conditions on. The drift report of [Quantized] walks the quantized engine and reads
   this for each of its four draws, thus the two models are compared on one history and
   one chain, and what is left between them is the quantization. *)
let logits (config : Config.t) params ~frames ~positions ~drawn =
  let phases = [| Array.map positions ~f:(fun at -> at % Jsb.bar_steps) |] in
  let h = hidden config params ~classes:(seat_classes [| frames |]) ~phases in
  let last = Array.length frames - 1 in
  let stream = Nx.reshape [| 1; 1; config.d |] (Nx.get [ 0; last ] h) in
  let rows =
    seat_logits params stream ~drawn:(Array.map drawn ~f:(fun index -> [| [| index |] |]))
  in
  Array.init Frame.voices ~f:(fun seat ->
    List.Assoc.find_exn rows seat ~equal:Int.equal |> Nx.to_array)
;;

(* the state of one walk: the generator, and the history with the newest step first *)
type walk =
  { rng : Prng.state
  ; frames : int list
  ; phases : int list
  }

(* The newest steps of a history, oldest first: the row the forward pass reads. The drift
   report of [Quantized] cuts its own history by this rule and compares the two models
   over the result, thus the rule stands here and not inside the sampler. *)
let window history ~context = List.take history context |> List.rev |> Array.of_list

(* The bounds of the draw. The quantized twin states the same two, thus one module owns
   them and a reader finds one message for each. *)
let check_policy ~temperature ~min_p =
  if Float.(temperature <= 0.0) then invalid_arg "the temperature is positive";
  if Float.(min_p < 0.0 || min_p >= 1.0) then invalid_arg "min_p is 0 up to 1"
;;

(* The draw the ear elected on 2026-08-18. Every player, the quantized twin and the
   bitstream start from these two, thus a number the ear moves moves one time. *)
let elected_temperature = 1.0
let elected_min_p = 0.05

let sample (config : Config.t) params ~seed ~steps ~temperature ~min_p =
  check_policy ~temperature ~min_p;
  let add walk ~frame ~step =
    { walk with
      frames = frame :: walk.frames
    ; phases = (step % Jsb.bar_steps) :: walk.phases
    }
  in
  (* The boot of docs/transformer.md: a lead-in of silence. Attention needs one position,
     and the packed corpus holds a run of silent frames at every seam, thus this is a
     condition the model trained on and the model opens the music itself. One bar is the
     longest seam of that corpus, and it leaves the first draw on a downbeat. The lead-in
     counts inside [steps], because it is silence the walk really plays. *)
  let lead = min steps Jsb.bar_steps in
  let booted =
    List.fold
      (List.range 0 lead)
      ~init:{ rng = Prng.create_folded ~seed; frames = []; phases = [] }
      ~f:(fun walk step -> add walk ~frame:Frame.silent ~step)
  in
  let drawn =
    List.fold (List.range lead steps) ~init:booted ~f:(fun walk step ->
      let frames = window walk.frames ~context:config.context in
      let phases = window walk.phases ~context:config.context in
      let h =
        hidden config params ~classes:(seat_classes [| frames |]) ~phases:[| phases |]
      in
      let stream =
        Nx.reshape [| 1; config.d |] (Nx.get [ 0; Array.length frames - 1 ] h)
      in
      let rng, classes =
        draw_frame config params ~temperature ~min_p ~rng:walk.rng stream
      in
      add { walk with rng } ~frame:(Vocab.frame_of_classes classes) ~step)
  in
  Array.of_list (List.rev drawn.frames)
;;

(* the shapes of a test model: small enough to run in a test, and the same structure *)
let test_config = { Config.baseline with d = 32; layers = 1; heads = 2; context = 24 }

let%expect_test "the shapes of the forward pass" =
  let params = Params.init test_config ~seed:1 in
  let frames = [| [| Frame.silent; 0xcac6c1ba; 0xca00c1ba |] |] in
  let phases = [| [| 0; 1; 2 |] |] in
  let classes = seat_classes frames in
  let h = hidden test_config params ~classes ~phases in
  print_s ([%sexp_of: int array] (Nx.shape h));
  [%expect {| (1 3 32) |}];
  (* one set of logits for each seat, over the classes of the vocabulary *)
  List.iter (seat_logits params h ~drawn:classes) ~f:(fun (seat, raw) ->
    printf "seat %d: %s\n" seat (Sexp.to_string ([%sexp_of: int array] (Nx.shape raw))));
  [%expect
    {|
    seat 3: (1 3 48)
    seat 2: (1 3 48)
    seat 1: (1 3 48)
    seat 0: (1 3 48)
    |}]
;;

let%expect_test "the loss of drawn weights is the uniform draw" =
  (* Weights of scale 0.02 put every class near the same logit, thus one seat costs about
     ln 48 = 3.871 nats and a step costs four of them. The number states the unit: nats
     for each STEP, and never nats for each prediction. *)
  let chorale cells = { Jsb.cells = Array.create ~len:8 cells; legal_shifts = [ 0 ] } in
  let stream = Jsb.pack [ chorale [ 67; 64; 60; 48 ]; chorale [ 69; 65; 62; 50 ] ] in
  let windows = Jsb.windows stream ~context:8 in
  let params = Params.init test_config ~seed:2 in
  printf
    "%d windows, %.4f nats for each step\n"
    (List.length windows)
    (loss test_config params ~windows);
  [%expect {| 3 windows, 14.8437 nats for each step |}]
;;

let%expect_test "the seed names the walk" =
  let params = Params.init test_config ~seed:3 in
  let draw seed =
    sample
      test_config
      params
      ~seed
      ~steps:20
      ~temperature:elected_temperature
      ~min_p:elected_min_p
  in
  let walk = draw 7 in
  (* The lead-in of one bar stands at the head of the walk, thus the first draw is
     step 16. These weights are drawn and not trained, thus the model opens at once; a
     trained one opens inside one bar of the end of the lead-in. *)
  let opens_at, (_ : int) =
    Option.value_exn (Array.findi walk ~f:(fun (_ : int) frame -> frame <> Frame.silent))
  in
  printf "%d steps, and the walk opens at step %d\n" (Array.length walk) opens_at;
  Array.iteri walk ~f:(fun step frame ->
    if step >= opens_at then printf "  step %d  %08x\n" step frame);
  [%expect
    {|
    20 steps, and the walk opens at step 16
      step 16  ceafc5a4
      step 17  c2acaeb8
      step 18  c0b3acb8
      step 19  c8bcb1bd
    |}];
  let same = Array.equal Int.equal in
  printf "the same seed repeats: %b\n" (same (draw 7) (draw 7));
  printf "another seed parts: %b\n" (not (same (draw 7) (draw 8)));
  [%expect {|
    the same seed repeats: true
    another seed parts: true
    |}]
;;
