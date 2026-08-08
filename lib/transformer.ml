open Core
module Ptree = Kaun.Ptree

type tensor = (float, Nx.float32_elt) Nx.t

(* the rows of the bar-phase table: the steps of one bar *)
let phase_buckets = 16

(* the element count of a shape: how many draws the walk owes it *)
let numel shape = Array.fold shape ~init:1 ~f:( * )

module Config = struct
  type t =
    { d : int
    ; layers : int
    ; heads : int
    ; context : int
    ; slope_span : int
    (** the ALiBi exponent span: the slope of head k is 2 ** -(span (k+1) / heads) *)
    }

  let baseline = { d = 64; layers = 2; heads = 4; context = 256; slope_span = 8 }

  (* The width and the layer count are in the shapes of the checkpoint: the embedding
     table is [vocab; d], and the layers take six tensors each after the two tables. The
     heads and the context are not there — no tensor shape holds them, because the heads
     only split the width at run time and ALiBi holds no position table. *)
  let of_checkpoint path ~heads ~context ~slope_span =
    let archive = Nx_io.load_safetensors path in
    let embed =
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
    { d = (Nx.shape embed).(1); layers = (tensors - 2) / 6; heads; context; slope_span }
  ;;
end

module Params = struct
  type t =
    { embed : tensor
    ; phase : tensor
    ; layers : layer array
    }

  and layer =
    { wq : tensor
    ; wk : tensor
    ; wv : tensor
    ; wo : tensor
    ; w1 : tensor
    ; w2 : tensor
    }

  let to_list { embed; phase; layers } =
    embed
    :: phase
    :: List.concat_map (Array.to_list layers) ~f:(fun { wq; wk; wv; wo; w1; w2 } ->
      [ wq; wk; wv; wo; w1; w2 ])
  ;;

  let of_list (config : Config.t) tensors =
    match tensors with
    | embed :: phase :: rest ->
      let layers =
        List.chunks_of rest ~length:6
        |> List.map ~f:(function
          | [ wq; wk; wv; wo; w1; w2 ] -> { wq; wk; wv; wo; w1; w2 }
          | _ -> invalid_arg "a layer takes six tensors")
        |> Array.of_list
      in
      if Array.length layers <> config.layers
      then
        invalid_argf
          "%d layer groups do not fit %d layers"
          (Array.length layers)
          config.layers
          ();
      { embed; phase; layers }
    | _ -> invalid_arg "the parameters start with the two tables"
  ;;

  (* the shapes in the flat order of [to_list], which [of_list] reads back *)
  let shapes (config : Config.t) =
    let d = config.d in
    (* wq, wk, wv and wo, then w1 and w2 of the feed-forward *)
    let layer_shapes =
      [ [| d; d |]; [| d; d |]; [| d; d |]; [| d; d |]; [| d; 4 * d |]; [| 4 * d; d |] ]
    in
    [ [| Token.vocab; d |]; [| phase_buckets; d |] ]
    @ List.concat (List.init config.layers ~f:(fun (_ : int) -> layer_shapes))
  ;;

  (* The draw is a walk, thus the order of the tensors is part of the result. A record
     literal and [Array.init] leave that order to the compiler; the fold over [shapes]
     states it. *)
  let draw config ~seed =
    let open Prng in
    (* the seam: [Prng] draws the numbers, Nx gives them a shape *)
    let normal shape =
      let+ draws = normals ~count:(numel shape) ~scale:0.02 in
      Nx.create Nx.float32 shape draws
    in
    let (_ : Prng.state), tensors =
      Prng.run (all (List.map (shapes config) ~f:normal)) (Prng.create_folded ~seed)
    in
    of_list config tensors
  ;;

  let to_ptree t = Ptree.list (List.map (to_list t) ~f:Ptree.tensor)

  let of_ptree config ptree =
    let leaves, (_ : Ptree.tensor list -> Ptree.t) = Ptree.flatten ptree in
    of_list
      config
      (List.map leaves ~f:(fun leaf ->
         match Ptree.Tensor.to_typed Nx.float32 leaf with
         | Some tensor -> tensor
         | None -> invalid_arg "a checkpoint tensor is not float32"))
  ;;

  let save t ~path = Kaun.Checkpoint.save path (to_ptree t)

  (* [Kaun.Checkpoint.load] reads the structure, the shapes and the dtype of the template
     and never its values, thus zeros serve and the load needs no draw. *)
  let load config ~path =
    let zeros shape = Ptree.tensor (Nx.zeros Nx.float32 shape) in
    let like = Ptree.list (List.map (shapes config) ~f:zeros) in
    of_ptree config (Kaun.Checkpoint.load path ~like)
  ;;
end

(* [batch; length] rows of codes become the [batch; length; num_classes] one-hot block.
   Float32, because the block goes into a matmul and carries a gradient. *)
let one_hot_rows ~num_classes rows =
  let batch = Array.length rows in
  let length = Array.length rows.(0) in
  let codes =
    Nx.init Nx.int32 [| batch; length |] (fun index ->
      Int32.of_int_exn rows.(index.(0)).(index.(1)))
  in
  Nx.astype Nx.float32 (Nx.one_hot ~num_classes codes)
;;

(* the table lookup as one-hot times table: small, and the gradient flows *)
let embed_rows table ~num_classes rows = Nx.matmul (one_hot_rows ~num_classes rows) table

(* RMSNorm with no scale: the trainer of the design document folds the scale away *)
let rms_norm x =
  let axis = Array.length (Nx.shape x) - 1 in
  let mean_square = Nx.mean (Nx.square x) ~axes:[ axis ] ~keepdims:true in
  Nx.mul x (Nx.rsqrt (Nx.add_s mean_square 1e-6))
;;

(* ALiBi and the causal wall, shape [1; heads; length; length]. The slope of head k is
   2 ** -(span (k+1) / heads): a power of two, a shift in the circuit. The slope is a
   recency prior, and the span sets how far the gentlest head sees — at the paper's span
   of 8 that head stands at -4 logits by 1024 tokens and -8 by 2048, which is blind to a
   phrase of a chorale. A wider span reaches further and stays a power of two. *)
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

(* The dropout of one training step, at the blocks the JAX sweep of 2026-08-07 found. A
   block draws from a walk of its own, thus a block that gains or loses draws never moves
   the masks of the others, and the step is a function of the seed alone. *)
module Dropout = struct
  (* A dropout holds its own walk, thus a block that takes one draws its mask alone and no
     state threads through the forward pass. [split] gives the blocks their walks. *)
  type t =
    | Off
    | On of
        { rate : float
        ; rng : Prng.state
        }

  let none = Off

  let create ~rate ~seed =
    (* a rate of 1 drops everything: the scale below divides by a keep of zero *)
    if Float.(rate >= 1.0) then invalid_arg "the dropout rate is below 1";
    if Float.(rate <= 0.0) then Off else On { rate; rng = Prng.create_folded ~seed }
  ;;

  (* [split t ~count] is [count] dropouts, each on a walk of its own. The masks of one are
     therefore independent of the masks of the others, and of how many either draws. *)
  let split t ~count =
    match t with
    | Off -> Array.create ~len:count Off
    | On { rate; rng } ->
      let (_ : Prng.state), rngs =
        Prng.run (Prng.all (List.init count ~f:(fun (_ : int) -> Prng.split))) rng
      in
      List.map rngs ~f:(fun rng -> On { rate; rng }) |> Array.of_list
  ;;

  (* [run t h] drops from [h], thus the mask takes the shape of [h] and no caller states
     the batch or the context. The scale is the inverted form of dropout: a survivor
     carries the mass of the dropped, thus the inference pass rescales nothing. *)
  let run t h =
    match t with
    | Off -> h
    | On { rate; rng } ->
      let keep = 1.0 -. rate in
      let shape = Nx.shape h in
      let (_ : Prng.state), coins =
        Prng.run (Prng.bernoullis ~count:(numel shape) ~probability:keep) rng
      in
      Array.map coins ~f:(fun coin -> coin /. keep)
      |> Nx.create Nx.float32 shape
      |> Nx.mul h
  ;;
end

(* the branch alone, dropped: the residual sum happens in [logits] *)
let attention (config : Config.t) (layer : Params.layer) ~bias ~dropout h =
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
  Nx.matmul merged layer.wo |> Dropout.run dropout
;;

let feed_forward (layer : Params.layer) ~dropout h =
  let normed = rms_norm h in
  let hidden = Nx.maximum_s (Nx.matmul normed layer.w1) 0.0 in
  Nx.matmul hidden layer.w2 |> Dropout.run dropout
;;

(* the two tables summed, dropped: the token and the bar phase of each position *)
let embedding (params : Params.t) ~codes ~phases ~dropout =
  let tokens = embed_rows params.embed ~num_classes:Token.vocab codes in
  let bar = embed_rows params.phase ~num_classes:phase_buckets phases in
  Nx.add tokens bar |> Dropout.run dropout
;;

let logits (config : Config.t) (params : Params.t) ~codes ~phases ~dropout =
  let length = Array.length codes.(0) in
  let bias = attention_bias ~heads:config.heads ~length ~span:config.slope_span in
  (* One walk for each block that drops: the embedding sum, then the two branches of each
     layer. A block draws from its own walk, thus the blocks are plain functions and the
     order in which the pass reaches them does not reach the masks. *)
  let walks = Dropout.split dropout ~count:(1 + (2 * config.layers)) in
  let h = embedding params ~codes ~phases ~dropout:walks.(0) in
  let h =
    Array.foldi params.layers ~init:h ~f:(fun layer h weights ->
      let dropout = walks.((2 * layer) + 1) in
      let h = Nx.add h (attention config weights ~bias ~dropout h) in
      let dropout = walks.((2 * layer) + 2) in
      Nx.add h (feed_forward weights ~dropout h))
  in
  Nx.matmul (rms_norm h) (Nx.transpose params.embed)
;;

(* The cross entropy of the labels, weighted by position. A weight of zero drops a
   position from the mean: the padding of a short piece takes it, because a padded label
   would teach the walk to hold the last chord and emit END for ever — the drone. *)
let weighted_cross_entropy raw labels ~weights =
  let axis = 2 in
  let peak = Nx.max raw ~axes:[ axis ] ~keepdims:true in
  let shifted = Nx.sub raw peak in
  let total = Nx.log (Nx.sum (Nx.exp shifted) ~axes:[ axis ] ~keepdims:true) in
  let log_probability = Nx.sub shifted total in
  let hot = one_hot_rows ~num_classes:Token.vocab labels in
  let picked = Nx.sum (Nx.mul log_probability hot) ~axes:[ axis ] in
  let weights =
    Nx.init Nx.float32 (Nx.shape picked) (fun index -> weights.(index.(0)).(index.(1)))
  in
  Nx.neg (Nx.div (Nx.sum (Nx.mul picked weights)) (Nx.sum weights))
;;

(* the additive form of the mask: 0 for a legal code, -1e9 else *)
let mask_bias ~masks =
  let batch = Array.length masks in
  let length = Array.length masks.(0) in
  let open Bigarray in
  let bias = Genarray.create float32 c_layout [| batch; length; Token.vocab |] in
  Genarray.fill bias (-1e9);
  Array.iteri masks ~f:(fun b row ->
    Array.iteri row ~f:(fun t mask ->
      Array.iteri mask ~f:(fun code legal ->
        if legal then Genarray.set bias [| b; t; code |] 0.0)));
  Nx.of_bigarray bias
;;

(* The grammar sits inside the softmax, thus the model spends no mass on a code that the
   sampler would refuse. Therefore its raw mass outside the legal set stays untrained, and
   the same mask must guard every draw. *)
let loss (config : Config.t) params ~codes ~phases ~masks ~weights ~dropout =
  let length = Array.length codes.(0) - 1 in
  let inputs = Array.map codes ~f:(fun row -> Array.subo row ~len:length) in
  let labels = Array.map codes ~f:(fun row -> Array.sub row ~pos:1 ~len:length) in
  let input_phases = Array.map phases ~f:(fun row -> Array.subo row ~len:length) in
  let raw = logits config params ~codes:inputs ~phases:input_phases ~dropout in
  weighted_cross_entropy (Nx.add raw (mask_bias ~masks)) labels ~weights
;;

type sample_stats =
  { refused : float
  ; illegal_mass : float
  ; illegal_top : float
  ; draws : int
  }

(* The running telemetry of one sampling run. It holds the sums; [sample_stats] takes the
   means at the end. *)
type tally =
  { mutable refused : float
  ; mutable illegal_mass : float
  ; mutable illegal_top : int
  ; mutable draws : int
  }

(* The sampler is a loop with state at the edge of the module: the histories, the walk of
   the mask, and the PRNG. The forward pass recomputes the whole window for each token;
   the host affords that, per the design document. *)
let sample (config : Config.t) params ~seed ~steps ~temperature ~min_p =
  if Float.(temperature <= 0.0) then invalid_arg "the temperature is positive";
  if Float.(min_p < 0.0 || min_p >= 1.0) then invalid_arg "min_p is 0 up to 1";
  (* the newest items of a history, oldest first: the row the forward pass reads *)
  let window history = List.take history config.context |> List.rev |> Array.of_list in
  (* The logits of the code that follows the window. The forward pass gives one row for
     each position of the window, and the last row is the draw that comes next. *)
  let next_code_logits ~codes ~phases =
    logits config params ~codes:[| codes |] ~phases:[| phases |] ~dropout:Dropout.none
    |> Nx.get [ 0; Array.length codes - 1 ]
    |> Nx.to_array
  in
  (* the code of the largest logit; the first wins when two are equal *)
  let peak_code raw =
    let best = ref Float.neg_infinity in
    let top = ref 0 in
    Array.iteri raw ~f:(fun code value ->
      if Float.(value > !best)
      then (
        best := value;
        top := code));
    !top
  in
  (* The tempered weight of each code, against the peak of the set that [keep] admits: the
     peak weighs one, and a code outside the set weighs zero. Therefore the min-p filter
     is one compare for each code. *)
  let tempered raw ~keep =
    let peak = ref Float.neg_infinity in
    Array.iteri raw ~f:(fun code value ->
      if keep code && Float.(value > !peak) then peak := value);
    Array.mapi raw ~f:(fun code value ->
      if keep code then Float.exp ((value -. !peak) /. temperature) else 0.0)
  in
  (* the share of the raw mass that sits outside the legal set: the rate at which a
     sampler with no mask would emit an illegal token *)
  let illegal_share raw ~mask =
    let weights = tempered raw ~keep:(fun (_ : int) -> true) in
    let all = Array.fold weights ~init:0.0 ~f:( +. ) in
    let legal =
      Array.foldi weights ~init:0.0 ~f:(fun code total weight ->
        if mask.(code) then total +. weight else total)
    in
    1.0 -. (legal /. all)
  in
  (* [min_p] removes each code whose tempered weight is below [min_p] of the peak's, which
     is one. The peak always stays, thus a draw always exists, and zero turns the filter
     off. *)
  let above_min_p weights =
    if Float.(min_p <= 0.0)
    then weights
    else
      Array.map weights ~f:(fun weight -> if Float.(weight >= min_p) then weight else 0.0)
  in
  (* the code whose running total passes [draw]; a total that never passes it falls to 0 *)
  let pick weights ~draw =
    let rec walk code total =
      if code = Token.vocab - 1
      then code
      else (
        let total = total +. weights.(code) in
        if Float.(total > draw) then code else walk (code + 1) total)
    in
    let chosen = walk 0 0.0 in
    if Float.(weights.(chosen) > 0.0) then chosen else 0
  in
  let stream = ref (Prng.create_folded ~seed) in
  (* The boot of the design document: an empty context, then START — power on, music on.
     START takes phase zero; the host takes zero as the boot value of the bar counter, the
     choice the RTL keeps free. *)
  let codes = ref [ Token.to_code Token.Start ] in
  let phases = ref [ 0 ] in
  let state = ref Sounding_state.silence in
  let step_index = ref 0 in
  let drawn = ref 0 in
  let current = ref [] in
  let out = ref [] in
  let tally = { refused = 0.0; illegal_mass = 0.0; illegal_top = 0; draws = 0 } in
  while !drawn < steps do
    let raw = next_code_logits ~codes:(window !codes) ~phases:(window !phases) in
    let mask = Sounding_state.legal_mask !state in
    (* the guard-fire instrumentation: the raw distribution against the mask *)
    tally.illegal_mass <- tally.illegal_mass +. illegal_share raw ~mask;
    if not mask.(peak_code raw) then tally.illegal_top <- tally.illegal_top + 1;
    let legal = tempered raw ~keep:(fun code -> mask.(code)) in
    let weights = above_min_p legal in
    (* the refused share is the mass the model wanted and the filter did not give *)
    let legal_total = Array.fold legal ~init:0.0 ~f:( +. ) in
    let total = Array.fold weights ~init:0.0 ~f:( +. ) in
    tally.refused <- tally.refused +. ((legal_total -. total) /. legal_total);
    tally.draws <- tally.draws + 1;
    let draw =
      let next, uniform = Prng.run Prng.uniform !stream in
      stream := next;
      uniform *. total
    in
    let code = pick weights ~draw in
    let token = Token.of_code code in
    codes := code :: !codes;
    phases := (!step_index mod phase_buckets) :: !phases;
    state := Sounding_state.step !state token;
    match token with
    | Start ->
      (* the mask refuses START at every draw *)
      assert false
    | On _ | Off _ -> current := token :: !current
    | End ->
      out := List.rev !current :: !out;
      current := [];
      incr step_index;
      incr drawn
  done;
  let count = Float.of_int (max 1 tally.draws) in
  (* the annotation picks the means, not the sums of [tally] beside them *)
  let stats : sample_stats =
    { refused = tally.refused /. count
    ; illegal_mass = tally.illegal_mass /. count
    ; illegal_top = Float.of_int tally.illegal_top /. count
    ; draws = tally.draws
    }
  in
  ~music:(List.rev !out), ~stats
;;

let%expect_test "the seed names the walk" =
  let config = { Config.baseline with d = 32; layers = 1 } in
  let embed params = Nx.to_array (List.hd_exn (Params.to_list params)) in
  let moments draws =
    let count = Float.of_int (Array.length draws) in
    let mean = Array.fold draws ~init:0.0 ~f:( +. ) /. count in
    let square acc draw = acc +. ((draw -. mean) ** 2.0) in
    mean, Float.sqrt (Array.fold draws ~init:0.0 ~f:square /. count)
  in
  (* 0 is legal here and no state of the walk: [Prng.create_folded] moves it to the top *)
  let zero = embed (Params.draw config ~seed:0) in
  let one = embed (Params.draw config ~seed:1) in
  let mean, deviation = moments one in
  printf
    "mean %.4f  deviation %.4f  repeats %b  differs %b\n"
    mean
    deviation
    (Array.equal Float.equal one (embed (Params.draw config ~seed:1)))
    (not (Array.equal Float.equal zero one));
  [%expect {| mean 0.0000  deviation 0.0199  repeats true  differs true |}]
;;

let%expect_test "each block draws from a walk of its own" =
  let dropout = Dropout.create ~rate:0.5 ~seed:3 in
  let blocks = Dropout.split dropout ~count:3 in
  let ones = Nx.ones Nx.float32 [| 1; 1; 32 |] in
  let mask dropout = Nx.to_array (Dropout.run dropout ones) in
  let masks = Array.map blocks ~f:mask in
  let kept mask = Array.count mask ~f:(fun value -> Float.(value > 0.0)) in
  let agree a b = Array.equal Float.equal a b in
  printf
    "a survivor weighs %.1f   kept %d, %d and %d of 32\n"
    (Option.value_exn (Array.max_elt masks.(0) ~compare:Float.compare))
    (kept masks.(0))
    (kept masks.(1))
    (kept masks.(2));
  (* the blocks must differ from one another, and each must repeat on its own walk *)
  printf
    "0 and 1 agree %b   0 and 2 agree %b   block 0 repeats %b\n"
    (agree masks.(0) masks.(1))
    (agree masks.(0) masks.(2))
    (agree masks.(0) (mask blocks.(0)));
  [%expect {| |}]
;;

let%expect_test "the shapes of the forward pass" =
  let config = { Config.d = 8; layers = 1; heads = 2; context = 8; slope_span = 8 } in
  let params = Params.draw config ~seed:1 in
  let out =
    logits
      config
      params
      ~codes:[| [| 0; 188; 60; 0 |] |]
      ~phases:[| [| 0; 1; 1; 1 |] |]
      ~dropout:Dropout.none
  in
  print_s ([%sexp_of: int array] (Nx.shape out));
  [%expect {| (1 4 256) |}]
;;
