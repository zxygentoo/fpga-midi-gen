open Core
module Ptree = Kaun.Ptree

type tensor = (float, Nx.float32_elt) Nx.t

(* the rows of the bar-phase table: the steps of one bar *)
let phase_buckets = 16

(* the rows of the piece-position table: the parts of one piece *)
let progress_buckets = 16

(* a chorale runs 228 steps at the median, thus a bucket is 14.2 steps there and 16 is the
   nearest power of two — and a power of two is a bit-slice in the circuit *)
let progress_stride = 16

(* The steps of one synthetic piece at the draw: the arc of the piece-position table, one
   time around. A walk that re-anchors takes this as its period, thus the piece boundary
   and bucket zero are the same instant. *)
let piece_steps = progress_stride * progress_buckets
let numel shape = Array.fold shape ~init:1 ~f:( * )

module Config = struct
  type t =
    { d : int
    ; layers : int
    ; heads : int
    ; context : int
    ; slope_span : int
    }

  let baseline = { d = 64; layers = 2; heads = 4; context = 256; slope_span = 8 }

  let of_checkpoint path ~heads ~context ~slope_span =
    let archive = Nx_io.load_safetensors path in
    let embed =
      match Stdlib.Hashtbl.find_opt archive "0" with
      | Some packed -> Nx_io.to_typed Nx.float32 packed
      | None -> invalid_argf "%s holds no tensor named 0: not a checkpoint" path ()
    in
    let tensors = Stdlib.Hashtbl.length archive in
    if tensors < 9 || (tensors - 3) % 6 <> 0
    then
      invalid_argf
        "%s holds %d tensors: not three tables and six for each layer"
        path
        tensors
        ();
    { d = (Nx.shape embed).(1); layers = (tensors - 3) / 6; heads; context; slope_span }
  ;;
end

module Params_data = struct
  type 'a t =
    { embed : 'a
    ; phase : 'a
    ; progress : 'a
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

  let to_list { embed; phase; progress; layers } =
    embed
    :: phase
    :: progress
    :: List.concat_map (Array.to_list layers) ~f:(fun { wq; wk; wv; wo; w1; w2 } ->
      [ wq; wk; wv; wo; w1; w2 ])
  ;;

  let of_list ~layers items =
    match items with
    | embed :: phase :: progress :: rest ->
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
      { embed; phase; progress; layers = groups }
    | _ -> invalid_arg "the parameters start with the three tables"
  ;;
end

module Params = struct
  type t = tensor Params_data.t
  type layer = tensor Params_data.layer

  let to_list = Params_data.to_list

  let of_list (config : Config.t) tensors =
    Params_data.of_list ~layers:config.layers tensors
  ;;

  (* the shapes in the flat order of [to_list], which [of_list] reads back *)
  let shapes (config : Config.t) =
    let d = config.d in
    (* wq, wk, wv and wo, then w1 and w2 of the feed-forward *)
    let layer_shapes =
      [ [| d; d |]; [| d; d |]; [| d; d |]; [| d; d |]; [| d; 4 * d |]; [| 4 * d; d |] ]
    in
    let tables =
      [ [| Token.vocab; d |]; [| phase_buckets; d |]; [| progress_buckets; d |] ]
    in
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

(* float32, because the block goes into a matmul and carries a gradient *)
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

module Dropout = struct
  type t =
    | Off
    | On of
        { rate : float
        ; rng : Prng.state
        }

  let none = Off

  let create ~rate ~seed =
    if Float.(rate >= 1.0) then invalid_arg "the dropout rate is below 1";
    if Float.(rate <= 0.0) then Off else On { rate; rng = Prng.create_folded ~seed }
  ;;

  (* each walk is independent of the others, and of how many draws either takes *)
  let split t ~count =
    match t with
    | Off -> Array.create ~len:count Off
    | On { rate; rng } ->
      let (_ : Prng.state), rngs =
        Prng.run (Prng.all (List.init count ~f:(fun (_ : int) -> Prng.split))) rng
      in
      List.map rngs ~f:(fun rng -> On { rate; rng }) |> Array.of_list
  ;;

  (* the shape comes from [h]; the scale is the inverted form, thus the inference pass
     rescales nothing *)
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

(* The bar phase says where a step is in the bar; the piece position says where it is in
   the piece, and nothing in the token stream says either. The two tables are the same
   shape and they add together. *)
let embedding (params : Params.t) ~codes ~phases ~progress ~dropout =
  let tokens = embed_rows params.embed ~num_classes:Token.vocab codes in
  let bar = embed_rows params.phase ~num_classes:phase_buckets phases in
  let piece = embed_rows params.progress ~num_classes:progress_buckets progress in
  Nx.add (Nx.add tokens bar) piece |> Dropout.run dropout
;;

let logits (config : Config.t) (params : Params.t) ~codes ~phases ~progress ~dropout =
  let length = Array.length codes.(0) in
  let bias = attention_bias ~heads:config.heads ~length ~span:config.slope_span in
  (* One walk for each block that drops: the embedding sum, then the two branches of each
     layer. A block draws from its own walk, thus the blocks are plain functions and the
     order in which the pass reaches them does not reach the masks. *)
  let walks = Dropout.split dropout ~count:(1 + (2 * config.layers)) in
  let h = embedding params ~codes ~phases ~progress ~dropout:walks.(0) in
  let h =
    Array.foldi params.layers ~init:h ~f:(fun layer h weights ->
      let dropout = walks.((2 * layer) + 1) in
      let h = Nx.add h (attention config weights ~bias ~dropout h) in
      let dropout = walks.((2 * layer) + 2) in
      Nx.add h (feed_forward weights ~dropout h))
  in
  Nx.matmul (rms_norm h) (Nx.transpose params.embed)
;;

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

let loss (config : Config.t) params ~codes ~phases ~progress ~masks ~weights ~dropout =
  let length = Array.length codes.(0) - 1 in
  let inputs = Array.map codes ~f:(fun row -> Array.subo row ~len:length) in
  let labels = Array.map codes ~f:(fun row -> Array.sub row ~pos:1 ~len:length) in
  let input_phases = Array.map phases ~f:(fun row -> Array.subo row ~len:length) in
  let raw = logits config params ~codes:inputs ~phases:input_phases ~progress ~dropout in
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

(* The draw stage of the sampler, at module level: one definition serves [sample], which
   takes its telemetry from the pieces, and [draw_code], which the drift walk of
   [Quantized] runs against the quantized draw. *)

(* The tempered weight of each code, against the peak of the set that [keep] admits: the
   peak weighs one, and a code outside the set weighs zero. Therefore the min-p filter is
   one compare for each code. *)
let tempered raw ~keep ~temperature =
  let peak = ref Float.neg_infinity in
  Array.iteri raw ~f:(fun code value ->
    if keep code && Float.(value > !peak) then peak := value);
  Array.mapi raw ~f:(fun code value ->
    if keep code then Float.exp ((value -. !peak) /. temperature) else 0.0)
;;

let above_min_p weights ~min_p =
  if Float.(min_p <= 0.0)
  then weights
  else
    Array.map weights ~f:(fun weight -> if Float.(weight >= min_p) then weight else 0.0)
;;

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
;;

let draw_code raw ~mask ~temperature ~min_p ~uniform =
  let weights =
    above_min_p (tempered raw ~keep:(fun code -> mask.(code)) ~temperature) ~min_p
  in
  let total = Array.fold weights ~init:0.0 ~f:( +. ) in
  pick weights ~draw:(uniform *. total)
;;

(* The sampler is a loop with state at the edge of the module: the histories, the walk of
   the mask, and the PRNG. The forward pass recomputes the whole window for each token;
   the host affords that, per the design document. *)
let sample (config : Config.t) params ~seed ~steps ~temperature ~min_p ~piece_steps =
  if Float.(temperature <= 0.0) then invalid_arg "the temperature is positive";
  if Float.(min_p < 0.0 || min_p >= 1.0) then invalid_arg "min_p is 0 up to 1";
  Option.iter piece_steps ~f:(fun steps ->
    if steps <= 0 then invalid_arg "piece_steps is positive");
  (* the newest items of a history, oldest first: the row the forward pass reads *)
  let window history = List.take history config.context |> List.rev |> Array.of_list in
  (* The logits of the code that follows the window. The forward pass gives one row for
     each position of the window, and the last row is the draw that comes next. *)
  let next_code_logits ~codes ~phases ~progress =
    logits
      config
      params
      ~codes:[| codes |]
      ~phases:[| phases |]
      ~progress:[| progress |]
      ~dropout:Dropout.none
    |> Nx.get [ 0; Array.length codes - 1 ]
    |> Nx.to_array
  in
  (* The piece position of a step. The corpus divides a piece by its length, but a draw
     has no length to divide by: the board plays for ever, and a long draw would stretch
     one arc of sixteen buckets over a span no training piece ever had. Therefore the draw
     counts instead, and the arc repeats every [progress_stride * progress_buckets] steps
     — a walk of chorale-shaped arcs, one after another. A draw of exactly that many steps
     gives the same buckets the old ratio gave. *)
  let bucket step = step / progress_stride % progress_buckets in
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
  let illegal_share raw ~mask =
    let weights = tempered raw ~keep:(fun (_ : int) -> true) ~temperature in
    let all = Array.fold weights ~init:0.0 ~f:( +. ) in
    let legal =
      Array.foldi weights ~init:0.0 ~f:(fun code total weight ->
        if mask.(code) then total +. weight else total)
    in
    1.0 -. (legal /. all)
  in
  let stream = ref (Prng.create_folded ~seed) in
  (* The boot of the design document: an empty context, then START — power on, music on.
     START takes phase zero; the host takes zero as the boot value of the bar counter, the
     choice the RTL keeps free. *)
  let codes = ref [ Token.to_code Token.Start ] in
  let phases = ref [ 0 ] in
  let progress = ref [ 0 ] in
  let state = ref Sounding_state.silence in
  let step_index = ref 0 in
  let current = ref [] in
  let out = ref [] in
  let tally = { refused = 0.0; illegal_mass = 0.0; illegal_top = 0; draws = 0 } in
  (* The piece boundary: the walk plays a sequence of pieces and not one endless stream,
     which decays into a drone. The histories return to START alone and the sounding set
     to silence, thus the model draws from the condition the corpus trained it on; the
     step index and the PRNG carry, thus each piece is a new draw. START's row reads the
     carried step index, as every row does — at the default arc its phase and bucket are
     zero, because the arc and the two table periods divide it. The release states the
     OFFs of the sounding pitches, climbing, as the grammar would state them.

     The rule is [Quantized.Engine.next_step]'s, and the two need not agree token for
     token: quantization has already parted them. The boundary reads the step index and
     never the music, thus both take it at the same step however far they have parted. *)
  let boundary () =
    match piece_steps with
    | Some steps when !step_index % steps = 0 ->
      (* [current] holds a sentence reversed, thus the release goes in descending and
         comes out climbing, ahead of the tokens the next step draws *)
      current := List.rev_map (Sounding_state.sounding !state) ~f:(fun p -> Token.Off p);
      codes := [ Token.to_code Token.Start ];
      phases := [ !step_index mod phase_buckets ];
      progress := [ bucket !step_index ];
      state := Sounding_state.silence
    | Some (_ : int) | None -> ()
  in
  while !step_index < steps do
    let raw =
      next_code_logits
        ~codes:(window !codes)
        ~phases:(window !phases)
        ~progress:(window !progress)
    in
    let mask = Sounding_state.legal_mask !state in
    tally.illegal_mass <- tally.illegal_mass +. illegal_share raw ~mask;
    if not mask.(peak_code raw) then tally.illegal_top <- tally.illegal_top + 1;
    let legal = tempered raw ~keep:(fun code -> mask.(code)) ~temperature in
    let weights = above_min_p legal ~min_p in
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
    progress := bucket !step_index :: !progress;
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
      (* the loop above walks tokens and this walks steps, thus the boundary lands here *)
      boundary ()
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
  let zero = embed (Params.init config ~seed:0) in
  let one = embed (Params.init config ~seed:1) in
  let mean, deviation = moments one in
  printf
    "mean %.4f  deviation %.4f  repeats %b  differs %b\n"
    mean
    deviation
    (Array.equal Float.equal one (embed (Params.init config ~seed:1)))
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
  [%expect
    {|
    a survivor weighs 2.0   kept 19, 16 and 18 of 32
    0 and 1 agree false   0 and 2 agree false   block 0 repeats true
    |}]
;;

let%expect_test "the shapes of the forward pass" =
  let config = { Config.d = 8; layers = 1; heads = 2; context = 8; slope_span = 8 } in
  let params = Params.init config ~seed:1 in
  let out =
    logits
      config
      params
      ~codes:[| [| 0; 188; 60; 0 |] |]
      ~phases:[| [| 0; 1; 1; 1 |] |]
      ~progress:[| [| 0; 0; 0; 0 |] |]
      ~dropout:Dropout.none
  in
  print_s ([%sexp_of: int array] (Nx.shape out));
  [%expect {| (1 4 256) |}]
;;

(* The mask is the model now, not a filter over it: the loss trained inside it, thus the
   raw mass outside the legal set is untrained and the draw must carry the same mask. This
   walks the drawn music back through that mask. The weights are a draw of their own, so
   the test says nothing of the music itself — only that the grammar held at every token,
   which is the property the sampler owes whatever the weights say. *)
let%expect_test "the sampler draws only what the mask allows" =
  let config = { Config.d = 16; layers = 1; heads = 2; context = 64; slope_span = 8 } in
  let params = Params.init config ~seed:5 in
  let ~music, ~stats =
    (* the short arc crosses two boundaries, thus the walk below also proves the release
       legal: its OFFs climb from a sentence that holds no ON yet *)
    sample
      config
      params
      ~seed:7
      ~steps:24
      ~temperature:0.9
      ~min_p:(1.0 /. 256.0)
      ~piece_steps:(Some 8)
  in
  (* [sample] drops the [End] that closed each sentence; the walk needs it back *)
  let sentence step = step @ [ Token.End ] in
  let walk (state, illegal) token =
    let mask = Sounding_state.legal_mask state in
    let illegal = if mask.(Token.to_code token) then illegal else illegal + 1 in
    Sounding_state.step state token, illegal
  in
  let tokens = List.concat_map music ~f:sentence in
  let (_ : Sounding_state.t), illegal =
    List.fold tokens ~init:(Sounding_state.silence, 0) ~f:walk
  in
  printf
    "%d steps  %d tokens  %d illegal   the mask held %.4f of the raw mass over %d draws\n"
    (List.length music)
    (List.length tokens)
    illegal
    stats.illegal_mass
    stats.draws;
  (* the first steps that carry an event, so the record shows what a sentence looks like *)
  List.filter music ~f:(fun step -> not (List.is_empty step))
  |> (fun steps -> List.take steps 3)
  |> List.iter ~f:(fun step -> print_s ([%sexp_of: Token.t list] step));
  (* the same seed draws the same music; Token.t carries no compare, thus the codes do *)
  let ~music:again, ~stats:_ =
    sample
      config
      params
      ~seed:7
      ~steps:24
      ~temperature:0.9
      ~min_p:(1.0 /. 256.0)
      ~piece_steps:(Some 8)
  in
  let codes steps = List.map steps ~f:(List.map ~f:Token.to_code) in
  printf
    "the seed repeats: %b\n"
    (List.equal (List.equal Int.equal) (codes music) (codes again));
  [%expect
    {|
    24 steps  80 tokens  0 illegal   the mask held 0.8232 of the raw mass over 72 draws
    ((On 114) (On 30) (On 22))
    ((On 82))
    ((Off 22) (On 52))
    the seed repeats: true
    |}]
;;
