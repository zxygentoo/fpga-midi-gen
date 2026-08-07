open Core
module Ptree = Kaun.Ptree

type tensor = (float, Nx.float32_elt) Nx.t

(* the rows of the bar-phase table: the steps of one bar *)
let phase_buckets = 16

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
  type layer =
    { wq : tensor
    ; wk : tensor
    ; wv : tensor
    ; wo : tensor
    ; w1 : tensor
    ; w2 : tensor
    }

  type t =
    { embed : tensor
    ; phase : tensor
    ; layers : layer array
    }

  (* a deterministic normal draw: the OCaml PRNG and Box-Muller, thus Nx keeps no hidden
     random state and the seed is an input *)
  let normal rng ~scale shape =
    let numel = Array.fold shape ~init:1 ~f:( * ) in
    let draws =
      Array.init numel ~f:(fun _ ->
        let u1 = Float.max 1e-12 (Random.State.float rng 1.0) in
        let u2 = Random.State.float rng 1.0 in
        scale *. Float.sqrt (-2. *. Float.log u1) *. Float.cos (2. *. Float.pi *. u2))
    in
    Nx.init Nx.float32 shape (fun index ->
      let flat =
        Array.foldi index ~init:0 ~f:(fun axis acc i -> (acc * shape.(axis)) + i)
      in
      draws.(flat))
  ;;

  let draw (config : Config.t) ~seed =
    let rng = Random.State.make [| seed |] in
    let normal shape = normal rng ~scale:0.02 shape in
    let d = config.d in
    let embed = normal [| Token.vocab; d |] in
    let phase = normal [| phase_buckets; d |] in
    let layers =
      Array.init config.layers ~f:(fun (_ : int) ->
        { wq = normal [| d; d |]
        ; wk = normal [| d; d |]
        ; wv = normal [| d; d |]
        ; wo = normal [| d; d |]
        ; w1 = normal [| d; 4 * d |]
        ; w2 = normal [| 4 * d; d |]
        })
    in
    { embed; phase; layers }
  ;;

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
end

let int32_tensor rows =
  let batch = Array.length rows in
  let length = Array.length rows.(0) in
  Nx.init Nx.int32 [| batch; length |] (fun index ->
    Int32.of_int_exn rows.(index.(0)).(index.(1)))
;;

(* the table lookup as one-hot times table: small, and the gradient flows *)
let embed_rows table ~num_classes rows =
  let hot = Nx.astype Nx.float32 (Nx.one_hot ~num_classes (int32_tensor rows)) in
  Nx.matmul hot table
;;

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

let softmax_last x =
  let axis = Array.length (Nx.shape x) - 1 in
  let shifted = Nx.sub x (Nx.max x ~axes:[ axis ] ~keepdims:true) in
  let exp = Nx.exp shifted in
  Nx.div exp (Nx.sum exp ~axes:[ axis ] ~keepdims:true)
;;

(* The dropout masks of one training step: the scaled Bernoulli draw, one mask for the
   embedding sum and two for each layer — the attention branch and the feed-forward
   branch, as the JAX sweep of 2026-08-07 found them. The masks are drawn before the
   gradient runs and passed in, thus the step stays pure and the seed reproduces it. *)
module Dropout = struct
  type t = tensor list

  let none = []

  let draw (config : Config.t) ~rate ~batch ~length ~seed =
    if Float.( <= ) rate 0.0
    then none
    else (
      let rng = Random.State.make [| seed |] in
      let keep = 1.0 -. rate in
      let masks = 1 + (2 * config.layers) in
      let numel = batch * length * config.d in
      (* the fill is a loop, not [Array.init]: the draw order of [init] is not the index
         order, and a mask must be the same for the same seed *)
      let draws = Array.create ~len:(masks * numel) 0.0 in
      for i = 0 to Array.length draws - 1 do
        draws.(i)
        <- (if Float.( < ) (Random.State.float rng 1.0) keep then 1.0 /. keep else 0.0)
      done;
      List.init masks ~f:(fun mask ->
        Nx.init Nx.float32 [| batch; length; config.d |] (fun index ->
          let flat = (((index.(0) * length) + index.(1)) * config.d) + index.(2) in
          draws.((mask * numel) + flat))))
  ;;

  (* the head of the list applies here; the tail serves the sites that follow *)
  let apply t h =
    match t with
    | [] -> [], h
    | mask :: rest -> rest, Nx.mul h mask
  ;;
end

(* the branch alone: the residual sum happens in [logits], where dropout can sit between *)
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
  let context = Nx.matmul (softmax_last scores) v in
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

let logits (config : Config.t) (params : Params.t) ~codes ~phases ~dropout =
  let length = Array.length codes.(0) in
  let bias = attention_bias ~heads:config.heads ~length ~span:config.slope_span in
  let tokens = embed_rows params.embed ~num_classes:Token.vocab codes in
  let bar = embed_rows params.phase ~num_classes:phase_buckets phases in
  let dropout, h = Dropout.apply dropout (Nx.add tokens bar) in
  let (_ : Dropout.t), h =
    Array.fold params.layers ~init:(dropout, h) ~f:(fun (dropout, h) layer ->
      let dropout, branch = Dropout.apply dropout (attention config layer ~bias h) in
      let h = Nx.add h branch in
      let dropout, branch = Dropout.apply dropout (feed_forward layer h) in
      dropout, Nx.add h branch)
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
  let hot =
    Nx.astype Nx.float32 (Nx.one_hot ~num_classes:Token.vocab (int32_tensor labels))
  in
  let picked = Nx.sum (Nx.mul log_probability hot) ~axes:[ axis ] in
  let weights =
    Nx.init Nx.float32 (Nx.shape picked) (fun index -> weights.(index.(0)).(index.(1)))
  in
  Nx.neg (Nx.div (Nx.sum (Nx.mul picked weights)) (Nx.sum weights))
;;

(* No mask sits in the loss: the model learns the instrument from the data, and the guard
   of the sampler holds the line at the draw. *)
let loss (config : Config.t) params ~codes ~phases ~weights ~dropout =
  let length = Array.length codes.(0) - 1 in
  let inputs = Array.map codes ~f:(fun row -> Array.subo row ~len:length) in
  let labels = Array.map codes ~f:(fun row -> Array.sub row ~pos:1 ~len:length) in
  let input_phases = Array.map phases ~f:(fun row -> Array.subo row ~len:length) in
  let raw = logits config params ~codes:inputs ~phases:input_phases ~dropout in
  weighted_cross_entropy raw labels ~weights
;;

(* the additive form of the mask, for the control loss: 0 for a legal code, -1e9 else *)
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

let masked_loss (config : Config.t) params ~codes ~phases ~masks ~weights ~dropout =
  let length = Array.length codes.(0) - 1 in
  let inputs = Array.map codes ~f:(fun row -> Array.subo row ~len:length) in
  let labels = Array.map codes ~f:(fun row -> Array.sub row ~pos:1 ~len:length) in
  let input_phases = Array.map phases ~f:(fun row -> Array.subo row ~len:length) in
  let raw = logits config params ~codes:inputs ~phases:input_phases ~dropout in
  weighted_cross_entropy (Nx.add raw (mask_bias ~masks)) labels ~weights
;;

module Guard = struct
  type t =
    | Grammar
    | Hazards
end

module Sample_stats = struct
  type t =
    { refused : float
    ; illegal_mass : float
    ; illegal_top : float
    ; draws : int
    }
end

(* The sampler is a loop with state at the edge of the module: the histories, the walk of
   the mask, and the PRNG. The forward pass recomputes the whole window for each token;
   the host affords that, per the design document. *)
let sample (config : Config.t) params ~seed ~steps ~temperature ~min_p ~guard =
  if Float.( <= ) temperature 0.0 then invalid_arg "the temperature is positive";
  if Float.( < ) min_p 0.0 || Float.( >= ) min_p 1.0 then invalid_arg "min_p is 0 up to 1";
  let rng = Random.State.make [| seed |] in
  (* The boot of the design document: an empty context, then START — power on, music on.
     START takes phase zero; the host takes zero as the boot value of the bar counter, the
     choice the RTL keeps free. *)
  let codes = ref [ Token.to_byte Token.Start ] in
  let phases = ref [ 0 ] in
  let state = ref Sounding_state.silence in
  let step_index = ref 0 in
  let drawn = ref 0 in
  let current = ref [] in
  let out = ref [] in
  let refused = ref 0.0 in
  let illegal_mass = ref 0.0 in
  let illegal_top = ref 0 in
  let draws = ref 0 in
  while !drawn < steps do
    let window list = Array.of_list (List.rev (List.take list config.context)) in
    let window_codes = window !codes in
    let all =
      logits
        config
        params
        ~codes:[| window_codes |]
        ~phases:[| window !phases |]
        ~dropout:Dropout.none
    in
    let last = Nx.to_array (Nx.get [ 0; Array.length window_codes - 1 ] all) in
    let mask =
      match (guard : Guard.t) with
      | Grammar -> Sounding_state.legal_mask !state
      | Hazards -> Sounding_state.safe_mask !state
    in
    (* the guard-fire instrumentation: the raw distribution against the mask *)
    (let raw_best = ref Float.neg_infinity in
     let raw_top = ref 0 in
     Array.iteri last ~f:(fun code value ->
       if Float.( > ) value !raw_best
       then (
         raw_best := value;
         raw_top := code));
     let raw_all = ref 0.0 in
     let raw_legal = ref 0.0 in
     Array.iteri last ~f:(fun code value ->
       let weight = Float.exp ((value -. !raw_best) /. temperature) in
       raw_all := !raw_all +. weight;
       if mask.(code) then raw_legal := !raw_legal +. weight);
     illegal_mass := !illegal_mass +. (1.0 -. (!raw_legal /. !raw_all));
     if not mask.(!raw_top) then incr illegal_top);
    let best = ref Float.neg_infinity in
    Array.iteri last ~f:(fun code value ->
      if mask.(code) && Float.( > ) value !best then best := value);
    let weights =
      Array.mapi last ~f:(fun code value ->
        if mask.(code) then Float.exp ((value -. !best) /. temperature) else 0.0)
    in
    (* A weight is the tempered probability against the peak, whose own weight is one.
       Therefore the min-p filter is one compare for each code, and the peak stays. The
       refused share is the mass the model wanted and the filter did not give. *)
    let legal_total = Array.fold weights ~init:0.0 ~f:( +. ) in
    let weights =
      if Float.( > ) min_p 0.0
      then
        Array.map weights ~f:(fun weight ->
          if Float.( >= ) weight min_p then weight else 0.0)
      else weights
    in
    let total = Array.fold weights ~init:0.0 ~f:( +. ) in
    refused := !refused +. ((legal_total -. total) /. legal_total);
    incr draws;
    let draw = Random.State.float rng total in
    let code =
      let rec walk code acc =
        if code = Token.vocab - 1
        then code
        else (
          let acc = acc +. weights.(code) in
          if Float.( > ) acc draw then code else walk (code + 1) acc)
      in
      let chosen = walk 0 0.0 in
      if Float.( > ) weights.(chosen) 0.0 then chosen else 0
    in
    let token = Token.of_byte code in
    codes := code :: !codes;
    phases := (!step_index mod phase_buckets) :: !phases;
    state := Sounding_state.step !state token;
    match token with
    | Start ->
      (* both guards refuse START at every draw *)
      assert false
    | On _ | Off _ -> current := token :: !current
    | End ->
      out := List.rev !current :: !out;
      current := [];
      incr step_index;
      incr drawn
  done;
  let count = max 1 !draws in
  ( List.rev !out
  , { Sample_stats.refused = !refused /. Float.of_int count
    ; illegal_mass = !illegal_mass /. Float.of_int count
    ; illegal_top = Float.of_int !illegal_top /. Float.of_int count
    ; draws = !draws
    } )
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
