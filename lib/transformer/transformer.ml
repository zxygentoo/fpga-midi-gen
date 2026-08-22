open Core
module Ptree = Kaun.Ptree

type tensor = (float, Nx.float32_elt) Nx.t

let numel = Mgen_nn.Checkpoint.numel

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

(* the shared float rules: the head, the norm and the draw chain; the trunk stays here *)
module Reference = Mgen_nn.Reference

let seat_classes = Reference.seat_classes
let rms_norm = Reference.rms_norm

(* ALiBi and the causal wall, shape [1; heads; length; length]. The slope is a recency
   prior, and [Config.slope_span] of the interface holds its reasoning. *)
let attention_bias ~heads ~length ~span =
  let positions =
    Nx.reshape [| length; 1 |] (Nx.astype Nx.float32 (Nx.arange Nx.int32 0 length 1))
  in
  let distance = Nx.sub positions (Nx.transpose positions) in
  let slopes =
    Nx.init Nx.float32 [| 1; heads; 1; 1 |] (fun index ->
      Reference.alibi_slope ~span ~heads ~head:index.(1))
  in
  let alibi = Nx.mul slopes (Nx.reshape [| 1; 1; length; length |] distance) in
  let wall = Nx.mul_s (Nx.triu ~k:1 (Nx.ones Nx.float32 [| length; length |])) (-1e9) in
  Nx.add alibi (Nx.reshape [| 1; 1; length; length |] wall)
;;

(* one attention layer's branch over a whole window: [y] is the normed stream, and the
   residual join is the caller's *)
let attention (config : Config.t) (layer : Params.layer) ~bias y =
  let d = config.d in
  let heads = config.heads in
  let shape = Nx.shape y in
  let batch = shape.(0) in
  let length = shape.(1) in
  let head_d = d / heads in
  let split x =
    Nx.transpose ~axes:[ 0; 2; 1; 3 ] (Nx.reshape [| batch; length; heads; head_d |] x)
  in
  let q = split (Nx.matmul y layer.wq) in
  let k = split (Nx.matmul y layer.wk) in
  let v = split (Nx.matmul y layer.wv) in
  let scores =
    Nx.add
      (Nx.mul_s
         (Nx.matmul q (Nx.transpose ~axes:[ 0; 1; 3; 2 ] k))
         (1. /. Float.sqrt (Float.of_int head_d)))
      bias
  in
  let context = Nx.matmul (Reference.softmax scores ~axis:3) v in
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

let embedding params ~classes ~phases =
  Reference.embedding
    ~seats:params.Params_data.seats
    ~phase:params.Params_data.phase
    ~classes
    ~phases
;;

(* the residual stream after the last layer and before any readout *)
let hidden (config : Config.t) params ~classes ~phases =
  let length = Array.length phases.(0) in
  let bias = attention_bias ~heads:config.heads ~length ~span:config.slope_span in
  let h = embedding params ~classes ~phases in
  Array.fold params.Params_data.layers ~init:h ~f:(fun h weights ->
    let h = Nx.add h (attention config weights ~bias (rms_norm h)) in
    Nx.add h (Reference.feed_forward ~w1:weights.w1 ~w2:weights.w2 (rms_norm h)))
;;

let seat_logits params h ~drawn =
  Reference.seat_logits ~seats:params.Params_data.seats h ~drawn
;;

let loss (config : Config.t) params ~windows =
  Reference.loss
    ~seats:params.Params_data.seats
    ~hidden:(fun ~classes ~phases -> hidden config params ~classes ~phases)
    ~windows
;;

let draw_class = Mgen_nn.Policy.draw_class

let draw_frame (config : Config.t) params ~temperature ~min_p ~rng stream =
  Reference.draw_frame
    ~seats:params.Params_data.seats
    ~d:config.d
    ~temperature
    ~min_p
    ~rng
    stream
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
let check_policy = Mgen_nn.Policy.check_policy
let elected_temperature = Mgen_nn.Policy.elected_temperature
let elected_min_p = Mgen_nn.Policy.elected_min_p

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

let refusal = Mgen_nn.Checkpoint.refusal

module For_test = struct
  let with_checkpoint = Mgen_nn.Checkpoint.with_checkpoint
  let refusal = Mgen_nn.Checkpoint.scrubbed_refusal
end

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

(* The one crossing of the JAX-to-OCaml seam, on a file the gate writes itself. What the
   trainer states in a checkpoint is the shape and the values; a reader that took either
   of them wrong would be found on the board and not here, thus the seam runs here. *)
let%expect_test "the checkpoint seam: the readers take what the trainer writes" =
  let { Config.d = (_ : int); layers = (_ : int); heads; context; slope_span } =
    test_config
  in
  let params = Params.init test_config ~seed:5 in
  let flat p = Array.concat_map (Array.of_list (Params.to_list p)) ~f:Nx.to_array in
  For_test.with_checkpoint (Params.to_list params) ~f:(fun path ->
    (* the width and the layer count come from the shapes in the file, thus a player
       states neither *)
    let read = Config.of_checkpoint path ~heads ~context ~slope_span in
    printf "the file states d %d, layers %d\n" read.d read.layers;
    printf
      "%d values, every one the value written: %b\n"
      (Array.length (flat params))
      (Array.equal Float.equal (flat (Params.load read ~path)) (flat params)));
  [%expect
    {|
    the file states d 32, layers 1
    18944 values, every one the value written: true
    |}]
;;

let%expect_test "the checkpoint seam: a wrong seat table stops at the door" =
  let { Config.d; layers = (_ : int); heads; context; slope_span } = test_config in
  (* one seat short: the table states the seats and the classes, thus both readers can
     answer for it before any arithmetic reads a row *)
  let tensors =
    List.mapi (Params.shapes test_config) ~f:(fun index shape ->
      Nx.zeros
        Nx.float32
        (if index = 0 then [| Frame.voices - 1; Vocab.classes; d |] else shape))
  in
  For_test.with_checkpoint tensors ~f:(fun path ->
    let show f = printf "%s\n" (For_test.refusal ~path f) in
    show (fun () ->
      let (_ : Config.t) = Config.of_checkpoint path ~heads ~context ~slope_span in
      ());
    (* The load takes the shapes from the configuration it is given, thus it answers for
       the tensor the file really holds. Its message is Kaun's and not this repository's:
       an upgrade of the library can move that text and fail this gate for no regression
       at all. It stays, because it states where the refusal really comes from and a
       re-promotion costs nothing. *)
    show (fun () ->
      let (_ : Params.t) = Params.load test_config ~path in
      ()));
  [%expect
    {|
    the seat table of <file> is (3 48 32), and not 4 seats of 48 classes
    Checkpoint.load: shape mismatch for "0": expected [4; 48; 32], got [3; 48; 32]
    |}]
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

let%expect_test "a walk shorter than the lead-in is silence, and it is that long" =
  let params = Params.init test_config ~seed:3 in
  let walk =
    sample
      test_config
      params
      ~seed:7
      ~steps:5
      ~temperature:elected_temperature
      ~min_p:elected_min_p
  in
  (* the lead-in counts inside [steps], thus a short walk is short and not one bar long *)
  printf
    "%d steps, all of them silence: %b\n"
    (Array.length walk)
    (Array.for_all walk ~f:(fun frame -> frame = Frame.silent));
  [%expect {| 5 steps, all of them silence: true |}]
;;

let%expect_test "the loss takes one window or more" =
  let params = Params.init test_config ~seed:2 in
  printf
    "%s\n"
    (refusal (fun () ->
       let (_ : float) = loss test_config params ~windows:[] in
       ()));
  [%expect {| the loss takes one window or more |}]
;;

(* The flat order of the checkpoint. [to_list] states it, [of_list] reads it back, and the
   quantized twin instantiates the same structure, thus a break here would put a layer's
   tensors into another layer's seats in both models at one time. *)
let%expect_test "the flat order of the checkpoint reads back" =
  (* the order alone, on numbers: two tables, then six for each layer *)
  let flat layers = List.init (2 + (6 * layers)) ~f:Fn.id in
  let round layers = Params_data.to_list (Params_data.of_list ~layers (flat layers)) in
  List.iter [ 1; 2; 6 ] ~f:(fun layers ->
    printf
      "layers %d, %d items, the order returns: %b\n"
      layers
      (List.length (flat layers))
      ([%compare.equal: int list] (round layers) (flat layers)));
  let read ~layers items =
    refusal (fun () ->
      let (_ : int Params_data.t) = Params_data.of_list ~layers items in
      ())
  in
  printf "one item: %s\n" (read ~layers:1 [ 0 ]);
  printf "a layer one tensor short: %s\n" (read ~layers:1 (List.init 7 ~f:Fn.id));
  printf "two layers of tensors for one layer: %s\n" (read ~layers:1 (flat 2));
  [%expect
    {|
    layers 1, 8 items, the order returns: true
    layers 2, 14 items, the order returns: true
    layers 6, 38 items, the order returns: true
    one item: the parameters start with the two tables
    a layer one tensor short: a layer takes six tensors
    two layers of tensors for one layer: 2 layer groups do not fit 1 layers
    |}]
;;

let%expect_test "the window is the newest steps of the history, oldest first" =
  (* the history holds the newest step first, and a forward pass reads them oldest first *)
  let history = [ 5; 4; 3; 2; 1; 0 ] in
  let show context =
    printf
      "context %d: %s\n"
      context
      (Sexp.to_string ([%sexp_of: int array] (window history ~context)))
  in
  show 10;
  show 6;
  show 3;
  show 1;
  [%expect
    {|
    context 10: (0 1 2 3 4 5)
    context 6: (0 1 2 3 4 5)
    context 3: (3 4 5)
    context 1: (5)
    |}]
;;
