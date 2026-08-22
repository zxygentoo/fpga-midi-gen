open Core
module Ptree = Kaun.Ptree

type tensor = (float, Nx.float32_elt) Nx.t

let numel = Mgen_nn.Checkpoint.numel

(* The depth of the key and value ring an attention layer reads at inference, and it is
   ERA FOUR'S TRAINING WINDOW. A block carries a state of one size and knows no context;
   an attention layer carries a ring, thus a plan that holds one has a context where the
   trunk had none. At this depth a window of [loss] reads exactly the attention the
   trainer computed, because the window and the ring are the same 256 steps.

   A ring of 128 was measured against a 512-step window and parts from it by 7.6e-05
   relative, thus the depth can buy back four tiles of block RAM if a build ever wants
   them. It holds only while the ALiBi span is 4: span and ring are one decision. *)
let elected_ring = 256

(* Era four's elected ALiBi span, and the span a checkpoint that states none was trained
   at: the trainer of [jax/mamba/train.py] defaults its flag to this same number. Spans 4
   and 8 were measured again in this trunk and read null, thus era four's stands. *)
let elected_span = 4

(* The kinds of layer, and the PLAN of a model is the sequence of them. Six blocks is the
   trunk of docs/mamba.md; the elected model of the era puts the Zamba attention sublayer
   and then the feed-forward after them.

   [Attention] is half a Zamba block: the query and the key read the ORIGINAL EMBEDDING
   beside the residual stream, thus their matrices are [2 d] by [d]; the value reads the
   stream alone. Era four's plain attention — a square query over the stream — was
   measured null in this trunk three times, thus it is not a kind here. *)
module Kind = struct
  type t =
    | Block
    | Attention
    | Feed_forward
  [@@deriving equal, sexp_of]

  (* the tensors a layer of this kind holds in the checkpoint *)
  let tensors = function
    | Block -> 6
    | Attention -> 4
    | Feed_forward -> 2
  ;;

  (* A plan as one letter for each layer, which is how the design document, the checkpoint
     names and the [--plan] flag of the trainer all spell it. The elected model is
     MMMMMMZF. The trainer knows a fourth letter, A for era four's plain attention; this
     library does not, and [Config.of_checkpoint] refuses a file that holds one. *)
  let spell plan =
    String.of_char_list
      (List.map (Array.to_list plan) ~f:(function
        | Block -> 'M'
        | Attention -> 'Z'
        | Feed_forward -> 'F'))
  ;;
end

module Config = struct
  type t =
    { d : int
    ; d_in : int
    ; heads : int
    ; state : int
    ; taps : int
    ; plan : Kind.t array
    ; span : int
    ; ring : int
    }

  (* the shape of docs/mamba.md: d 64, the expansion of two, one head for each 32 lanes,
     the Mamba-1 state width, the Mamba tap count, and the elected plan — six blocks, the
     Zamba head, the feed-forward — at era four's ALiBi span *)
  let baseline =
    { d = 64
    ; d_in = 128
    ; heads = 4
    ; state = 16
    ; taps = 4
    ; plan = [| Block; Block; Block; Block; Block; Block; Attention; Feed_forward |]
    ; span = 4
    ; ring = elected_ring
    }
  ;;

  let blocks t = Array.count t.plan ~f:(Kind.equal Block)
  let attentions t = Array.count t.plan ~f:(Kind.equal Attention)

  (* The ordinal of each layer among the layers of ITS OWN KIND, and it is what indexes a
     memory. A layer's place in the plan is not its place in a memory: the state RAM and
     the tap ring hold one region for each block, and the key and value rings one for each
     attention layer, thus the seventh layer of the elected plan owns ring 0. *)
  let ordinals t =
    Array.folding_map t.plan ~init:[] ~f:(fun seen kind ->
      kind :: seen, List.count seen ~f:(Kind.equal kind))
  ;;

  (* P, the head width: the state is [heads] blocks of [head] by [state] *)
  let head t = t.d_in / t.heads

  (* the head width of an attention layer, which splits the residual width and not [d_in] *)
  let head_d t = t.d / t.heads

  (* the channels the convolution walks: x, then B and C *)
  let channels t = t.d_in + (2 * t.state)

  (* the width of the input projection: the gate, the convolution input and the raw dt *)
  let projection t = (2 * t.d_in) + (2 * t.state) + t.heads

  (* Every width AND THE PLAN are in the file, thus a player states none of them. Era four
     had to be told the heads, the context and the ALiBi span — none of the three sizes a
     tensor there, and a span played back wrong is silently wrong music. Here the head
     count sizes [dt_bias], the state width falls out of the projection, the tensor shapes
     name the kind of each layer, and the span stands after the last layer as a value of
     its own.

     [ring] is the one number that cannot be in the file: it is the depth of the ring at
     INFERENCE, which is a choice of the player and not a fact of the training run. *)
  let of_checkpoint ?(ring = elected_ring) path =
    let archive = Nx_io.load_safetensors path in
    let tensor_at index =
      Option.map
        (Stdlib.Hashtbl.find_opt archive (Int.to_string index))
        ~f:(Nx_io.to_typed Nx.float32)
    in
    let shape_at index = Option.map (tensor_at index) ~f:Nx.shape in
    let refuse name index wanted =
      invalid_argf
        "the %s of %s is %s at tensor %d, and not %s"
        name
        path
        (match shape_at index with
         | Some shape -> Sexp.to_string ([%sexp_of: int array] shape)
         | None -> "absent")
        index
        wanted
        ()
    in
    (match shape_at 0 with
     | Some [| voices; classes; (_ : int) |]
       when voices = Frame.voices && classes = Vocab.classes -> ()
     | _ ->
       refuse
         "seat table"
         0
         (Printf.sprintf "%d seats of %d classes" Frame.voices Vocab.classes));
    let d = (Option.value_exn (shape_at 0)).(2) in
    (* THE FIRST TENSOR OF A GROUP NAMES ITS KIND, thus the walk is sequential and it
       reads the kind before it reads the count. [w_in] is [d] by the projection, and the
       projection is [2 d_in + 2 N + H] — never [d], [2 d] or [4 d] — thus no block head
       can be read as an attention or a feed-forward head. *)
    let kind_at index =
      match shape_at index with
      | Some [| rows; cols |] when rows = 2 * d && cols = d -> Kind.Attention
      | Some [| rows; cols |] when rows = d && cols = 4 * d -> Kind.Feed_forward
      | Some [| rows; cols |] when rows = d && cols = d ->
        invalid_argf
          "%s opens a layer with a square query at tensor %d: era four's attention is \
           not a layer of this model"
          path
          index
          ()
      | Some [| (_ : int); (_ : int) |] -> Kind.Block
      | _ -> refuse "layer" index "a matrix"
    in
    (* the plan, and where each group starts with it: the checks below read the tensors of
       the first BLOCK, which is not tensor 2 under every plan *)
    let rec walk index plan =
      match shape_at index with
      (* the span stands alone after the last layer, and no layer group opens with a
         single value — [w_in], [wq] and [w1] are all matrices *)
      | None | Some [| 1 |] -> List.rev plan, index
      | Some (_ : int array) ->
        let kind = kind_at index in
        walk (index + Kind.tensors kind) ((kind, index) :: plan)
    in
    let plan, after = walk 2 [] in
    let tensors = Stdlib.Hashtbl.length archive in
    let span =
      match tensor_at after with
      | Some value -> Int.of_float (Float.round_nearest (Nx.item [ 0 ] value))
      (* a file that states no span was trained at era four's, which is the one the
         trainer defaults to *)
      | None -> elected_span
    in
    if List.is_empty plan || after + Bool.to_int (after < tensors) <> tensors
    then
      invalid_argf
        "%s holds %d tensors: not two tables, whole layer groups and the span"
        path
        tensors
        ();
    let at =
      match List.find plan ~f:(fun (kind, (_ : int)) -> Kind.equal kind Block) with
      | Some ((_ : Kind.t), index) -> fun k -> index + k
      | None ->
        invalid_argf
          "%s holds no block: a plan of attention alone is not this model"
          path
          ()
    in
    (match shape_at (at 5) with
     | Some [| (_ : int); cols |] when cols = d -> ()
     | _ -> refuse "output projection" (at 5) (Printf.sprintf "a matrix of %d columns" d));
    (match shape_at (at 2) with
     | Some [| (_ : int) |] -> ()
     | _ -> refuse "dt bias" (at 2) "one value for each head");
    let d_in = (Option.value_exn (shape_at (at 5))).(0) in
    let heads = (Option.value_exn (shape_at (at 2))).(0) in
    let width = (2 * d_in) + heads in
    (match shape_at (at 0) with
     | Some [| rows; cols |] when rows = d && cols >= width + 2 && (cols - width) % 2 = 0
       -> ()
     | _ ->
       refuse
         "input projection"
         (at 0)
         (Printf.sprintf "%d rows of two states above %d" d width));
    if d_in % heads <> 0
    then
      invalid_argf
        "%s takes an inner width of %d over %d heads, which do not divide it"
        path
        d_in
        heads
        ();
    if d % heads <> 0
    then
      invalid_argf
        "%s takes a residual width of %d over %d heads, which do not divide it"
        path
        d
        heads
        ();
    let state = ((Option.value_exn (shape_at (at 0))).(1) - width) / 2 in
    (match shape_at (at 1) with
     | Some [| rows; (_ : int) |] when rows = d_in + (2 * state) -> ()
     | _ ->
       refuse
         "kernel"
         (at 1)
         (Printf.sprintf "K taps for each of %d channels" (d_in + (2 * state))));
    { d
    ; d_in
    ; heads
    ; state
    ; taps = (Option.value_exn (shape_at (at 1))).(1)
    ; plan = Array.of_list_map plan ~f:fst
    ; span
    ; ring
    }
  ;;
end

(* The structure of the parameters over any tensor type, and the flat order of the
   checkpoint with it: the two tables, then the tensors of each layer in the order of the
   plan. The integer twin instantiates the same structure, thus the order has one
   definition. *)
module Params_data = struct
  type 'a t =
    { seats : 'a (** the four tied tables in one tensor, seat 0 first *)
    ; phase : 'a
    ; layers : 'a layer array
    }

  and 'a layer =
    | Block of 'a block
    | Attention of 'a attention
    | Feed_forward of 'a feed_forward

  and 'a block =
    { w_in : 'a (** the projection: the gate, the convolution input, the raw dt *)
    ; conv : 'a (** the depthwise kernel, one row of taps for each channel *)
    ; dt_bias : 'a
    ; a_log : 'a (** the log of the decay rate of each head *)
    ; d_skip : 'a (** the skip of each head, around the state *)
    ; w_out : 'a
    }

  and 'a attention =
    { wq : 'a (** [2 d] by [d]: the query reads the stream beside the embedding *)
    ; wk : 'a (** [2 d] by [d], the same source *)
    ; wv : 'a (** [d] by [d]: the value reads the stream alone *)
    ; wo : 'a
    }

  and 'a feed_forward =
    { w1 : 'a (** [d] by [4 d] *)
    ; w2 : 'a (** [4 d] by [d] *)
    }

  let layer_to_list = function
    | Block { w_in; conv; dt_bias; a_log; d_skip; w_out } ->
      [ w_in; conv; dt_bias; a_log; d_skip; w_out ]
    | Attention { wq; wk; wv; wo } -> [ wq; wk; wv; wo ]
    | Feed_forward { w1; w2 } -> [ w1; w2 ]
  ;;

  let to_list { seats; phase; layers } =
    seats :: phase :: List.concat_map (Array.to_list layers) ~f:layer_to_list
  ;;

  (* one group of the flat order as its layer, and the items after it *)
  let layer_of_kind kind items =
    match kind, items with
    | Kind.Block, w_in :: conv :: dt_bias :: a_log :: d_skip :: w_out :: rest ->
      Block { w_in; conv; dt_bias; a_log; d_skip; w_out }, rest
    | Kind.Attention, wq :: wk :: wv :: wo :: rest -> Attention { wq; wk; wv; wo }, rest
    | Kind.Feed_forward, w1 :: w2 :: rest -> Feed_forward { w1; w2 }, rest
    | kind, (_ : 'a list) ->
      invalid_argf
        "a %s layer takes %d tensors"
        (Sexp.to_string (Kind.sexp_of_t kind))
        (Kind.tensors kind)
        ()
  ;;

  let of_list ~plan items =
    match items with
    | seats :: phase :: rest ->
      let rest, groups =
        Array.fold_map plan ~init:rest ~f:(fun items kind ->
          let layer, rest = layer_of_kind kind items in
          rest, layer)
      in
      if not (List.is_empty rest)
      then invalid_argf "%d tensors stand after the plan" (List.length rest) ();
      { seats; phase; layers = groups }
    | _ -> invalid_arg "the parameters start with the two tables"
  ;;
end

module Params = struct
  type t = tensor Params_data.t
  type layer = tensor Params_data.layer
  type block = tensor Params_data.block
  type attention = tensor Params_data.attention
  type feed_forward = tensor Params_data.feed_forward

  let to_list = Params_data.to_list
  let of_list (config : Config.t) tensors = Params_data.of_list ~plan:config.plan tensors

  (* the shapes in the flat order of [Params_data.to_list], which [of_list] reads back *)
  let shapes (config : Config.t) =
    let { Config.d; d_in; heads; state = (_ : int); _ } = config in
    let layer_shapes = function
      | Kind.Block ->
        [ [| d; Config.projection config |]
        ; [| Config.channels config; config.taps |]
        ; [| heads |]
        ; [| heads |]
        ; [| heads |]
        ; [| d_in; d |]
        ]
      | Kind.Attention -> [ [| 2 * d; d |]; [| 2 * d; d |]; [| d; d |]; [| d; d |] ]
      | Kind.Feed_forward -> [ [| d; 4 * d |]; [| 4 * d; d |] ]
    in
    let tables = [ [| Frame.voices; Vocab.classes; d |]; [| Jsb.bar_steps; d |] ] in
    tables @ List.concat_map (Array.to_list config.plan) ~f:layer_shapes
  ;;

  (* The draw of the initial parameters, in the flat order of the checkpoint: the order is
     part of the result, thus the walk states it and a record literal does not.

     It is the draw of jax/mamba/train.py in shape and not in value — the two generators
     are different and only a trained checkpoint crosses that seam. What matters here is
     that the DISTRIBUTIONS agree: a drift report over weights that put every decay near
     one, or the skip near zero, would measure a model this era does not train. *)
  let init (config : Config.t) ~seed =
    let open Prng in
    let normal shape =
      let+ draws = normals ~count:(numel shape) ~scale:0.02 in
      Nx.create Nx.float32 shape draws
    in
    let uniforms ~count = all (List.init count ~f:(fun (_ : int) -> uniform)) in
    let d = config.d in
    let heads = config.heads in
    let block =
      let* w_in = normal [| d; Config.projection config |] in
      let* conv = normal [| Config.channels config; config.taps |] in
      (* the inverse softplus of a step in [0.001, 0.1]: softplus of it is that step *)
      let* steps = uniforms ~count:heads in
      let* rates = uniforms ~count:heads in
      let+ w_out = normal [| config.d_in; d |] in
      Params_data.Block
        { w_in
        ; conv
        ; dt_bias =
            Nx.create
              Nx.float32
              [| heads |]
              (Array.of_list_map steps ~f:(fun u ->
                 Float.log (Float.exp (0.001 +. (u *. 0.099)) -. 1.0)))
        ; a_log =
            Nx.create
              Nx.float32
              [| heads |]
              (Array.of_list_map rates ~f:(fun u -> Float.log (1.0 +. (u *. 15.0))))
        ; d_skip = Nx.ones Nx.float32 [| heads |]
        ; w_out
        }
    in
    (* the four matrices of the head and the two of the feed-forward take the same normal
       as every other matrix: era four drew them so, and one rule covers the whole model *)
    let attention =
      let* wq = normal [| 2 * d; d |] in
      let* wk = normal [| 2 * d; d |] in
      let* wv = normal [| d; d |] in
      let+ wo = normal [| d; d |] in
      Params_data.Attention { wq; wk; wv; wo }
    in
    let feed_forward =
      let* w1 = normal [| d; 4 * d |] in
      let+ w2 = normal [| 4 * d; d |] in
      Params_data.Feed_forward { w1; w2 }
    in
    let layer = function
      | Kind.Block -> block
      | Kind.Attention -> attention
      | Kind.Feed_forward -> feed_forward
    in
    let draw =
      let* seats = normal [| Frame.voices; Vocab.classes; d |] in
      let* phase = normal [| Jsb.bar_steps; d |] in
      let+ layers = all (List.map (Array.to_list config.plan) ~f:layer) in
      { Params_data.seats; phase; layers = Array.of_list layers }
    in
    snd (Prng.run draw (Prng.create_folded ~seed))
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
let silu x = Nx.mul x (Nx.sigmoid x)

(* softplus, in the two parts the integer twin also takes: the ramp, which is exact, and a
   correction that a table holds there. Written this way the float model and the twin
   agree about WHICH quantity a table approximates, and the drift report answers for that
   one. *)
let softplus x = Nx.add (Nx.relu x) (Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs x))) 1.0))

let embedding params ~classes ~phases =
  Reference.embedding
    ~seats:params.Params_data.seats
    ~phase:params.Params_data.phase
    ~classes
    ~phases
;;

module Memory = struct
  type layer =
    | Block of
        { state : tensor (** [batch; heads; head; state] — the recurrence *)
        ; taps : tensor (** [batch; taps - 1; channels] — the steps behind this one *)
        }
    | Ring of
        { keys : tensor (** [batch; ring; d], the newest at the end *)
        ; values : tensor
        ; filled : int (** the steps the walk has really taken, up to the ring *)
        }
    | Stateless (** the feed-forward reads one step and remembers none *)

  type t = layer array

  let origin (config : Config.t) ~batch =
    let zeros shape = Nx.zeros Nx.float32 shape in
    Array.map config.plan ~f:(function
      | Kind.Block ->
        Block
          { state = zeros [| batch; config.heads; Config.head config; config.state |]
          ; taps = zeros [| batch; config.taps - 1; Config.channels config |]
          }
      (* The ring opens empty and carries a count of the steps really taken, thus its
         unwritten slots are masked and the first step of a walk attends to itself alone —
         which is the first position of a training window exactly. *)
      | Kind.Attention ->
        Ring
          { keys = zeros [| batch; config.ring; config.d |]
          ; values = zeros [| batch; config.ring; config.d |]
          ; filled = 0
          }
      | Kind.Feed_forward -> Stateless)
  ;;
end

(* One step of the depthwise causal convolution: the sum over the taps, and the taps of
   the step after it.

   Tap k reads the input k steps back and tap 0 is the step itself, thus a walk that has
   not run K steps yet reads zeros for the taps it does not have. The origin needs no
   clearing walk under this rule, which is why the tap ring of the circuit holds it. *)
let convolve (conv : tensor) ~taps ~u =
  let shape = Nx.shape u in
  let batch = shape.(0) in
  let channels = shape.(1) in
  (* K is the width of the kernel and nothing else states it *)
  let width = (Nx.shape conv).(1) in
  let history = Nx.concatenate ~axis:1 [ Nx.reshape [| batch; 1; channels |] u; taps ] in
  let kernel = Nx.reshape [| 1; width; channels |] (Nx.contiguous (Nx.transpose conv)) in
  ( Nx.sum (Nx.mul history kernel) ~axes:[ 1 ]
  , Nx.contiguous (Nx.slice [ A; R (0, width - 1); A ] history) )
;;

(* The state update and the readout of one step:

   {v
     S[p, n] <- alpha * S[p, n] + x[p] * (dt * B[n])
     y[p]     = sum over n of S[p, n] * C[n] + D * x[p]
   v}

   The decay is one scalar for each head — the Mamba-2 form — and that is what makes the
   block affordable on this machine: six exponentials a step for each layer, where Mamba-1
   would want two thousand. The readout reads the state the update just wrote. *)
let selective_state (config : Config.t) (layer : Params.block) ~state ~x ~b ~c ~dt =
  let batch = (Nx.shape x).(0) in
  let heads = config.heads in
  let head = Config.head config in
  let n = config.state in
  let lanes = Nx.reshape [| batch; heads; head |] x in
  let alpha = Nx.exp (Nx.neg (Nx.mul dt (Nx.exp layer.a_log))) in
  let beta =
    Nx.mul (Nx.reshape [| batch; heads; 1 |] dt) (Nx.reshape [| batch; 1; n |] b)
  in
  let state =
    Nx.add
      (Nx.mul (Nx.reshape [| batch; heads; 1; 1 |] alpha) state)
      (Nx.mul
         (Nx.reshape [| batch; heads; head; 1 |] lanes)
         (Nx.reshape [| batch; heads; 1; n |] beta))
  in
  let read =
    Nx.add
      (Nx.sum (Nx.mul state (Nx.reshape [| batch; 1; 1; n |] c)) ~axes:[ 3 ])
      (Nx.mul (Nx.reshape [| 1; heads; 1 |] layer.d_skip) lanes)
  in
  state, Nx.reshape [| batch; config.d_in |] (Nx.contiguous read)
;;

(* One block's branch: the projection, the convolution, the recurrence and the gated norm
   of Mamba-2. The residual join is the caller's, as it is on the JAX side. *)
let block (config : Config.t) (layer : Params.block) ~state ~taps y =
  let d_in = config.d_in in
  let channels = Config.channels config in
  let part tensor ~low ~high = Nx.contiguous (Nx.slice [ A; R (low, high) ] tensor) in
  let zxbcdt = Nx.matmul y layer.w_in in
  let z = part zxbcdt ~low:0 ~high:d_in in
  let u = part zxbcdt ~low:d_in ~high:(d_in + channels) in
  let dt_raw = part zxbcdt ~low:(d_in + channels) ~high:(Config.projection config) in
  let conv_out, taps = convolve layer.conv ~taps ~u in
  let xbc = silu conv_out in
  let x = part xbc ~low:0 ~high:d_in in
  let b = part xbc ~low:d_in ~high:(d_in + config.state) in
  let c = part xbc ~low:(d_in + config.state) ~high:channels in
  let dt = softplus (Nx.add dt_raw layer.dt_bias) in
  let state, read = selective_state config layer ~state ~x ~b ~c ~dt in
  Memory.Block { state; taps }, rms_norm (Nx.mul read (silu z))
;;

(* The bias of one step against the ring, over the slots and the heads.

   Slot [j] sits [ring - 1 - j] steps back, thus the distance IS the slot and no position
   counter enters the arithmetic. A slot whose distance stands at or above the fill has
   never been written, and the wall refuses it: a walk that has not filled the ring reads
   the scores a training window reads at the same position. *)
let attention_bias (config : Config.t) ~filled =
  let heads = config.heads in
  let slopes =
    Array.init heads ~f:(fun head -> Reference.alibi_slope ~span:config.span ~heads ~head)
  in
  Nx.init Nx.float32 [| 1; config.ring; heads |] (fun index ->
    let distance = config.ring - 1 - index.(1) in
    if distance >= filled then -1e9 else slopes.(index.(2)) *. Float.of_int distance)
;;

(* One attention layer's branch at one step, against the ring of the keys and values
   behind it. It is era four's attention with one addition: [source] — the normed stream
   beside the normed EMBEDDING — feeds the query and the key, where the value reads the
   stream alone. Six blocks of recurrence smear which note was really played; the
   embedding still says it, and the head needs it to match on. *)
let attention (config : Config.t) (layer : Params.attention) ~keys ~values ~filled ~y ~e =
  let { Config.d; heads; ring; _ } = config in
  let batch = (Nx.shape y).(0) in
  let head_d = Config.head_d config in
  let source = Nx.concatenate ~axis:1 [ y; e ] in
  (* the newest stands at the end, thus a step shifts the ring by one and appends *)
  let append behind row =
    Nx.concatenate
      ~axis:1
      [ Nx.contiguous (Nx.slice [ A; R (1, ring); A ] behind)
      ; Nx.reshape [| batch; 1; d |] row
      ]
  in
  let keys = append keys (Nx.matmul source layer.wk) in
  let values = append values (Nx.matmul y layer.wv) in
  let filled = Int.min (filled + 1) ring in
  let lanes rows = Nx.reshape [| batch; ring; heads; head_d |] rows in
  let query = Nx.reshape [| batch; 1; heads; head_d |] (Nx.matmul source layer.wq) in
  let scores =
    Nx.add
      (Nx.mul_s
         (Nx.sum (Nx.mul query (lanes keys)) ~axes:[ 3 ])
         (1.0 /. Float.sqrt (Float.of_int head_d)))
      (attention_bias config ~filled)
  in
  let weights = Reference.softmax scores ~axis:1 in
  let merged =
    Nx.sum
      (Nx.mul (Nx.reshape [| batch; ring; heads; 1 |] weights) (lanes values))
      ~axes:[ 1 ]
  in
  ( Memory.Ring { keys; values; filled }
  , Nx.matmul (Nx.reshape [| batch; d |] (Nx.contiguous merged)) layer.wo )
;;

(* era four's feed-forward, position-wise, thus one form serves the window and the step *)
let feed_forward (layer : Params.feed_forward) y =
  Nx.matmul (Nx.relu (Nx.matmul y layer.w1)) layer.w2
;;

(* One layer's whole branch at one step, whichever kind it is: the thing the stream adds,
   and the memory after it. [e] is the normed embedding, which only the attention reads. *)
let branch (config : Config.t) layer memory ~y ~e =
  match (layer : Params.layer), (memory : Memory.layer) with
  | Block layer, Block { state; taps } ->
    let after, gated = block config layer ~state ~taps y in
    after, Nx.matmul gated layer.w_out
  | Attention layer, Ring { keys; values; filled } ->
    attention config layer ~keys ~values ~filled ~y ~e
  | Feed_forward layer, Stateless ->
    Memory.Stateless, Reference.feed_forward ~w1:layer.w1 ~w2:layer.w2 y
  | (_ : Params.layer), (_ : Memory.layer) ->
    invalid_arg "the memory of a layer does not fit its kind"
;;

(* One step of the whole trunk: the embedded frame in, the residual stream out, and the
   memory of every layer carried forward. *)
let trunk_step (config : Config.t) (params : Params.t) (memory : Memory.t) h =
  (* the embedding the attention reads, normed once: it is the input of layer 0 and it
     does not change as the stream is written *)
  let e = rms_norm h in
  let h, next =
    List.fold_mapi
      (Array.to_list params.Params_data.layers)
      ~init:h
      ~f:(fun index h layer ->
        let after, added = branch config layer memory.(index) ~y:(rms_norm h) ~e in
        Nx.add h added, after)
  in
  Array.of_list next, h
;;

(* The residual stream at every step of a batch of windows. The walk opens on a zero state
   and an empty ring, which is where the boot of the sampler opens, thus a window is not a
   slice of a longer walk and needs no lead-in of its own.

   A block needs no causal wall: the recurrence cannot see forward, thus causality is the
   shape of the machine and not a mask over it. An attention layer carries the wall in its
   fill count, and at [Config.ring] of 256 — the training window — this walk reads exactly
   the attention the trainer computed over the whole window. *)
let hidden (config : Config.t) params ~classes ~phases =
  let embedded = embedding params ~classes ~phases in
  let shape = Nx.shape embedded in
  let batch = shape.(0) in
  let length = shape.(1) in
  let (_ : Memory.t), rows =
    List.fold_map
      (List.range 0 length)
      ~init:(Memory.origin config ~batch)
      ~f:(fun memory at ->
        trunk_step config params memory (Nx.contiguous (Nx.slice [ A; I at ] embedded)))
  in
  Nx.stack ~axis:1 rows
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

(* the stream of one walk as the tensor the head reads: one row of [d] *)
let stream_tensor stream ~d = Nx.reshape [| 1; d |] (Nx.create Nx.float32 [| d |] stream)

let forward (config : Config.t) params memory ~frame ~phase =
  let classes = seat_classes [| [| frame |] |] in
  let embedded = embedding params ~classes ~phases:[| [| phase |] |] in
  let memory, h =
    trunk_step
      config
      params
      memory
      (Nx.reshape [| 1; config.d |] (Nx.contiguous embedded))
  in
  memory, Nx.to_array h
;;

let logits (config : Config.t) params ~stream ~drawn =
  let rows =
    seat_logits
      params
      (Nx.reshape [| 1; 1; config.d |] (stream_tensor stream ~d:config.d))
      ~drawn:(Array.map drawn ~f:(fun index -> [| [| index |] |]))
  in
  Array.init Frame.voices ~f:(fun seat ->
    List.Assoc.find_exn rows seat ~equal:Int.equal |> Nx.to_array)
;;

let draw_frame (config : Config.t) params ~temperature ~min_p ~rng ~stream =
  Reference.draw_frame
    ~seats:params.Params_data.seats
    ~d:config.d
    ~temperature
    ~min_p
    ~rng
    (stream_tensor stream ~d:config.d)
;;

let check_policy = Mgen_nn.Policy.check_policy
let elected_temperature = Mgen_nn.Policy.elected_temperature
let elected_min_p = Mgen_nn.Policy.elected_min_p

(* the walk of the sampler: the generator, the memory of the model, and the stream the
   chain of the next step will read *)
type walk =
  { rng : Prng.state
  ; memory : Memory.t
  ; stream : float array
  }

let sample (config : Config.t) params ~seed ~steps ~temperature ~min_p =
  check_policy ~temperature ~min_p;
  let one walk at =
    (* The boot of docs/mamba.md: a lead-in of silence, one bar of it, drawing nothing and
       taking no number from the generator. The state opens at zero, which is where a
       training window opens, thus the model meets the condition it trained on and opens
       the music itself. The lead-in counts inside [steps], because it is silence the walk
       really plays. *)
    let rng, frame =
      if at < Jsb.bar_steps
      then walk.rng, Frame.silent
      else (
        let rng, classes =
          draw_frame config params ~temperature ~min_p ~rng:walk.rng ~stream:walk.stream
        in
        rng, Vocab.frame_of_classes classes)
    in
    let memory, stream =
      forward config params walk.memory ~frame ~phase:(at % Jsb.bar_steps)
    in
    { rng; memory; stream }, frame
  in
  let origin =
    { rng = Prng.create_folded ~seed
    ; memory = Memory.origin config ~batch:1
    ; stream = Array.create ~len:config.d 0.0
    }
  in
  let (_ : walk), frames = List.fold_map (List.range 0 steps) ~init:origin ~f:one in
  Array.of_list frames
;;

let refusal = Mgen_nn.Checkpoint.refusal

module For_test = struct
  let with_checkpoint = Mgen_nn.Checkpoint.with_checkpoint
  let refusal = Mgen_nn.Checkpoint.scrubbed_refusal
end

(* the shape of a test model: small enough to run in a test, and the whole plan of the era
   — a block, the Zamba head, the feed-forward — because a plan of one kind would not test
   the reader, the memories or the branch dispatch *)
let test_config =
  { Config.d = 32
  ; d_in = 64
  ; heads = 2
  ; state = 8
  ; taps = 4
  ; plan = [| Block; Attention; Feed_forward |]
  ; span = elected_span
  ; ring = 8
  }
;;

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

(* The one thing that could be wrong in two places at once: the batched walk of [hidden]
   and the one-walk step of [forward] are the same recurrence, and the sampler runs the
   second where the loss runs the first. A tap read backward or a state carried one step
   late would part them here. *)
let%expect_test "the batched walk and the one-walk step are one recurrence" =
  let params = Params.init test_config ~seed:4 in
  let frames = [| Frame.silent; 0xcac6c1ba; 0xca00c1ba; 0xb0b1b2b3; 0xc0c1c2c3 |] in
  let classes = seat_classes [| frames |] in
  let phases = [| Array.mapi frames ~f:(fun at (_ : int) -> at % Jsb.bar_steps) |] in
  let batched = hidden test_config params ~classes ~phases in
  let (_ : Memory.t), rows =
    List.fold_map
      (List.range 0 (Array.length frames))
      ~init:(Memory.origin test_config ~batch:1)
      ~f:(fun memory at ->
        forward test_config params memory ~frame:frames.(at) ~phase:(at % Jsb.bar_steps))
  in
  let gap =
    List.foldi rows ~init:0.0 ~f:(fun at worst row ->
      let batched_row = Nx.to_array (Nx.contiguous (Nx.slice [ I 0; I at ] batched)) in
      Array.fold2_exn row batched_row ~init:worst ~f:(fun worst a b ->
        Float.max worst (Float.abs (a -. b))))
  in
  printf "%d steps, the two walks part by at most %.3e\n" (List.length rows) gap;
  [%expect {| 5 steps, the two walks part by at most 0.000e+00 |}]
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
  [%expect {| 3 windows, 14.9039 nats for each step |}]
;;

(* The one crossing of the JAX-to-OCaml seam, on a file the gate writes itself. What the
   trainer states in a checkpoint is the shape and the values; a reader that took either
   of them wrong would be found on the board and not here, thus the seam runs here. *)
let%expect_test "the checkpoint seam: the readers take what the trainer writes" =
  let params = Params.init test_config ~seed:5 in
  let flat p = Array.concat_map (Array.of_list (Params.to_list p)) ~f:Nx.to_array in
  For_test.with_checkpoint (Params.to_list params) ~f:(fun path ->
    (* every width comes from the shapes in the file, thus a player states none *)
    let read = Config.of_checkpoint ~ring:test_config.ring path in
    printf
      "the file states d %d, d_in %d, heads %d, state %d, taps %d, span %d\n"
      read.d
      read.d_in
      read.heads
      read.state
      read.taps
      read.span;
    printf "the plan is %s\n" (Kind.spell read.plan);
    printf
      "%d values, every one the value written: %b\n"
      (Array.length (flat params))
      (Array.equal Float.equal (flat (Params.load read ~path)) (flat params)));
  [%expect
    {|
    the file states d 32, d_in 64, heads 2, state 8, taps 4, span 4
    the plan is MZF
    28038 values, every one the value written: true
    |}]
;;

let%expect_test "the checkpoint seam: a wrong seat table stops at the door" =
  (* one seat short: the table states the seats and the classes, thus both readers can
     answer for it before any arithmetic reads a row *)
  let tensors =
    List.mapi (Params.shapes test_config) ~f:(fun index shape ->
      Nx.zeros
        Nx.float32
        (if index = 0 then [| Frame.voices - 1; Vocab.classes; test_config.d |] else shape))
  in
  For_test.with_checkpoint tensors ~f:(fun path ->
    let show f = printf "%s\n" (For_test.refusal ~path f) in
    show (fun () ->
      let (_ : Config.t) = Config.of_checkpoint path in
      ());
    show (fun () ->
      let (_ : Params.t) = Params.load test_config ~path in
      ()));
  [%expect
    {|
    the seat table of <file> is (3 48 32) at tensor 0, and not 4 seats of 48 classes
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
     step 16. *)
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
      step 18  bfb3acb8
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
  let flat plan = List.init (2 + Array.sum (module Int) plan ~f:Kind.tensors) ~f:Fn.id in
  let round plan = Params_data.to_list (Params_data.of_list ~plan (flat plan)) in
  let plans =
    [ [| Kind.Block |]; [| Kind.Block; Block |]; Config.baseline.plan; test_config.plan ]
  in
  List.iter plans ~f:(fun plan ->
    printf
      "%s, %d items, the order returns: %b\n"
      (Kind.spell plan)
      (List.length (flat plan))
      ([%compare.equal: int list] (round plan) (flat plan)));
  let read ~plan items =
    refusal (fun () ->
      let (_ : int Params_data.t) = Params_data.of_list ~plan items in
      ())
  in
  printf "one item: %s\n" (read ~plan:[| Kind.Block |] [ 0 ]);
  printf
    "a block one tensor short: %s\n"
    (read ~plan:[| Kind.Block |] (List.init 7 ~f:Fn.id));
  printf
    "a head where a block stands: %s\n"
    (read ~plan:[| Kind.Block; Block |] (flat [| Kind.Block; Attention |]));
  printf
    "two blocks of tensors for one block: %s\n"
    (read ~plan:[| Kind.Block |] (flat [| Kind.Block; Block |]));
  [%expect
    {|
    M, 8 items, the order returns: true
    MM, 14 items, the order returns: true
    MMMMMMZF, 44 items, the order returns: true
    MZF, 14 items, the order returns: true
    one item: the parameters start with the two tables
    a block one tensor short: a Block layer takes 6 tensors
    a head where a block stands: a Block layer takes 6 tensors
    two blocks of tensors for one block: 6 tensors stand after the plan
    |}]
;;
