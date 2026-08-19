open Core
module Ptree = Kaun.Ptree

type tensor = (float, Nx.float32_elt) Nx.t

let numel shape = Array.fold shape ~init:1 ~f:( * )

(* The Mamba default, and not a lever of this prototype: four taps carry the short-range
   half of the block, and the state carries the long-range half. The tap ring of the
   circuit is sized by it. *)
let conv_taps = 4

module Config = struct
  type t =
    { d : int
    ; d_in : int
    ; heads : int
    ; state : int
    ; layers : int
    }

  (* the shape of docs/mamba.md: d 64, the expansion of two, one head for each 32 lanes,
     the Mamba-1 state width, era four's depth *)
  let baseline = { d = 64; d_in = 128; heads = 4; state = 16; layers = 6 }

  (* P, the head width: the state is [heads] blocks of [head] by [state] *)
  let head t = t.d_in / t.heads

  (* the channels the convolution walks: x, then B and C *)
  let channels t = t.d_in + (2 * t.state)

  (* the width of the input projection: the gate, the convolution input and the raw dt *)
  let projection t = (2 * t.d_in) + (2 * t.state) + t.heads

  (* Every width is in the file, thus a player states none of them. Era four had to be
     told the heads, the context and the ALiBi span — none of the three sizes a tensor
     there. Here the head count sizes [dt_bias], the state width falls out of the
     projection, and the recurrence has no window at all. *)
  let of_checkpoint path =
    let archive = Nx_io.load_safetensors path in
    let shape_of name =
      match Stdlib.Hashtbl.find_opt archive name with
      | Some packed -> Nx.shape (Nx_io.to_typed Nx.float32 packed)
      | None -> invalid_argf "%s holds no tensor named %s: not a checkpoint" path name ()
    in
    let tensors = Stdlib.Hashtbl.length archive in
    if tensors < 8 || (tensors - 2) % 6 <> 0
    then
      invalid_argf
        "%s holds %d tensors: not two tables and six for each layer"
        path
        tensors
        ();
    let seats = shape_of "0" in
    if Array.length seats <> 3 || seats.(0) <> Frame.voices || seats.(1) <> Vocab.classes
    then
      invalid_argf
        "the seat table of %s is %s, and not %d seats of %d classes"
        path
        (Sexp.to_string ([%sexp_of: int array] seats))
        Frame.voices
        Vocab.classes
        ();
    let d = seats.(2) in
    (* the first layer states the rest: w_out is 2, dt_bias is 4 and w_in is 2 above the
       tables, in the flat order of [Params_data.to_list] *)
    let w_out = shape_of "7" in
    let dt_bias = shape_of "4" in
    let w_in = shape_of "2" in
    if Array.length w_out <> 2 || w_out.(1) <> d
    then
      invalid_argf
        "the output projection of %s is %s, and not a matrix of %d columns"
        path
        (Sexp.to_string ([%sexp_of: int array] w_out))
        d
        ();
    if Array.length dt_bias <> 1
    then
      invalid_argf
        "the dt bias of %s is %s, and not one value for each head"
        path
        (Sexp.to_string ([%sexp_of: int array] dt_bias))
        ();
    let d_in = w_out.(0) in
    let heads = dt_bias.(0) in
    let width = (2 * d_in) + heads in
    if Array.length w_in <> 2
       || w_in.(0) <> d
       || w_in.(1) < width + 2
       || (w_in.(1) - width) % 2 <> 0
    then
      invalid_argf
        "the input projection of %s is %s, and not %d rows of two states above %d"
        path
        (Sexp.to_string ([%sexp_of: int array] w_in))
        d
        width
        ();
    if d_in % heads <> 0
    then
      invalid_argf
        "%s takes an inner width of %d over %d heads, which do not divide it"
        path
        d_in
        heads
        ();
    { d; d_in; heads; state = (w_in.(1) - width) / 2; layers = (tensors - 2) / 6 }
  ;;
end

(* The structure of the parameters over any tensor type, and the flat order of the
   checkpoint with it: the two tables, then six tensors for each layer. The integer twin
   instantiates the same structure, thus the order has one definition. *)
module Params_data = struct
  type 'a t =
    { seats : 'a (** the four tied tables in one tensor, seat 0 first *)
    ; phase : 'a
    ; layers : 'a layer array
    }

  and 'a layer =
    { w_in : 'a (** the projection: the gate, the convolution input, the raw dt *)
    ; conv : 'a (** the depthwise kernel, one row of taps for each channel *)
    ; dt_bias : 'a
    ; a_log : 'a (** the log of the decay rate of each head *)
    ; d_skip : 'a (** the skip of each head, around the state *)
    ; w_out : 'a
    }

  let to_list { seats; phase; layers } =
    seats
    :: phase
    :: List.concat_map
         (Array.to_list layers)
         ~f:(fun { w_in; conv; dt_bias; a_log; d_skip; w_out } ->
           [ w_in; conv; dt_bias; a_log; d_skip; w_out ])
  ;;

  let of_list ~layers items =
    match items with
    | seats :: phase :: rest ->
      let groups =
        List.chunks_of rest ~length:6
        |> List.map ~f:(function
          | [ w_in; conv; dt_bias; a_log; d_skip; w_out ] ->
            { w_in; conv; dt_bias; a_log; d_skip; w_out }
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
    let { Config.d; d_in; heads; state = (_ : int); layers } = config in
    let layer_shapes =
      [ [| d; Config.projection config |]
      ; [| Config.channels config; conv_taps |]
      ; [| heads |]
      ; [| heads |]
      ; [| heads |]
      ; [| d_in; d |]
      ]
    in
    let tables = [ [| Frame.voices; Vocab.classes; d |]; [| Jsb.bar_steps; d |] ] in
    tables @ List.concat (List.init layers ~f:(fun (_ : int) -> layer_shapes))
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
    let heads = config.heads in
    let layer =
      let* w_in = normal [| config.d; Config.projection config |] in
      let* conv = normal [| Config.channels config; conv_taps |] in
      (* the inverse softplus of a step in [0.001, 0.1]: softplus of it is that step *)
      let* steps = uniforms ~count:heads in
      let* rates = uniforms ~count:heads in
      let+ w_out = normal [| config.d_in; config.d |] in
      { Params_data.w_in
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
    let draw =
      let* seats = normal [| Frame.voices; Vocab.classes; config.d |] in
      let* phase = normal [| Jsb.bar_steps; config.d |] in
      let+ layers = all (List.init config.layers ~f:(fun (_ : int) -> layer)) in
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

let silu x = Nx.mul x (Nx.sigmoid x)

(* softplus, in the two parts the integer twin also takes: the ramp, which is exact, and a
   correction that a table holds there. Written this way the float model and the twin
   agree about WHICH quantity a table approximates, and the drift report answers for that
   one. *)
let softplus x = Nx.add (Nx.relu x) (Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs x))) 1.0))

(* The input of one step: the four seat rows sum, and the bar phase adds to them.

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

module Memory = struct
  type layer =
    { state : tensor (** [batch; heads; head; state] — the recurrence *)
    ; taps : tensor (** [batch; conv_taps - 1; channels] — the steps behind this one *)
    }

  type t = layer array

  let origin (config : Config.t) ~batch =
    Array.init config.layers ~f:(fun (_ : int) ->
      { state =
          Nx.zeros Nx.float32 [| batch; config.heads; Config.head config; config.state |]
      ; taps = Nx.zeros Nx.float32 [| batch; conv_taps - 1; Config.channels config |]
      })
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
  let history = Nx.concatenate ~axis:1 [ Nx.reshape [| batch; 1; channels |] u; taps ] in
  let kernel =
    Nx.reshape [| 1; conv_taps; channels |] (Nx.contiguous (Nx.transpose conv))
  in
  ( Nx.sum (Nx.mul history kernel) ~axes:[ 1 ]
  , Nx.contiguous (Nx.slice [ A; R (0, conv_taps - 1); A ] history) )
;;

(* The state update and the readout of one step:

   {v
     S[p, n] <- alpha * S[p, n] + x[p] * (dt * B[n])
     y[p]     = sum over n of S[p, n] * C[n] + D * x[p]
   v}

   The decay is one scalar for each head — the Mamba-2 form — and that is what makes the
   block affordable on this machine: six exponentials a step for each layer, where Mamba-1
   would want two thousand. The readout reads the state the update just wrote. *)
let selective_state (config : Config.t) (layer : Params.layer) ~state ~x ~b ~c ~dt =
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

(* One layer's branch: the projection, the convolution, the recurrence and the gated norm
   of Mamba-2. The residual join is the caller's, as it is on the JAX side. *)
let block (config : Config.t) (layer : Params.layer) (memory : Memory.layer) y =
  let d_in = config.d_in in
  let channels = Config.channels config in
  let part tensor ~low ~high = Nx.contiguous (Nx.slice [ A; R (low, high) ] tensor) in
  let zxbcdt = Nx.matmul y layer.w_in in
  let z = part zxbcdt ~low:0 ~high:d_in in
  let u = part zxbcdt ~low:d_in ~high:(d_in + channels) in
  let dt_raw = part zxbcdt ~low:(d_in + channels) ~high:(Config.projection config) in
  let conv_out, taps = convolve layer.conv ~taps:memory.taps ~u in
  let xbc = silu conv_out in
  let x = part xbc ~low:0 ~high:d_in in
  let b = part xbc ~low:d_in ~high:(d_in + config.state) in
  let c = part xbc ~low:(d_in + config.state) ~high:channels in
  let dt = softplus (Nx.add dt_raw layer.dt_bias) in
  let state, read = selective_state config layer ~state:memory.state ~x ~b ~c ~dt in
  { Memory.state; taps }, rms_norm (Nx.mul read (silu z))
;;

(* One step of the whole trunk: the embedded frame in, the residual stream out, and the
   memory of every layer carried forward. *)
let trunk_step (config : Config.t) (params : Params.t) (memory : Memory.t) h =
  let h, next =
    List.fold_mapi
      (Array.to_list params.Params_data.layers)
      ~init:h
      ~f:(fun index h layer ->
        let after, g = block config layer memory.(index) (rms_norm h) in
        Nx.add h (Nx.matmul g layer.w_out), after)
  in
  Array.of_list next, h
;;

(* The residual stream at every step of a batch of windows. The walk opens on a zero
   state, which is where the boot of the sampler opens, thus a window is not a slice of a
   longer walk and needs no lead-in of its own.

   There is no context parameter and no causal wall: the recurrence cannot see forward,
   thus causality is the shape of the machine and not a mask over it. *)
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

(* The seats of the chain, soprano first: the head draws seat 3 and then walks down.

   The order keeps the one decision the ear accepted in era three — the top voice is
   chosen first, and it conditions on no voice under it, as the music is written. *)
let chain_seats = List.rev (List.range 0 Frame.voices)

(* The chained head of era four, unchanged. Each seat reads the stream that the seats
   above it have written:

   {v
     h3 = h                   logits(seat 3) = E[3] . rms(h3)
     h2 = h3 + E[3][c3]       logits(seat 2) = E[2] . rms(h2)
     h1 = h2 + E[2][c2]       logits(seat 1) = E[1] . rms(h1)
     h0 = h1 + E[1][c1]       logits(seat 0) = E[0] . rms(h0)
   v}

   [drawn] holds the classes the chain conditions on — the true frame in training, where
   the four heads then run in one pass with no sampling. Only seats 3, 2 and 1 are read. *)
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
   running total, where no class passes at all.

   Against this total the draw is strictly below it: the uniform falls under 1 by 2 ** -24
   at the least. Therefore the walk always ends on a class, and that class always holds
   weight — to reach the last index is to know that no earlier total passed, thus the
   weight there is the difference of two totals across the draw. No fallback is necessary,
   and none is written. *)
let pick weights ~uniform =
  let running = running_totals weights in
  let last = Array.length running - 1 in
  let draw = uniform *. running.(last) in
  let rec walk index =
    if index = last || Float.(running.(index) > draw) then index else walk (index + 1)
  in
  walk 0
;;

let draw_class raw ~temperature ~min_p ~uniform =
  pick (above_min_p (tempered raw ~temperature) ~min_p) ~uniform
;;

(* the row of a table as a stream of one position, thus the chain can add it *)
let table_row table index ~d = Nx.reshape [| 1; d |] (Nx.get [ index ] table)

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

(* One step of the chained draw, on the host and between two steps of the recurrence: the
   soprano first, and each seat under it reading the stream the seats above have written.
   It gives the classes of the frame, seat 0 first. *)
let draw_frame (config : Config.t) params ~temperature ~min_p ~rng ~stream =
  let (rng, (_ : tensor)), classes =
    List.fold_map
      chain_seats
      ~init:(rng, stream_tensor stream ~d:config.d)
      ~f:(fun (rng, stream) seat ->
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

(* The bounds of the draw. The quantized twin states the same two, thus one module owns
   them and a reader finds one message for each. *)
let check_policy ~temperature ~min_p =
  if Float.(temperature <= 0.0) then invalid_arg "the temperature is positive";
  if Float.(min_p < 0.0 || min_p >= 1.0) then invalid_arg "min_p is 0 up to 1"
;;

(* The draw era four elected on 2026-08-18, carried over. This era re-elects it by ear
   with the whole chain in view; until then the two eras are auditioned on one policy. *)
let elected_temperature = 1.0
let elected_min_p = 0.05

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

(* the message of the [Invalid_argument] that a rule raises, and ["no raise"] when the
   rule accepts: a gate that pins a message needs no exception handler of its own *)
let refusal f =
  match f () with
  | () -> "no raise"
  | exception Invalid_argument message -> message
;;

module For_test = struct
  (* The checkpoint as [jax/mamba/train.py] writes it: the tensors named "0" upward, in
     the flat order of [Params_data.to_list]. Three readers cross that seam —
     [Config.of_checkpoint], [Params.load] and [Quantized.Model.of_checkpoint] — thus one
     writer states the naming rule and the gates of both modules read one file.

     The gate makes the file itself, thus it reads no file that git ignores, and [f] holds
     the whole life of the file: it goes when [f] gives and when [f] raises. *)
  let with_checkpoint tensors ~f =
    let path = Stdlib.Filename.temp_file "mgen_mamba_checkpoint" ".safetensors" in
    Exn.protect
      ~f:(fun () ->
        Nx_io.save_safetensors
          path
          (List.mapi tensors ~f:(fun index tensor -> Int.to_string index, Nx_io.P tensor));
        f path)
      ~finally:(fun () -> Stdlib.Sys.remove path)
  ;;

  let refusal ~path f =
    String.substr_replace_all (refusal f) ~pattern:path ~with_:"<file>"
  ;;
end

(* the shape of a test model: small enough to run in a test, and the same structure *)
let test_config = { Config.d = 32; d_in = 64; heads = 2; state = 8; layers = 1 }

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
  [%expect {| 3 windows, 14.8705 nats for each step |}]
;;

(* The one crossing of the JAX-to-OCaml seam, on a file the gate writes itself. What the
   trainer states in a checkpoint is the shape and the values; a reader that took either
   of them wrong would be found on the board and not here, thus the seam runs here. *)
let%expect_test "the checkpoint seam: the readers take what the trainer writes" =
  let params = Params.init test_config ~seed:5 in
  let flat p = Array.concat_map (Array.of_list (Params.to_list p)) ~f:Nx.to_array in
  For_test.with_checkpoint (Params.to_list params) ~f:(fun path ->
    (* every width comes from the shapes in the file, thus a player states none *)
    let read = Config.of_checkpoint path in
    printf
      "the file states d %d, d_in %d, heads %d, state %d, layers %d\n"
      read.d
      read.d_in
      read.heads
      read.state
      read.layers;
    printf
      "%d values, every one the value written: %b\n"
      (Array.length (flat params))
      (Array.equal Float.equal (flat (Params.load read ~path)) (flat params)));
  [%expect
    {|
    the file states d 32, d_in 64, heads 2, state 8, layers 1
    13702 values, every one the value written: true
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
      step 18  c0b2acb8
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

(* ==================================================================== *)
(* The rules both pipelines share *)
(* ==================================================================== *)

(* The quantized twin draws through these rules and never through a copy of them, thus a
   rule that moved here would move the circuit's walk with it. *)
let%expect_test "the fuzz: a pick lands on a class that holds weight" =
  let state = Random.State.make [| 20260819 |] in
  let drawn_weights (_ : int) =
    let classes = Random.State.int_incl state 1 Vocab.classes in
    let raw =
      Array.init classes ~f:(fun (_ : int) -> Random.State.float_range state (-20.) 20.)
    in
    let temperature = Random.State.float_range state 0.5 1.5 in
    let min_p = Random.State.float_range state 0.0 0.9 in
    above_min_p (tempered raw ~temperature) ~min_p
  in
  let edges = [ [| 1.0 |]; [| 1.0; 0.0 |]; [| 0.0; 1.0 |]; [| 0.0; 0.0; 1.0; 0.0 |] ] in
  let cases = edges @ List.map (List.range 0 200) ~f:drawn_weights in
  let uniforms = [ 0.0; 0.5; Float.of_int 0xFFFFFF *. 0x1p-24 ] in
  let fault weights =
    List.find_map uniforms ~f:(fun uniform ->
      let index = pick weights ~uniform in
      if Float.(weights.(index) > 0.0) then None else Some (Array.length weights, index))
  in
  (match List.filter_map cases ~f:fault with
   | [] ->
     printf
       "%d weight rows over %d uniforms: every pick holds weight\n"
       (List.length cases)
       (List.length uniforms)
   | (classes, index) :: (_ : (int * int) list) ->
     printf "a row of %d classes picked %d, which holds no weight\n" classes index);
  [%expect {| 204 weight rows over 3 uniforms: every pick holds weight |}]
;;

let%expect_test "the policy bounds: one message for each" =
  let policy ~temperature ~min_p = refusal (fun () -> check_policy ~temperature ~min_p) in
  printf "temperature 0: %s\n" (policy ~temperature:0.0 ~min_p:elected_min_p);
  printf "min_p 1: %s\n" (policy ~temperature:elected_temperature ~min_p:1.0);
  printf "min_p below 0: %s\n" (policy ~temperature:elected_temperature ~min_p:(-0.1));
  printf
    "the elected policy: %s\n"
    (policy ~temperature:elected_temperature ~min_p:elected_min_p);
  [%expect
    {|
    temperature 0: the temperature is positive
    min_p 1: min_p is 0 up to 1
    min_p below 0: min_p is 0 up to 1
    the elected policy: no raise
    |}]
;;
