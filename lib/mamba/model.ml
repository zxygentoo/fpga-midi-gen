open Core
module Nn_quantized = Mgen_nn.Quantized
module Contract_file = Mgen_nn.Contract_file

module Constants = struct
  (* the shared rules of [Mgen_nn.Quantized] — the exp2, sigmoid and softplus tables stand
     there, one time for every era — and this era's own formats beside them; the logits
     stay wide and no constant names them *)
  include Nn_quantized.Constants

  let v_q = 12
  let s_q = 12
  let alpha_q = 15
  let beta_q = 15

  (* the gate product, in an int32: two Q12 values multiply and nothing truncates them
     before the norm that reads them *)
  let gate_q = 2 * v_q

  (* the Q the Decay op's constant carries. It rides the 25-bit port and [dt] the 18-bit
     one, which is the way round that costs nothing — [dt] is int16 — and it leaves the
     constant three million units of room where the other order would clamp a decay rate
     above 22. *)
  let decay_q = 12

  (* The score rule of the attention head, era four's unchanged: the rings store Q[v_q]
     rows, and the shared rule takes that format by name. *)
  let score_shift ~head_d = score_shift ~row_q:v_q ~head_d
end

module Kind = struct
  type t =
    | Block
    | Attention
    | Feed_forward
  [@@deriving equal, sexp_of]

  (* the tensors a layer of this kind carries into the image, which is the checkpoint's
     count for every kind but the block *)
  let tensors = function
    | Block -> 3
    | Attention -> 4
    | Feed_forward -> 2
  ;;

  (* A plan as one letter for each layer, which is how the design document, the checkpoint
     names and the [--plan] flag of the trainer all spell it. The elected model is
     MMMMMMZF. The trainer knows a fourth letter, A for era four's plain attention; this
     library does not, and the reader refuses a file that holds one. *)
  let spell plan =
    String.of_char_list
      (List.map (Array.to_list plan) ~f:(function
        | Block -> 'M'
        | Attention -> 'Z'
        | Feed_forward -> 'F'))
  ;;
end

module Rom_data = struct
  type 'a t =
    { seats : 'a
    ; phase : 'a
    ; layers : 'a layer array
    }

  and 'a layer =
    | Block of 'a block
    | Attention of 'a attention
    | Feed_forward of 'a feed_forward

  and 'a block =
    { w_in : 'a
    ; conv : 'a
    ; w_out : 'a
    }

  and 'a attention =
    { wq : 'a
    ; wk : 'a
    ; wv : 'a
    ; wo : 'a
    }

  and 'a feed_forward =
    { w1 : 'a
    ; w2 : 'a
    }

  let layer_to_list = function
    | Block { w_in; conv; w_out } -> [ w_in; conv; w_out ]
    | Attention { wq; wk; wv; wo } -> [ wq; wk; wv; wo ]
    | Feed_forward { w1; w2 } -> [ w1; w2 ]
  ;;

  let to_list { seats; phase; layers } =
    seats :: phase :: List.concat_map (Array.to_list layers) ~f:layer_to_list
  ;;

  let layer_of_kind kind items =
    match (kind : Kind.t), items with
    | Block, w_in :: conv :: w_out :: rest -> Block { w_in; conv; w_out }, rest
    | Attention, wq :: wk :: wv :: wo :: rest -> Attention { wq; wk; wv; wo }, rest
    | Feed_forward, w1 :: w2 :: rest -> Feed_forward { w1; w2 }, rest
    | kind, (_ : 'a list) ->
      invalid_argf
        "a %s layer takes %d tensors in the image"
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
      then invalid_argf "%d image tensors stand after the plan" (List.length rest) ();
      { seats; phase; layers = groups }
    | _ -> invalid_arg "the image starts with the two tables"
  ;;
end

type quantized = Nn_quantized.quantized =
  { q : int array
  ; e : int
  }

(* The weights of one block as the machine holds them: three tensors in the ROM, and the
   per-head numbers as constants the ops carry.

   [a_log], [dt_bias] and [d_skip] are [heads] values a layer, and an int8 tensor cannot
   hold them. The bias enters a softplus: a step of one part in 127 of its range moves
   [dt] by more than a small [dt] is, and the decay would follow it. They quantize above
   the seam instead, into the numbers the ops carry — [a * log2(e)] folds into one Q
   constant for each head, exactly as the temperature folds into the temper — thus the run
   time never sees them as tensors. *)
type block =
  { w_in : quantized
  ; conv : quantized
  ; w_out : quantized
  ; decay : Constants.scale array (** [a * log2(e)], one for each head *)
  ; dt_bias : int array (** Q12 *)
  ; d_skip : int array (** Q12 *)
  }

type attention =
  { wq : quantized
  ; wk : quantized
  ; wv : quantized
  ; wo : quantized
  }

type feed_forward =
  { w1 : quantized
  ; w2 : quantized
  }

type layer =
  | Block of block
  | Attention of attention
  | Feed_forward of feed_forward

let kind_of_layer : layer -> Kind.t = function
  | Block (_ : block) -> Block
  | Attention (_ : attention) -> Attention
  | Feed_forward (_ : feed_forward) -> Feed_forward
;;

type t =
  { d : int
  ; d_in : int
  ; heads : int
  ; state : int
  ; taps : int
  ; plan : Kind.t array
  ; span : int
  ; ring : int
  ; seats : quantized
  ; phase : quantized
  ; layers : layer array
  ; temper : Constants.scale
  ; min_weight : int
  }

let blocks t = Array.count t.plan ~f:(Kind.equal Block)
let attentions t = Array.count t.plan ~f:(Kind.equal Attention)

(* The ordinal of each layer among the layers of ITS OWN KIND, and it is what indexes a
   memory. A layer's place in the plan is not its place in a memory: the state RAM and the
   tap ring hold one region for each block, and the key and value rings one for each
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

(* The arithmetic of the circuit is shifts and address concatenations, thus the shape
   obeys their rules. The record is open — a contract file and a drawn model both build
   one — thus a model that no constructor here made can break a rule, and the elaboration
   calls this where a bad shape must fail loudly. *)
let check_shape t =
  let power_of_two name v =
    if not (Int.is_pow2 v) then invalid_argf "%s is %d, not a power of two" name v ()
  in
  (* the rms_norm of the stream divides by [d] and the gated norm by [d_in]: a shift *)
  power_of_two "d" t.d;
  power_of_two "d_in" t.d_in;
  (* the state address is (head, lane, n) concatenated, and the tap address (channel,
     tap), thus every field of them is a power of two *)
  power_of_two "the head count" t.heads;
  power_of_two "the head width" (head t);
  power_of_two "the state width" t.state;
  power_of_two "the tap count" t.taps;
  (* the ring wraps by a mask, and the head splits [d] as it splits [d_in] *)
  power_of_two "the ring" t.ring;
  power_of_two "the attention head width" (head_d t);
  if Array.length t.plan <> Array.length t.layers
  then
    invalid_argf
      "the plan spells %d layers and the model holds %d"
      (Array.length t.plan)
      (Array.length t.layers)
      ();
  Array.iteri t.layers ~f:(fun index l ->
    if not (Kind.equal t.plan.(index) (kind_of_layer l))
    then
      invalid_argf
        "layer %d is a %s and the plan spells a %s"
        index
        (Kind.spell [| kind_of_layer l |])
        (Kind.spell [| t.plan.(index) |])
        ();
    match l with
    | Block l ->
      let rows name row =
        if Array.length row <> t.heads
        then
          invalid_argf
            "the %s of block %d holds %d rows, not the %d heads"
            name
            index
            (Array.length row)
            t.heads
            ()
      in
      rows "decay" l.decay;
      rows "dt bias" l.dt_bias;
      rows "skip" l.d_skip
    | Attention (_ : attention) | Feed_forward (_ : feed_forward) -> ());
  (* the seat rows and the phase row add row for row — the Embed op of the circuit walks
     them as one tensor — thus one exponent covers both *)
  if t.phase.e <> t.seats.e
  then
    invalid_argf
      "the phase table reads exponent %d and the seat tables %d"
      t.phase.e
      t.seats.e
      ();
  let seats = Frame.voices * Vocab.classes * t.d in
  if Array.length t.seats.q <> seats
  then
    invalid_argf
      "the seat tables hold %d weights, not %d"
      (Array.length t.seats.q)
      seats
      ()
;;

(* the element counts of the ROM tensors in the order of the image *)
let rom_sizes t =
  let layer = function
    | Kind.Block -> [ t.d * projection t; channels t * t.taps; t.d_in * t.d ]
    | Attention -> [ 2 * t.d * t.d; 2 * t.d * t.d; t.d * t.d; t.d * t.d ]
    | Feed_forward -> [ t.d * 4 * t.d; 4 * t.d * t.d ]
  in
  [ Frame.voices * Vocab.classes * t.d; Jsb.bar_steps * t.d ]
  @ List.concat_map (Array.to_list t.plan) ~f:layer
;;

let rom_data t =
  { Rom_data.seats = t.seats
  ; phase = t.phase
  ; layers =
      Array.map t.layers ~f:(function
        | Block l -> Rom_data.Block { w_in = l.w_in; conv = l.conv; w_out = l.w_out }
        | Attention l -> Attention { wq = l.wq; wk = l.wk; wv = l.wv; wo = l.wo }
        | Feed_forward l -> Feed_forward { w1 = l.w1; w2 = l.w2 })
  }
;;

let rom_tensors t = Rom_data.to_list (rom_data t)
let rom_bits t = Nn_quantized.rom_bits (rom_tensors t)

(* the base of each tensor inside the ROM: the exclusive prefix scan of the sizes, handed
   back through the one definition of the order *)
let rom_bases t =
  Nn_quantized.bases_of (Array.of_list_map (rom_tensors t) ~f:(fun q -> Array.length q.q))
  |> Array.to_list
  |> Rom_data.of_list ~plan:t.plan
;;

(* [transpose ~rows ~cols v] exchanges the two axes of a flat row-major tensor.

   THE IMAGE STORES W_in TRANSPOSED, and the reason is the address. The circuit reaches a
   weight by CONCATENATING the two walk counters, which costs nothing and is the row-major
   address only when the dimension under the outer counter is a power of two. [d] is one;
   the projection — [2 d_in + 2 N + H], 292 at the baseline — is not. Storing the tensor
   the other way round puts [d] under the outer counter and the concatenation is right
   again. The quantizer above the seam writes it so, thus the two agree.

   The alternative was a constant multiply on the ROM address, and era four's measurement
   says not to: that address cone is the path the whole layer scaling of that block turned
   on. *)
let transpose ~rows ~cols (v : float array) =
  Array.init (rows * cols) ~f:(fun k -> v.((k % rows * cols) + (k / rows)))
;;

(* the tensors the file carries beside its numbered weights *)
let exponents_tensor = "exponents"
let span_tensor = "span"
let ring_tensor = "ring"
let temper_tensor = "temper"
let min_weight_tensor = "min_weight"
let decay_q_value_tensor = "decay_q_value"
let decay_q_tensor = "decay_q"
let dt_bias_tensor = "dt_bias"
let d_skip_tensor = "d_skip"
let beside_the_weights = 9

let of_int8_checkpoint path =
  let file = Contract_file.open_ path in
  let values = Contract_file.values file in
  let only = Contract_file.only file in
  let shape_at at = Contract_file.shape file (Int.to_string at) in
  let count = Contract_file.tensor_count file ~beside:beside_the_weights in
  if count < 3
  then invalid_argf "%s: %d tensors is no quantized state model" path count ();
  let d =
    match shape_at 0 with
    | [| voices; classes; d |] when voices = Frame.voices && classes = Vocab.classes -> d
    | shape ->
      invalid_argf
        "the seat table of %s is %s, and not %d seats of %d classes"
        path
        (Sexp.to_string ([%sexp_of: int array] shape))
        Frame.voices
        Vocab.classes
        ()
  in
  (* THE FIRST TENSOR OF A GROUP NAMES ITS KIND, thus the walk is sequential and it reads
     the kind before it reads the count. A transposed [w_in] is [projection] by [d], and
     the projection is [2 d_in + 2 N + H] — never [d], [2 d] or [4 d] — thus no block head
     can be read as an attention or a feed-forward head. *)
  let kind_at at =
    match shape_at at with
    | [| rows; cols |] when rows = 2 * d && cols = d -> Kind.Attention
    | [| rows; cols |] when rows = d && cols = 4 * d -> Kind.Feed_forward
    | [| rows; cols |] when rows = d && cols = d ->
      invalid_argf
        "%s opens a layer with a square query at tensor %d: era four's attention is not \
         a layer of this model"
        path
        at
        ()
    | [| (_ : int); cols |] when cols = d -> Kind.Block
    | shape ->
      invalid_argf
        "%s opens a layer at tensor %d with %s, which is no image tensor"
        path
        at
        (Sexp.to_string ([%sexp_of: int array] shape))
        ()
  in
  let rec walk at plan =
    if at = count
    then List.rev plan
    else if at > count
    then invalid_argf "%s: %d image tensors do not fill whole layer groups" path count ()
    else (
      let kind = kind_at at in
      walk (at + Kind.tensors kind) ((kind, at) :: plan))
  in
  let plan_at = walk 2 [] in
  let plan = Array.of_list_map plan_at ~f:fst in
  (* EVERY WIDTH OF A BLOCK FALLS OUT OF ITS OWN IMAGE: the projection stands over the
     transposed [w_in], [w_out] gives the inner width, the convolution gives the channels
     and the taps, the channels give the state, and the heads are what the projection has
     left. Nothing states a width twice. *)
  let d_in, state, heads, taps =
    match List.find plan_at ~f:(fun (kind, (_ : int)) -> Kind.equal kind Block) with
    | None -> invalid_argf "%s: a plan of no block is not this model" path ()
    | Some ((_ : Kind.t), at) ->
      let projection = (shape_at at).(0) in
      let channels, taps =
        match shape_at (at + 1) with
        | [| channels; taps |] -> channels, taps
        | shape ->
          invalid_argf
            "the convolution of %s is %s, and not one row for each channel"
            path
            (Sexp.to_string ([%sexp_of: int array] shape))
            ()
      in
      let d_in = (shape_at (at + 2)).(0) in
      let state = (channels - d_in) / 2 in
      d_in, state, projection - (2 * d_in) - (2 * state), taps
  in
  let exponents = values exponents_tensor in
  if Array.length exponents <> count
  then
    invalid_argf "%s: %d exponents for %d tensors" path (Array.length exponents) count ();
  let image =
    List.init count ~f:(fun at -> { q = values (Int.to_string at); e = exponents.(at) })
    |> Rom_data.of_list ~plan
  in
  let temper = Contract_file.scale file temper_tensor in
  (* the four per-head rows stand one row for each BLOCK, indexed by the ordinal of the
     block among the blocks — which is what indexes a memory of the circuit as well *)
  let per_block name =
    let row = values name in
    if Array.length row % heads <> 0
    then
      invalid_argf
        "%s: %s holds %d values, and no whole row of %d heads"
        path
        name
        (Array.length row)
        heads
        ();
    Array.init
      (Array.length row / heads)
      ~f:(fun at -> Array.sub row ~pos:(at * heads) ~len:heads)
  in
  let decay_q_value = per_block decay_q_value_tensor in
  let decay_q = per_block decay_q_tensor in
  let dt_bias = per_block dt_bias_tensor in
  let d_skip = per_block d_skip_tensor in
  let ordinal = ref (-1) in
  let layer : quantized Rom_data.layer -> layer = function
    | Block l ->
      Int.incr ordinal;
      let at = !ordinal in
      Block
        { w_in = l.w_in
        ; conv = l.conv
        ; w_out = l.w_out
        ; decay =
            Array.map2_exn decay_q_value.(at) decay_q.(at) ~f:(fun q_value q ->
              { Constants.q_value; q })
        ; dt_bias = dt_bias.(at)
        ; d_skip = d_skip.(at)
        }
    | Attention l -> Attention { wq = l.wq; wk = l.wk; wv = l.wv; wo = l.wo }
    | Feed_forward l -> Feed_forward { w1 = l.w1; w2 = l.w2 }
  in
  let model =
    { d
    ; d_in
    ; heads
    ; state
    ; taps
    ; plan
    ; span = only span_tensor
    ; ring = only ring_tensor
    ; seats = image.seats
    ; phase = image.phase
    ; layers = Array.map image.layers ~f:layer
    ; temper
    ; min_weight = only min_weight_tensor
    }
  in
  check_shape model;
  List.iter2_exn (rom_tensors model) (rom_sizes model) ~f:(fun tensor size ->
    if Array.length tensor.q <> size
    then
      invalid_argf
        "%s: a tensor holds %d values, not the %d its shape wants"
        path
        (Array.length tensor.q)
        size
        ());
  model
;;

module For_test = struct
  (* the shape numbers of a drawn model, without the tensors that carry them *)
  type shape =
    { d : int
    ; d_in : int
    ; heads : int
    ; state : int
    ; taps : int
    ; plan : Kind.t array
    ; span : int
    ; ring : int
    }

  (* The shape of a test model: small enough to run in a simulation, and the WHOLE PLAN of
     the era. A plan of blocks alone would elaborate no ring and no head, and the faults
     this era's gates found are address faults that only a second layer of a kind can
     show. *)
  let shape =
    { d = 16
    ; d_in = 32
    ; heads = 2
    ; state = 8
    ; taps = 4
    ; plan = [| Block; Attention; Feed_forward |]
    ; span = 4
    ; ring = 8
    }
  ;;

  (* the shape the ear elected, docs/mamba.md: six blocks, the Zamba head and the
     feed-forward, at span 4 and the elected ring. The cost model of a real step is read
     at this shape, and no gate elaborates it. *)
  let elected =
    { d = 64
    ; d_in = 128
    ; heads = 4
    ; state = 16
    ; taps = 4
    ; plan = [| Block; Block; Block; Block; Block; Block; Attention; Feed_forward |]
    ; span = 4
    ; ring = 256
    }
  ;;

  (* The stated exponent of the frozen eras; [Nn_quantized.For_test.drawn_tensor] carries
     the measurement behind it, and the seat and phase tables share it, which is what
     [check_shape] holds.

     The per-head numbers draw STRAIGHT INTO THE FORMS the circuit reads, and no float32
     round trip stands between: a test model is not a checkpoint, thus nothing here has to
     survive a safetensors file. The decay rate is uniform in [1, 16] and lands as
     [a * log2(e)] in Q12, which is the constant the Decay op carries; the step is uniform
     in [0.001, 0.1] through its inverse softplus, as [jax/mamba/train.py] draws it; and
     the skip opens at one, which is 4096 in Q12. A drift report over weights that put
     every decay near one would measure a model this era does not train.

     It is a TEST model: the walk it makes is what the cycle benches record, thus the
     seeds, the scale and these rules may not move. *)
  let drawn_exponent = 10
  let tensor = Nn_quantized.For_test.drawn_tensor ~e:drawn_exponent

  (* a per-head number in Q12, on the port that carries it: the bias joins an int16 sum
     and the skip rides the 18-bit operand port *)
  let fixed_q12 ~bound floats =
    Array.map floats ~f:(fun v ->
      Int.clamp_exn (Float.iround_nearest_exn (Float.ldexp v 12)) ~min:(-bound) ~max:bound)
  ;;

  let drawn (s : shape) ~seed =
    let open Prng in
    let d = s.d in
    let projection = (2 * s.d_in) + (2 * s.state) + s.heads in
    let channels = s.d_in + (2 * s.state) in
    let normal ~count = normals ~count ~scale:0.02 in
    let uniforms ~count = all (List.init count ~f:(fun (_ : int) -> uniform)) in
    let block =
      let* w_in = normal ~count:(d * projection) in
      let* conv = normal ~count:(channels * s.taps) in
      let* steps = uniforms ~count:s.heads in
      let* rates = uniforms ~count:s.heads in
      let+ w_out = normal ~count:(s.d_in * d) in
      Block
        { w_in = tensor (transpose ~rows:d ~cols:projection w_in)
        ; conv = tensor conv
        ; w_out = tensor w_out
        ; decay =
            Array.of_list_map rates ~f:(fun u ->
              { Constants.q_value =
                  Int.clamp_exn
                    (Float.iround_nearest_exn
                       (Float.ldexp
                          ((1.0 +. (u *. 15.0)) /. Float.log 2.0)
                          Constants.decay_q))
                    ~min:0
                    ~max:((1 lsl 24) - 1)
              ; q = Constants.decay_q
              })
            (* the inverse softplus of a step in [0.001, 0.1]: softplus of it is that step *)
        ; dt_bias =
            fixed_q12
              ~bound:32767
              (Array.of_list_map steps ~f:(fun u ->
                 Float.log (Float.exp (0.001 +. (u *. 0.099)) -. 1.0)))
            (* the skip opens at one *)
        ; d_skip = Array.create ~len:s.heads (1 lsl 12)
        }
    in
    (* the four matrices of the head and the two of the feed-forward take the same normal
       as every other matrix: era four drew them so, and one rule covers the whole model *)
    let attention =
      let* wq = normal ~count:(2 * d * d) in
      let* wk = normal ~count:(2 * d * d) in
      let* wv = normal ~count:(d * d) in
      let+ wo = normal ~count:(d * d) in
      Attention { wq = tensor wq; wk = tensor wk; wv = tensor wv; wo = tensor wo }
    in
    let feed_forward =
      let* w1 = normal ~count:(d * 4 * d) in
      let+ w2 = normal ~count:(4 * d * d) in
      Feed_forward { w1 = tensor w1; w2 = tensor w2 }
    in
    let layer = function
      | Kind.Block -> block
      | Attention -> attention
      | Feed_forward -> feed_forward
    in
    let draw =
      let* seats = normal ~count:(Frame.voices * Vocab.classes * d) in
      let* phase = normal ~count:(Jsb.bar_steps * d) in
      let+ layers = all (List.map (Array.to_list s.plan) ~f:layer) in
      { d
      ; d_in = s.d_in
      ; heads = s.heads
      ; state = s.state
      ; taps = s.taps
      ; plan = s.plan
      ; span = s.span
      ; ring = s.ring
      ; seats = tensor seats
      ; phase = tensor phase
      ; layers =
          Array.of_list layers
          (* THE ELECTED POLICY, STATED. The temper is [Constants.temper_at_one] and the
             floor is the elected min-p 0.05 as a share of the peak weight 2^15, which is
             [jax/fixed.py]'s [min_weight_of] and what [test_fixed.py] pins. The elected
             numbers themselves live above the seam now, in [ELECTED_TEMPERATURE] and
             [ELECTED_MIN_P] of [jax/fixed.py]. *)
      ; temper = Constants.temper_at_one
      ; min_weight = 1638
      }
    in
    snd (Prng.run draw (Prng.create_folded ~seed))
  ;;
end
