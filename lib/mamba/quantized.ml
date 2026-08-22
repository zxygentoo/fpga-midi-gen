open Base
module Params_data = Mamba.Params_data

module Constants = struct
  (* the shared rules of [Mgen_nn.Quantized] — the exp2, sigmoid and softplus tables stand
     there, one time for both eras — and this era's own formats beside them; the logits
     stay wide and no constant names them *)
  include Mgen_nn.Quantized.Constants

  let v_q = 12
  let s_q = 12
  let alpha_q = 15
  let beta_q = 15

  (* the gate product, in an int32: two Q12 values multiply and nothing truncates them
     before the norm that reads them *)
  let gate_q = 2 * v_q

  (* [decay_scale ~a] is [a * log2(e)] as the constant the Decay op carries: the run time
     multiplies [dt] by it and the exp2 unit reads the product.

     The constant rides the 25-bit port and [dt] the 18-bit one, which is the way round
     that costs nothing — [dt] is int16 — and it leaves the constant three million units
     of room where the other order would clamp a decay rate above 22. *)
  let decay_q = 12

  let decay_scale ~a =
    { q_value =
        Int.clamp_exn
          (Float.iround_nearest_exn (Float.ldexp (a /. Float.log 2.0) decay_q))
          ~min:0
          ~max:((1 lsl 24) - 1)
    ; q = decay_q
    }
  ;;

  (* The score rule of the attention head, era four's unchanged: the rings store Q[v_q]
     rows, and the shared rule takes that format by name. *)
  let score_shift ~head_d = score_shift ~row_q:v_q ~head_d
end

module Tensor = Mgen_nn.Quantized.Tensor

module Model = struct
  type quantized = Mgen_nn.Quantized.quantized =
    { q : Tensor.t
    ; e : int
    }

  (* The tensors the ROM carries, in the order it carries them: the two tables, then the
     matrices of each layer. It is NOT the order of the checkpoint — a block holds six
     tensors there and three of them never reach the ROM — thus the two orders are two
     structures and neither is implied by the other. An attention layer and a feed-forward
     layer carry every tensor they hold, thus only a block parts the two orders. *)
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

    (* the tensors a layer of this kind carries into the image, which is the checkpoint's
       count for every kind but the block *)
    let tensors = function
      | Mamba.Kind.Block -> 3
      | Attention -> 4
      | Feed_forward -> 2
    ;;

    let to_list { seats; phase; layers } =
      seats :: phase :: List.concat_map (Array.to_list layers) ~f:layer_to_list
    ;;

    let layer_of_kind kind items =
      match (kind : Mamba.Kind.t), items with
      | Block, w_in :: conv :: w_out :: rest -> Block { w_in; conv; w_out }, rest
      | Attention, wq :: wk :: wv :: wo :: rest -> Attention { wq; wk; wv; wo }, rest
      | Feed_forward, w1 :: w2 :: rest -> Feed_forward { w1; w2 }, rest
      | kind, (_ : 'a list) ->
        invalid_arg
          (Printf.sprintf
             "a %s layer takes %d tensors in the image"
             (Sexp.to_string (Mamba.Kind.sexp_of_t kind))
             (tensors kind))
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
        then
          invalid_arg
            (Printf.sprintf "%d image tensors stand after the plan" (List.length rest));
        { seats; phase; layers = groups }
      | _ -> invalid_arg "the image starts with the two tables"
    ;;
  end

  (* The weights of one block as the machine holds them: three tensors in the ROM, and the
     per-head numbers as constants the ops carry.

     [a_log], [dt_bias] and [d_skip] are [heads] values a layer, and an int8 tensor cannot
     hold them. The bias enters a softplus: a step of one part in 127 of its range moves
     [dt] by more than a small [dt] is, and the decay would follow it. They quantize at
     elaboration instead, into the numbers the ops carry — [a * log2(e)] folds into one Q
     constant for each head, exactly as era four folded log2(e) into the temper — thus the
     run time never sees them as tensors. *)
  type block =
    { w_in : quantized
    ; conv : quantized
    ; w_out : quantized
    ; decay : Constants.scale array (** [a * log2(e)], one for each head *)
    ; dt_bias : Tensor.t (** Q12 *)
    ; d_skip : Tensor.t (** Q12 *)
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

  let kind_of_layer : layer -> Mamba.Kind.t = function
    | Block (_ : block) -> Block
    | Attention (_ : attention) -> Attention
    | Feed_forward (_ : feed_forward) -> Feed_forward
  ;;

  type t =
    { config : Mamba.Config.t
    ; seats : quantized
    ; phase : quantized
    ; layers : layer array
    ; temper : Constants.scale
    ; min_weight : int
    }

  (* The arithmetic of the circuit is shifts and address concatenations, thus the shape
     obeys their rules. The reference holds this check because the reference states the
     rules; the circuit calls it at elaboration, where a bad shape must fail loudly. *)
  let check_shape t =
    let { Mamba.Config.d; d_in; heads; state; taps; plan; span = (_ : int); ring } =
      t.config
    in
    (* the rms_norm of the stream divides by [d] and the gated norm by [d_in]: a shift *)
    assert (Int.is_pow2 d);
    assert (Int.is_pow2 d_in);
    (* the state address is (head, lane, n) concatenated, and the tap address (channel,
       tap), thus every field of them is a power of two *)
    assert (Int.is_pow2 heads);
    assert (Int.is_pow2 (Mamba.Config.head t.config));
    assert (Int.is_pow2 state);
    assert (Int.is_pow2 taps);
    (* the ring wraps by a mask, and the head splits [d] as it splits [d_in] *)
    assert (Int.is_pow2 ring);
    assert (Int.is_pow2 (Mamba.Config.head_d t.config));
    assert (Array.length plan = Array.length t.layers);
    Array.iteri t.layers ~f:(fun index l ->
      assert (Mamba.Kind.equal plan.(index) (kind_of_layer l));
      match l with
      | Block l ->
        assert (Array.length l.decay = heads);
        assert (Array.length l.dt_bias = heads);
        assert (Array.length l.d_skip = heads)
      | Attention (_ : attention) | Feed_forward (_ : feed_forward) -> ());
    (* the seat rows and the phase row add row for row — [Engine.embed] adds them, and the
       Embed op of the circuit walks them as one tensor — thus one exponent covers both *)
    assert (t.phase.e = t.seats.e);
    assert (Array.length t.seats.q = Frame.voices * Vocab.classes * d)
  ;;

  (* the element counts of the ROM tensors in the order of the image *)
  let rom_sizes (config : Mamba.Config.t) =
    let d = config.d in
    let layer = function
      | Mamba.Kind.Block ->
        [ d * Mamba.Config.projection config
        ; Mamba.Config.channels config * config.taps
        ; config.d_in * d
        ]
      | Attention -> [ 2 * d * d; 2 * d * d; d * d; d * d ]
      | Feed_forward -> [ d * 4 * d; 4 * d * d ]
    in
    [ Frame.voices * Vocab.classes * d; Jsb.bar_steps * d ]
    @ List.concat_map (Array.to_list config.plan) ~f:layer
  ;;

  let rom_tensors t =
    Rom_data.to_list
      { Rom_data.seats = t.seats
      ; phase = t.phase
      ; layers =
          Array.map t.layers ~f:(function
            | Block l -> Rom_data.Block { w_in = l.w_in; conv = l.conv; w_out = l.w_out }
            | Attention l -> Attention { wq = l.wq; wk = l.wk; wv = l.wv; wo = l.wo }
            | Feed_forward l -> Feed_forward { w1 = l.w1; w2 = l.w2 })
      }
  ;;

  (* the running sums of the sizes, handed back through the one definition of the order *)
  let rom_bases t =
    List.folding_map (rom_sizes t.config) ~init:0 ~f:(fun start size ->
      start + size, start)
    |> Rom_data.of_list ~plan:t.config.Mamba.Config.plan
  ;;

  let rom_bits t = Mgen_nn.Quantized.rom_bits (rom_tensors t)
  let policy = Mgen_nn.Quantized.policy
  let max_abs = Mgen_nn.Quantized.max_abs
  let max_exponent = Mgen_nn.Quantized.max_exponent
  let quantize = Mgen_nn.Quantized.quantize

  (* [transpose ~rows ~cols v] exchanges the two axes of a flat row-major tensor.

     THE IMAGE STORES W_in TRANSPOSED, and the reason is the address. The circuit reaches
     a weight by CONCATENATING the two walk counters, which costs nothing and is the
     row-major address only when the dimension under the outer counter is a power of two.
     [d] is one; the projection width — [2 d_in + 2 N + H], 292 at the baseline — is not.
     Storing the tensor the other way round puts [d] under the outer counter and the
     concatenation is right again. The engine reads it the same way, thus the two agree.

     The alternative was a constant multiply on the ROM address, and era four's
     measurement says not to: that address cone is the path the whole layer scaling of
     that block turned on. *)
  let transpose ~rows ~cols (v : Tensor.floats) =
    Array.init (rows * cols) ~f:(fun k -> v.((k % rows * cols) + (k / rows)))
  ;;

  (* a per-head number in Q12, on the port that carries it: the bias joins an int16 sum
     and the skip rides the 18-bit operand port *)
  let fixed_q12 ~bound floats =
    Array.map floats ~f:(fun v ->
      Int.clamp_exn (Float.iround_nearest_exn (Float.ldexp v 12)) ~min:(-bound) ~max:bound)
  ;;

  let of_floats (config : Mamba.Config.t) ~temperature ~min_p tensors =
    let temper, min_weight = policy ~temperature ~min_p in
    let { Params_data.seats; phase; layers } : Tensor.floats Params_data.t =
      Params_data.of_list ~plan:config.plan tensors
    in
    let e = max_exponent (Float.max (max_abs seats) (max_abs phase)) in
    let layer : Tensor.floats Params_data.layer -> layer = function
      | Block l ->
        Block
          { w_in =
              quantize
                (transpose ~rows:config.d ~cols:(Mamba.Config.projection config) l.w_in)
          ; conv = quantize l.conv
          ; w_out = quantize l.w_out
          ; decay =
              Array.map l.a_log ~f:(fun a_log ->
                Constants.decay_scale ~a:(Float.exp a_log))
          ; dt_bias = fixed_q12 ~bound:32767 l.dt_bias
          ; d_skip = fixed_q12 ~bound:131071 l.d_skip
          }
      | Attention l ->
        Attention
          { wq = quantize l.wq
          ; wk = quantize l.wk
          ; wv = quantize l.wv
          ; wo = quantize l.wo
          }
      | Feed_forward l -> Feed_forward { w1 = quantize l.w1; w2 = quantize l.w2 }
    in
    { config
    ; temper
    ; min_weight
    ; seats = quantize ~e seats
    ; phase = quantize ~e phase
    ; layers = Array.map layers ~f:layer
    }
  ;;

  let of_checkpoint
    ?(temperature = Mamba.elected_temperature)
    ?(min_p = Mamba.elected_min_p)
    (config : Mamba.Config.t)
    path
    =
    let archive = Nx_io.load_safetensors path in
    let tensor name =
      match Stdlib.Hashtbl.find_opt archive name with
      | None -> invalid_arg (Printf.sprintf "%s holds no tensor named %s" path name)
      | Some packed -> Nx.to_array (Nx_io.to_typed Nx.float32 packed)
    in
    let tensors =
      List.mapi (Mamba.Params.shapes config) ~f:(fun index shape ->
        let size = Mamba.numel shape in
        let values = tensor (Int.to_string index) in
        if Array.length values <> size
        then
          invalid_arg
            (Printf.sprintf
               "%s tensor %d holds %d values, not %d"
               path
               index
               (Array.length values)
               size);
        values)
    in
    of_floats config ~temperature ~min_p tensors
  ;;

  module For_test = struct
    (* The shape of a test model: small enough to run in a simulation, and the whole plan
       of the era. A plan of blocks alone would elaborate no ring and no head, and the
       faults this era's gates found are address faults that only a second layer of a kind
       can show. *)
    let config =
      { Mamba.Config.d = 16
      ; d_in = 32
      ; heads = 2
      ; state = 8
      ; taps = 4
      ; plan = [| Block; Attention; Feed_forward |]
      ; span = Mamba.elected_span
      ; ring = 8
      }
    ;;

    (* a model of drawn weights, quantized: a test reads no checkpoint, thus it reads no
       file that git ignores. The draw is [Mamba.Params.init], thus the decay rates and
       the skip come out where the trainer puts them and a drift report over this model
       measures the arithmetic this era really runs. *)
    let init config ~seed =
      of_floats
        config
        ~temperature:Mamba.elected_temperature
        ~min_p:Mamba.elected_min_p
        (List.map (Mamba.Params.to_list (Mamba.Params.init config ~seed)) ~f:Nx.to_array)
    ;;
  end
end

(* The clamps of the walk, and the chances each one had. The formats of this era are
   chosen with margin and not metered on a trained checkpoint, thus a clamp that fires is
   the finding that says which one is wrong — and where era four could let a hot signal
   die with its window, an error in the state carries forward. *)
module Clamps = struct
  type t =
    { dt : int
    ; dt_seen : int
    ; beta : int
    ; beta_seen : int
    ; state : int
    ; state_seen : int
    }

  let none = { dt = 0; dt_seen = 0; beta = 0; beta_seen = 0; state = 0; state_seen = 0 }
  let share hit seen = Float.of_int hit /. Float.of_int (Int.max 1 seen)
end

module Engine = struct
  (* Everything in [t] is frozen: an update copies the arrays it touches. The state and
     the taps are the memory of the walk and the only things that survive a step. *)
  type t =
    { model : Model.t
    ; state : Tensor.t
    (** [blocks * d_in * state] values, Q12 — the recurrence, and the one memory this
        machine modifies rather than rewrites *)
    ; taps : Tensor.t (** [blocks * channels * taps] values, Q12 *)
    ; kc : Tensor.t array
    (** the K ring: [attentions * ring] rows of [d], Q12 in a coarse byte *)
    ; vc : Tensor.t array (** the V ring, the same shape *)
    ; h : Tensor.t (** the residual stream after the last forwarded step, Q16 *)
    ; position : int (** one step of the recurrence for each step of music *)
    ; prng : Prng.state
    ; clamps : Clamps.t
    }

  let config t = t.model.Model.config

  type draw =
    { seat : int
    ; logits : Tensor.t
    ; uniform : float
    ; drawn : int
    }

  type step =
    { frame : int
    ; draws : draw list
    }

  let classes = Vocab.classes
  let voices = Frame.voices

  (* the silent lead-in of the boot, in steps: one bar, as the float sampler plays it *)
  let lead = Jsb.bar_steps

  (* value * 2^-from as value * 2^-target; the arithmetic shift floors *)
  let rescale ~from ~target v =
    if target >= from then v lsl (target - from) else v asr (from - target)
  ;;

  let clamp16 = Mgen_nn.Quantized.clamp16
  let clamps16 = Mgen_nn.Quantized.clamps16
  let sum = Mgen_nn.Quantized.sum
  let max_over = Mgen_nn.Quantized.max_over
  let isqrt = Mgen_nn.Quantized.isqrt
  let exp2_of_magnitude = Mgen_nn.Quantized.exp2_of_magnitude
  let sigmoid_q = Mgen_nn.Quantized.sigmoid_q
  let silu = Mgen_nn.Quantized.silu
  let softplus = Mgen_nn.Quantized.softplus

  (* rms_norm over [width] elements of a Q[from] vector, giving Q[y_q].

     The sum squares a Q[y_q] copy — one DSP-sized product — then one isqrt, and one
     division for each element. The division is toward zero, as every division of the
     circuit. The stream enters at Q16 and the gate of the block at Q12, thus the shift of
     the numerator is the one thing that moves between the two callers. *)
  let rms_norm ~from ~width (v : Tensor.t) : Tensor.t =
    let s =
      Array.fold v ~init:0 ~f:(fun acc x ->
        let q = rescale ~from ~target:Constants.y_q x in
        acc + (q * q))
    in
    let m = (s asr Int.floor_log2 width) + Constants.eps_q in
    let g = isqrt m in
    let numerator = 1 lsl ((2 * Constants.y_q) - from) in
    Array.map v ~f:(fun x -> clamp16 (x * numerator / g))
  ;;

  (* The row of one seat inside the seat tensor, which holds the four tables in one, seat
     0 first: the circuit reaches it with a shift and an add from the base of the tensor. *)
  let seat_row ~d ~seat ~index = ((seat * classes) + index) * d

  (* the embedding: the four seat rows and the phase row add in the shared exponent, then
     shift to Q16 *)
  let embed t ~frame ~phase : Tensor.t =
    let d = (config t).Mamba.Config.d in
    let drawn = Array.of_list (Vocab.classes_of_frame frame) in
    Array.init d ~f:(fun i ->
      let v =
        Array.foldi
          drawn
          ~init:t.model.phase.q.((phase * d) + i)
          ~f:(fun seat acc index -> acc + t.model.seats.q.(seat_row ~d ~seat ~index + i))
      in
      rescale ~from:t.model.seats.e ~target:Constants.h_q v)
  ;;

  (* One matvec column: [inner] terms of a Q[from] vector against a row of the weight.

     [outer_major] states which axis the tensor's rows are, and the circuit reads the same
     order: it is true for W_in, which the image stores transposed, and for the seat
     readout, which the checkpoint already stores that way. *)
  let matvec (y : Tensor.t) (w : Model.quantized) ~outer_major ~inner ~outer ~from ~target
    : Tensor.t
    =
    let at i o = if outer_major then (o * inner) + i else (i * outer) + o in
    Array.init outer ~f:(fun o ->
      clamp16
        (rescale ~from:(from + w.e) ~target (sum inner (fun i -> y.(i) * w.q.(at i o)))))
  ;;

  (* a residual join: [values] times [w] lands on the stream [h] — the stream after; the
     exponent of [w] folds into the shift with [from], the format of [values] *)
  let join t (h : Tensor.t) (w : Model.quantized) ~(values : Tensor.t) ~len ~from
    : Tensor.t
    =
    let d = (config t).Mamba.Config.d in
    Array.mapi h ~f:(fun o above ->
      let acc = sum len (fun i -> values.(i) * w.q.((i * d) + o)) in
      above + rescale ~from:(from + w.e) ~target:Constants.h_q acc)
  ;;

  (* The tap ring of one layer: a ring of [width] taps for each channel, and the position
     names the slot. Tap k reads the step k back, and it reads ZERO while the walk has not
     run k steps — thus the origin needs no clearing walk and the rule is a mux, as era
     four's fill count was. *)
  let tap_slot ~width ~base ~channel ~at = base + (channel * width) + (at land (width - 1))

  let tap_at ~width ~taps ~base ~position ~channel ~k =
    if position < k then 0 else taps.(tap_slot ~width ~base ~channel ~at:(position - k))
  ;;

  (* One block of the trunk. It gives the stream after the residual join and the clamps it
     met on the way, and it writes the state and the taps of its own region in place — the
     two arrays are copies the caller made for this step. [index] is the BLOCK ordinal and
     not the place in the plan: [Mamba.Config.ordinals] states the difference. *)
  let block t ~index (lay : Model.block) (h : Tensor.t) ~state ~taps ~clamps =
    (* the tap COUNT takes another name here: [taps] is the ring this step writes *)
    let { Mamba.Config.d; d_in; heads; state = n; taps = width; _ } = config t in
    let head = Mamba.Config.head (config t) in
    let channels = Mamba.Config.channels (config t) in
    let position = t.position in
    let y = rms_norm ~from:Constants.h_q ~width:d h in
    let zxbcdt =
      matvec
        y
        lay.w_in
        ~outer_major:true
        ~inner:d
        ~outer:(Mamba.Config.projection (config t))
        ~from:Constants.y_q
        ~target:Constants.v_q
    in
    (* the convolution: the step's input enters the ring, then a row of [taps] terms for
       each channel, then the SiLU chain over the sums *)
    let tap_base = index * channels * width in
    Array.iteri (Array.sub zxbcdt ~pos:d_in ~len:channels) ~f:(fun c v ->
      taps.(tap_slot ~width ~base:tap_base ~channel:c ~at:position) <- v);
    let xbc =
      Array.init channels ~f:(fun c ->
        let acc =
          sum width (fun k ->
            tap_at ~width ~taps ~base:tap_base ~position ~channel:c ~k
            * lay.conv.q.((c * width) + k))
        in
        silu
          (clamp16 (rescale ~from:(Constants.v_q + lay.conv.e) ~target:Constants.v_q acc)))
    in
    let x = Array.sub xbc ~pos:0 ~len:d_in in
    let b = Array.sub xbc ~pos:d_in ~len:n in
    let c = Array.sub xbc ~pos:(d_in + n) ~len:n in
    (* the decay of each head: softplus of the biased draw, then one exp2 *)
    let raw = Array.sub zxbcdt ~pos:(d_in + channels) ~len:heads in
    let dt = Array.mapi raw ~f:(fun hd v -> softplus (v + lay.dt_bias.(hd))) in
    let clamps =
      Array.fold dt ~init:clamps ~f:(fun (acc : Clamps.t) v ->
        { acc with
          dt = (acc.dt + if v = 32767 || v = -32768 then 1 else 0)
        ; dt_seen = acc.dt_seen + 1
        })
    in
    let alpha =
      Array.mapi dt ~f:(fun hd v -> exp2_of_magnitude (Constants.apply lay.decay.(hd) v))
    in
    (* the state update and the readout, head by head. [beta] is the inject operand of the
       head: [state] products of [dt] against B, written before the walk. *)
    let base = index * d_in * n in
    let clamps = ref clamps in
    let read = Array.create ~len:d_in 0 in
    for hd = 0 to heads - 1 do
      let beta =
        Array.init n ~f:(fun j ->
          let v =
            (dt.(hd) * b.(j)) asr (Constants.v_q + Constants.v_q - Constants.beta_q)
          in
          clamps
          := { !clamps with
               beta = (!clamps.beta + if clamps16 v then 1 else 0)
             ; beta_seen = !clamps.beta_seen + 1
             };
          clamp16 v)
      in
      for p = 0 to head - 1 do
        let lane = (hd * head) + p in
        let row = base + (lane * n) in
        for j = 0 to n - 1 do
          let v =
            ((alpha.(hd) * state.(row + j)) + (x.(lane) * beta.(j))) asr Constants.alpha_q
          in
          clamps
          := { !clamps with
               state = (!clamps.state + if clamps16 v then 1 else 0)
             ; state_seen = !clamps.state_seen + 1
             };
          state.(row + j) <- clamp16 v
        done;
        (* the readout reads the state the update just wrote, and the skip folds into the
           row as its last term *)
        read.(lane)
        <- clamp16
             ((sum n (fun j -> state.(row + j) * c.(j)) + (x.(lane) * lay.d_skip.(hd)))
              asr Constants.s_q)
      done
    done;
    (* The gated norm of Mamba-2, then the join.

       The gate product stays WIDE. Both operands are Q12 values well under one — the
       readout of a small state and the SiLU of a gate — thus a truncation back to Q12
       here would keep about five bits of a product that holds seventeen, and it would
       throw them away immediately before the one operation that would have used them: the
       norm divides by the size of the vector and does not care what scale it arrives in.
       Measured on the drawn weights of the gate shape, the truncation cost 0.10 of the
       cosine on its own. This is the format the era-four stream already had — Q16 in an
       int32, normalized down to Q12 — one axis further in. *)
    let gated = Array.mapi read ~f:(fun i v -> v * silu zxbcdt.(i)) in
    let g = rms_norm ~from:Constants.gate_q ~width:d_in gated in
    join t h lay.w_out ~values:g ~len:d_in ~from:Constants.v_q, !clamps
  ;;

  (* The ring keeps the top byte of a Q12 row: the circuit stores eight bits and restores
     eight zero low bits at the read, thus the granularity is 2^-4 and the format stays
     Q12. The query does not pass here — only the stored rows coarsen. This is era four's
     ring and not the state: the state carries an error forward and keeps its whole int16,
     where a ring error dies with its window. *)
  (* [asr] and [lsl] associate to the right; the parentheses are the expression *)
  let coarse_to_ring (row : Tensor.t) : Tensor.t =
    Array.map row ~f:(fun v -> (v asr 8) lsl 8)
  ;;

  (* Attention over the newest [n] steps of the ring: the merged context of the query [q],
     head by head. Age [a] reads slot [(cur - a) & (ring - 1)], thus the ALiBi distance IS
     the age and the causal wall is the walk. *)
  let attend t (kc : Tensor.t array) (vc : Tensor.t array) ~ring ~cur ~n (q : Tensor.t)
    : Tensor.t
    =
    let { Mamba.Config.heads; span; ring = slots; _ } = config t in
    let head_d = Mamba.Config.head_d (config t) in
    (* the ring row that age [a] reads. The rows depend on neither the head nor the lane,
       thus they are named once: the weighted sum below would otherwise walk the ring for
       every (head, lane, age). *)
    let row memory a = memory.((ring * slots) + ((cur - a) land (slots - 1))) in
    let k_rows = Array.init n ~f:(row kc) in
    let v_rows = Array.init n ~f:(row vc) in
    let head_context head =
      let hb = head * head_d in
      let slope_exponent = Constants.slope_exponent ~span ~heads ~head in
      let score_shift = Constants.score_shift ~head_d in
      let score a =
        let k = k_rows.(a) in
        (sum head_d (fun j -> q.(hb + j) * k.(hb + j)) asr score_shift)
        - (a lsl (Constants.y_q - slope_exponent))
      in
      let scores = Array.init n ~f:score in
      let peak = max_over n (fun a -> scores.(a)) in
      (* The exp2 weight of each age, Q15: the peak weighs 2^15.

         THE NEGATION STANDS OUTSIDE THE SCALE, as it stands outside the temper of the
         draw. The circuit scales the score's distance BELOW the peak — a nonpositive
         number — and negates the shifted product, thus a scale that did not divide
         exactly would round the other way if the reference negated first. *)
      let age_weight =
        Array.init n ~f:(fun a ->
          exp2_of_magnitude (-Constants.apply Constants.log2e (scores.(a) - peak)))
      in
      let den = sum n (fun a -> age_weight.(a)) in
      Array.init head_d ~f:(fun j ->
        clamp16 (sum n (fun a -> age_weight.(a) * v_rows.(a).(hb + j)) / den))
    in
    (* head [k] gives the lanes [k * head_d] up to the next head; the heads are
       independent, thus the order of [List.init] moves nothing *)
    List.init heads ~f:head_context |> Array.concat
  ;;

  (* One attention layer, and it is era four's with one addition: the query and the key
     read the JOINED vector — the normed stream beside the normed embedding — thus their
     walk is [2 d] terms where the value's is [d]. [ring] is the attention ordinal. *)
  let attention t ~ring (lay : Model.attention) (h : Tensor.t) ~e ~kc ~vc =
    let { Mamba.Config.d; ring = slots; _ } = config t in
    let cur = t.position land (slots - 1) in
    let n = Int.min (t.position + 1) slots in
    let y = rms_norm ~from:Constants.h_q ~width:d h in
    let joined = Array.append y e in
    let project w ~inner source =
      matvec
        source
        w
        ~outer_major:false
        ~inner
        ~outer:d
        ~from:Constants.y_q
        ~target:Constants.v_q
    in
    kc.((ring * slots) + cur) <- coarse_to_ring (project lay.wk ~inner:(2 * d) joined);
    vc.((ring * slots) + cur) <- coarse_to_ring (project lay.wv ~inner:d y);
    let context = attend t kc vc ~ring ~cur ~n (project lay.wq ~inner:(2 * d) joined) in
    join t h lay.wo ~values:context ~len:d ~from:Constants.v_q
  ;;

  (* Era four's feed-forward as a layer of its own: one matvec and a ReLU, Q10, then the
     join.

     The ReLU stands after the clamp of the matvec and the circuit takes it before, which
     is the same integer: a value the clamp raised was negative and the ReLU makes it zero
     either way, and a value it lowered was above the ceiling and stays there. *)
  let feed_forward t (lay : Model.feed_forward) (h : Tensor.t) =
    let d = (config t).Mamba.Config.d in
    let dff = 4 * d in
    let y = rms_norm ~from:Constants.h_q ~width:d h in
    let hidden =
      matvec
        y
        lay.w1
        ~outer_major:false
        ~inner:d
        ~outer:dff
        ~from:Constants.y_q
        ~target:Constants.hid_q
      |> Array.map ~f:(Int.max 0)
    in
    join t h lay.w2 ~values:hidden ~len:dff ~from:Constants.hid_q
  ;;

  (* one layer's whole step, whichever kind it is. [ordinal] is the layer's place among
     the layers of its own kind, which is what indexes the memory it owns. *)
  let branch t ~ordinal (lay : Model.layer) (h : Tensor.t) ~e ~state ~taps ~kc ~vc ~clamps
    =
    match lay with
    | Model.Block lay -> block t ~index:ordinal lay h ~state ~taps ~clamps
    | Attention lay -> attention t ~ring:ordinal lay h ~e ~kc ~vc, clamps
    | Feed_forward lay -> feed_forward t lay h, clamps
  ;;

  (* One step of the recurrence: the engine after it.

     The memories are copied once and then written in place through the layer walk. This
     is the local mutation the style rule allows for a mutable structure used as one: the
     state RAM of the circuit is written in place as well, and a fold that rebuilt a 12
     288-element array for each of six layers would model something the machine does not
     do. Nothing outside this function sees a half-written memory. *)
  let forward t ~frame ~phase =
    let d = (config t).Mamba.Config.d in
    let state = Array.copy t.state in
    let taps = Array.copy t.taps in
    let kc = Array.copy t.kc in
    let vc = Array.copy t.vc in
    let ordinals = Mamba.Config.ordinals (config t) in
    let origin = embed t ~frame ~phase in
    (* the embedding the attention reads, normed once: it is the input of layer 0 and it
       does not change as the stream is written *)
    let e = rms_norm ~from:Constants.h_q ~width:d origin in
    let h, clamps =
      Array.foldi t.model.layers ~init:(origin, t.clamps) ~f:(fun index (h, clamps) lay ->
        branch t ~ordinal:ordinals.(index) lay h ~e ~state ~taps ~kc ~vc ~clamps)
    in
    { t with h; state; taps; kc; vc; position = t.position + 1; clamps }
  ;;

  (* The origin of a walk: a zero state, an empty tap ring, an empty key and value ring,
     and no residual. The lead-in is not here — it is the first steps of the walk itself,
     thus [next_step] states it and a caller that counts steps counts the steps the float
     sampler counts. *)
  let init (model : Model.t) ~seed =
    Model.check_shape model;
    let { Mamba.Config.d; d_in; state = n; ring; _ } = model.config in
    let blocks = Mamba.Config.blocks model.config in
    let rows = Mamba.Config.attentions model.config * ring in
    let ring_of (_ : int) = Array.create ~len:d 0 in
    { model
    ; state = Array.create ~len:(blocks * d_in * n) 0
    ; taps =
        Array.create
          ~len:(blocks * Mamba.Config.channels model.config * model.config.taps)
          0
    ; kc = Array.init rows ~f:ring_of
    ; vc = Array.init rows ~f:ring_of
    ; h = Array.create ~len:d 0
    ; position = 0
    ; prng = Prng.create ~seed
    ; clamps = Clamps.none
    }
  ;;

  (* the tied head of one seat: rms_norm of the stream the chain has written so far, then
     that seat's table read backward; Q12 logits over the classes *)
  let seat_logits t (stream : Tensor.t) ~seat =
    let d = (config t).Mamba.Config.d in
    let y = rms_norm ~from:Constants.h_q ~width:d stream in
    Array.init classes ~f:(fun index ->
      sum d (fun i -> y.(i) * t.model.seats.q.(seat_row ~d ~seat ~index + i))
      asr t.model.seats.e)
  ;;

  (* what the chain adds after a seat draws: the drawn row, in the format of the stream *)
  let add_row t (stream : Tensor.t) ~seat ~index =
    let d = (config t).Mamba.Config.d in
    let base = seat_row ~d ~seat ~index in
    Array.mapi stream ~f:(fun i above ->
      above
      + rescale ~from:t.model.seats.e ~target:Constants.h_q t.model.seats.q.(base + i))
  ;;

  (* The draw over the logits of one seat, through the shared integer pick. No mask stands
     before it, because no frame is illegal, and the pick needs no fallback:
     [Mgen_nn.Quantized.draw] states why. *)
  let draw_of_logits t ~logits =
    let peak = max_over classes (fun c -> logits.(c)) in
    (* The tempered weight of one class: exp2 of the magnitude, and refused under min-p.
       The negation stands AFTER the scale, as the circuit negates — a reference that
       negated first would round the other way whenever the scale does not divide exactly,
       and the stream gate found exactly that. *)
    let weight c =
      let e = exp2_of_magnitude (-Constants.apply t.model.temper (logits.(c) - peak)) in
      if e >= t.model.min_weight then e else 0
    in
    let weights = Array.init classes ~f:weight in
    let prng, uniform, index = Mgen_nn.Quantized.draw ~weights t.prng in
    { t with prng }, uniform, index
  ;;

  (* One frame, drawn in a chain from the soprano down: each seat reads the stream that
     the seats above it have written. The draws come back in the order they happened. *)
  let chain t =
    let rec walk t stream seat drawn =
      if seat < 0
      then t, List.rev drawn
      else (
        let logits = seat_logits t stream ~seat in
        let t, uniform, index = draw_of_logits t ~logits in
        let stream = if seat = 0 then stream else add_row t stream ~seat ~index in
        walk t stream (seat - 1) ({ seat; logits; uniform; drawn = index } :: drawn))
    in
    walk t t.h (voices - 1) []
  ;;

  let next_step t =
    let phase = t.position % Jsb.bar_steps in
    (* The boot of docs/mamba.md: a lead-in of silence, one bar of it, drawing nothing and
       taking no number from the generator. The state opens at zero, which is where a
       training window opens, thus the model opens the music itself after it. *)
    let t, step =
      if t.position < lead
      then t, { frame = Frame.silent; draws = [] }
      else (
        let t, draws = chain t in
        ( t
        , { frame =
              Vocab.frame_of_classes (List.rev_map draws ~f:(fun (d : draw) -> d.drawn))
          ; draws
          } ))
    in
    forward t ~frame:step.frame ~phase, step
  ;;

  let clamps t = t.clamps

  (* the scalar rules the L0 circuit units must reproduce; their gate tests read them here
     rather than restate them *)
  module For_test = struct
    (* the residual stream, and the stream after each layer of one step. When a frame gate
       fails, the frame says only THAT the two parted; these say where. *)
    let stream t = t.h

    let layer_streams t ~frame ~phase =
      let d = (config t).Mamba.Config.d in
      let state = Array.copy t.state in
      let taps = Array.copy t.taps in
      let kc = Array.copy t.kc in
      let vc = Array.copy t.vc in
      let ordinals = Mamba.Config.ordinals (config t) in
      let origin = embed t ~frame ~phase in
      let e = rms_norm ~from:Constants.h_q ~width:d origin in
      let (_ : Tensor.t * Clamps.t), rows =
        List.fold_map
          (List.init (Array.length t.model.layers) ~f:Fn.id)
          ~init:(origin, t.clamps)
          ~f:(fun (h, clamps) index ->
            let next, clamps =
              branch
                t
                ~ordinal:ordinals.(index)
                t.model.layers.(index)
                h
                ~e
                ~state
                ~taps
                ~kc
                ~vc
                ~clamps
            in
            (next, clamps), next)
      in
      (* the embed stands at the head, thus the list is one entry for each write of the
         whole stream that the circuit makes *)
      origin :: rows
    ;;
  end
end

module Drift = struct
  type stats =
    { steps : int
    ; draws : int
    ; same_peak : int
    ; same_draw : int
    ; mean_cosine : float
    ; dt_clamped : float
    ; beta_clamped : float
    ; state_clamped : float
    }

  (* One weights source and one policy: the walk quantizes the float tensors itself, under
     the draw of the era, thus the pairing cannot slip.

     The float model walks BESIDE the integer one and carries its own memory. Era four had
     to re-run a whole window at every step, which made a long comparison quadratic; here
     both models take one step for one step, thus the walk can run past many decay
     lifetimes — which it must, because a state error carries forward where a ring error
     died with its window.

     The float pass is teacher-forced on the quantized walk: it reads the frames the
     engine drew and conditions each seat on the classes the engine chose, thus what the
     report measures is the quantization and never a walk that parted for another reason. *)
  let walk (config : Mamba.Config.t) params ~steps ~seed =
    let model =
      Model.of_floats
        config
        ~temperature:Mamba.elected_temperature
        ~min_p:Mamba.elected_min_p
        (* [Nx.to_array] is already row-major and flat: what the quantization reads *)
        (List.map (Mamba.Params.to_list params) ~f:Nx.to_array)
    in
    let engine = ref (Engine.init model ~seed) in
    let memory = ref (Mamba.Memory.origin config ~batch:1) in
    let stream = ref (Array.create ~len:config.d 0.0) in
    let drawn_classes draws =
      let seats = Array.create ~len:Frame.voices 0 in
      List.iter draws ~f:(fun (d : Engine.draw) -> seats.(d.seat) <- d.drawn);
      seats
    in
    let draws = ref 0 in
    let same_peak = ref 0 in
    let same_draw = ref 0 in
    let cosine_sum = ref 0.0 in
    for step = 0 to steps - 1 do
      let next, { Engine.frame; draws = chain } = Engine.next_step !engine in
      if not (List.is_empty chain)
      then (
        let floated =
          Mamba.logits config params ~stream:!stream ~drawn:(drawn_classes chain)
        in
        List.iter chain ~f:(fun (d : Engine.draw) ->
          let float_row = floated.(d.seat) in
          if Tensor.same_peak d.logits float_row then Int.incr same_peak;
          cosine_sum := !cosine_sum +. Tensor.cosine d.logits float_row;
          let float_class =
            Mamba.draw_class
              float_row
              ~temperature:Mamba.elected_temperature
              ~min_p:Mamba.elected_min_p
              ~uniform:d.uniform
          in
          if float_class = d.drawn then Int.incr same_draw;
          Int.incr draws));
      engine := next;
      let after, row =
        Mamba.forward config params !memory ~frame ~phase:(step % Jsb.bar_steps)
      in
      memory := after;
      stream := row
    done;
    let { Clamps.dt; dt_seen; beta; beta_seen; state; state_seen } =
      Engine.clamps !engine
    in
    { steps
    ; draws = !draws
    ; same_peak = !same_peak
    ; same_draw = !same_draw
    ; mean_cosine = !cosine_sum /. Float.of_int (max 1 !draws)
    ; dt_clamped = Clamps.share dt dt_seen
    ; beta_clamped = Clamps.share beta beta_seen
    ; state_clamped = Clamps.share state state_seen
    }
  ;;
end

(* the model the expect tests below walk: drawn weights in the test shape *)
let test_model ~seed = Model.For_test.(init config ~seed)

(* ==================================================================== *)
(* The image the bitstream carries *)
(* ==================================================================== *)

let%expect_test "the ROM image holds the three matrices of a layer and not the six \
                 tensors"
  =
  let config = Model.For_test.config in
  let model = Model.For_test.init config ~seed:11 in
  let bases = Model.Rom_data.to_list (Model.rom_bases model) in
  let sizes =
    List.map (Model.rom_tensors model) ~f:(fun (t : Model.quantized) -> Array.length t.q)
  in
  Stdio.printf "%s\n" (Sexp.to_string ([%sexp_of: int list] bases));
  let ends = List.map2_exn bases sizes ~f:( + ) in
  Stdio.printf
    "each base is the end of the one before: %b, and the image ends at the last: %b\n"
    ([%compare.equal: int list] (List.drop bases 1) (List.drop_last_exn ends))
    (List.last_exn ends = Array.length (Model.rom_bits model));
  (* the three per-head tensors never reach the image: they are the constants of the ops *)
  Stdio.printf
    "the checkpoint holds %d tensors and the image %d\n"
    (List.length (Mamba.Params.shapes config))
    (List.length sizes);
  [%expect
    {|
    (0 3072 3328 4640 4832 5344 5856 6368 6624 6880 7904)
    each base is the end of the one before: true, and the image ends at the last: true
    the checkpoint holds 14 tensors and the image 11
    |}]
;;

let%expect_test "the checkpoint seam: a file quantizes as its tensors do" =
  let config = Model.For_test.config in
  let tensors = Mamba.Params.to_list (Mamba.Params.init config ~seed:5) in
  let same (a : Model.quantized) (b : Model.quantized) =
    a.e = b.e && Array.equal Int.equal a.q b.q
  in
  Mamba.For_test.with_checkpoint tensors ~f:(fun path ->
    let read = Model.of_checkpoint config path in
    let made =
      Model.of_floats
        config
        ~temperature:Mamba.elected_temperature
        ~min_p:Mamba.elected_min_p
        (List.map tensors ~f:Nx.to_array)
    in
    Stdio.printf
      "%d image tensors, the file and the tensors quantize alike: %b\n"
      (List.length (Model.rom_tensors read))
      (List.for_all2_exn (Model.rom_tensors read) (Model.rom_tensors made) ~f:same);
    (* the per-head constants cross the seam as well, and they are where a coarse byte
       would have shown *)
    let per_head (a : Model.layer) (b : Model.layer) =
      match a, b with
      | Block a, Block b ->
        Array.equal Int.equal a.dt_bias b.dt_bias
        && Array.equal Int.equal a.d_skip b.d_skip
        && Array.for_all2_exn a.decay b.decay ~f:(fun x y ->
          x.q_value = y.q_value && x.q = y.q)
      (* the other kinds carry none: their tensors are the whole of them *)
      | (_ : Model.layer), (_ : Model.layer) -> true
    in
    Stdio.printf
      "the decay, the bias and the skip agree: %b\n"
      (Array.for_all2_exn read.layers made.layers ~f:per_head);
    Stdio.printf
      "the file takes the elected policy: %b\n"
      (read.temper.q_value = made.temper.q_value && read.min_weight = made.min_weight));
  [%expect
    {|
    11 image tensors, the file and the tensors quantize alike: true
    the decay, the bias and the skip agree: true
    the file takes the elected policy: true
    |}]
;;

let%expect_test "the checkpoint seam: a tensor of the wrong count raises" =
  let config = Model.For_test.config in
  let tensors =
    List.mapi (Mamba.Params.shapes config) ~f:(fun index shape ->
      Nx.zeros Nx.float32 (if index = 2 then [| shape.(0) - 1; shape.(1) |] else shape))
  in
  Mamba.For_test.with_checkpoint tensors ~f:(fun path ->
    Stdio.printf
      "%s\n"
      (Mamba.For_test.refusal ~path (fun () ->
         let (_ : Model.t) = Model.of_checkpoint config path in
         ())));
  [%expect {| <file> tensor 2 holds 1230 values, not 1312 |}]
;;

let%expect_test "the lead-in draws nothing, and the seed names the walk" =
  let walk ~seed ~steps =
    let engine = ref (Engine.init (test_model ~seed:1) ~seed) in
    List.map (List.range 0 steps) ~f:(fun (_ : int) ->
      let next, step = Engine.next_step !engine in
      engine := next;
      step)
  in
  let steps = walk ~seed:7 ~steps:20 in
  List.iteri steps ~f:(fun index (step : Engine.step) ->
    if index < 3 || index >= 15
    then
      Stdio.printf "step %2d  %08x  %d draws\n" index step.frame (List.length step.draws));
  [%expect
    {|
    step  0  00000000  0 draws
    step  1  00000000  0 draws
    step  2  00000000  0 draws
    step 15  00000000  0 draws
    step 16  ceafc5a4  4 draws
    step 17  c2adaeb7  4 draws
    step 18  c0b3adb8  4 draws
    step 19  c8bcb1bd  4 draws
    |}];
  let frames w = List.map w ~f:(fun (s : Engine.step) -> s.frame) in
  Stdio.printf
    "the same seed repeats: %b\n"
    (List.equal Int.equal (frames steps) (frames (walk ~seed:7 ~steps:20)));
  Stdio.printf
    "another seed parts: %b\n"
    (not (List.equal Int.equal (frames steps) (frames (walk ~seed:8 ~steps:20))));
  [%expect {|
    the same seed repeats: true
    another seed parts: true
    |}]
;;

(* The seed 0 is the fixed point of xorshift32, and the panel can state it: all the slide
   switches down is the rest position of the board. The walk therefore stands still —
   every uniform is 0, thus every threshold is 0, thus each seat takes the first class
   that min-p left standing. It is degenerate and it is well defined, and it is what the
   board plays for that seed, thus a gate pins it here as one pinned era four's.

   The float sampler answers another walk for the same number: [Mamba.sample] folds its
   seed and 0 is the one seed the fold does not carry to the board. *)
let%expect_test "the seed 0 stands still, and each seat takes the first class it may" =
  let engine = ref (Engine.init (test_model ~seed:1) ~seed:0) in
  let steps = ref [] in
  for _ = 1 to 20 do
    let next, (step : Engine.step) = Engine.next_step !engine in
    engine := next;
    steps := step :: !steps
  done;
  let drawn =
    List.rev !steps
    |> List.filter ~f:(fun (s : Engine.step) -> not (List.is_empty s.draws))
  in
  let stands_still (s : Engine.step) =
    List.for_all s.draws ~f:(fun (d : Engine.draw) -> Float.(d.uniform = 0.0))
  in
  Stdio.printf
    "%d drawn steps, every uniform 0: %b\n"
    (List.length drawn)
    (List.for_all drawn ~f:stands_still);
  List.iter drawn ~f:(fun (s : Engine.step) -> Stdio.printf "  %08x\n" s.frame);
  [%expect
    {|
    4 drawn steps, every uniform 0: true
      00000000
      00000000
      00000000
      00000000
    |}]
;;

let%expect_test "the chain draws from the soprano down, and each seat lands in its seat" =
  let engine = ref (Engine.init (test_model ~seed:2) ~seed:3) in
  let steps = ref [] in
  for _ = 1 to 20 do
    let next, (step : Engine.step) = Engine.next_step !engine in
    engine := next;
    steps := step :: !steps
  done;
  let drawn =
    List.rev !steps
    |> List.filter ~f:(fun (s : Engine.step) -> not (List.is_empty s.draws))
  in
  Stdio.printf
    "%d drawn steps, the order %s\n"
    (List.length drawn)
    (Sexp.to_string
       ([%sexp_of: int list]
          (List.map (List.hd_exn drawn).draws ~f:(fun (d : Engine.draw) -> d.seat))));
  List.iter (List.take drawn 4) ~f:(fun (step : Engine.step) ->
    let by_seat = Array.create ~len:Frame.voices 0 in
    List.iter step.draws ~f:(fun (d : Engine.draw) -> by_seat.(d.seat) <- d.drawn);
    Stdio.printf
      "  %08x  drawn %s  frame %s\n"
      step.frame
      (Sexp.to_string ([%sexp_of: int array] by_seat))
      (Sexp.to_string ([%sexp_of: int list] (Vocab.classes_of_frame step.frame))));
  [%expect
    {|
    4 drawn steps, the order (3 2 1 0)
      b5b0afb1  drawn (14 12 13 18)  frame (14 12 13 18)
      bbb1c1b1  drawn (14 30 14 24)  frame (14 30 14 24)
      a7cabeb2  drawn (15 27 39 4)  frame (15 27 39 4)
      cbccb3ce  drawn (43 16 41 40)  frame (43 16 41 40)
    |}]
;;

(* The one memory this machine modifies rather than rewrites. A tap ring read one slot
   late or a state row written into another lane's would still make music, and it would
   make music the circuit could reproduce exactly, thus the frame gate would pass. These
   hold the two memories to their rules directly. *)
let%expect_test "the tap ring reads the steps behind it, and zero before the walk began" =
  let width = Mamba.Config.baseline.taps in
  let taps = Array.init (3 * width) ~f:(fun (_ : int) -> 0) in
  (* one channel, the values 10, 20, 30, 40, 50 entering at positions 0 to 4 *)
  let read position =
    List.init width ~f:(fun k ->
      Engine.tap_at ~width ~taps ~base:0 ~position ~channel:1 ~k)
  in
  List.iteri [ 10; 20; 30; 40; 50 ] ~f:(fun position v ->
    taps.(Engine.tap_slot ~width ~base:0 ~channel:1 ~at:position) <- v;
    Stdio.printf
      "position %d: %s\n"
      position
      (Sexp.to_string ([%sexp_of: int list] (read position))));
  [%expect
    {|
    position 0: (10 0 0 0)
    position 1: (20 10 0 0)
    position 2: (30 20 10 0)
    position 3: (40 30 20 10)
    position 4: (50 40 30 20)
    |}]
;;
