open Base
module Params_data = Mamba.Params_data

module Constants = struct
  (* the logits stay wide; no constant names them *)
  let h_q = 16
  let y_q = 12
  let v_q = 12
  let s_q = 12
  let alpha_q = 15
  let beta_q = 15

  (* the gate product, in an int32: two Q12 values multiply and nothing truncates them
     before the norm that reads them *)
  let gate_q = 2 * v_q

  (* the rms epsilon of the float model, in the Q of the squared stream: the sum squares a
     Q12 copy, thus the mean is Q(2 y_q) *)
  let eps_q = Float.iround_nearest_exn (Float.ldexp 1e-6 (2 * y_q))

  (* A fixed-point multiplier: the value stands for [q_value * 2^-q]. The Q travels with
     the value because the two are one fact — a multiply that takes the wrong shift is
     silently wrong, and both the reference and the circuit apply these scales. *)
  type scale =
    { q_value : int
    ; q : int
    }

  let apply { q_value; q } v = (v * q_value) asr q

  (* log2(e): the exp2 form of an exponential *)
  let log2e =
    let q = 15 in
    { q_value = Float.iround_nearest_exn (Float.ldexp (1.0 /. Float.log 2.0) q); q }
  ;;

  (* The exp2 table of era four, read back through its public image. One table serves the
     softmax of that circuit and the decay of this one, and [Exp2] is the unit both drive,
     thus the numbers cannot part. *)
  let exp2_table =
    Array.map Mgen_transformer.Quantized.Constants.exp2_bits ~f:Hardcaml.Bits.to_int_trunc
  ;;

  (* The sigmoid of a Q12 value, in Q15. The input is int16, thus its range is |v| < 8
     exactly and the clamp of the design document costs nothing: 256 buckets of 256 Q12
     units cover it, and the index is the top eight bits with the sign flipped.

     The entry is the sigmoid at the CENTRE of its bucket and not at its left edge. The
     bucket is 1/16 wide and the slope peaks at 1/4, thus the left edge would bias every
     reading by up to 2^-10 of full scale; the centre halves the worst error and costs
     nothing at elaboration. The centres are symmetric about zero, thus the two halves of
     the table sum to 2^15 and sigmoid(-v) = 1 - sigmoid(v) survives the quantization. *)
  let sigmoid_table =
    Array.init 256 ~f:(fun j ->
      let v = (Float.of_int (j - 128) +. 0.5) /. 16.0 in
      Float.iround_nearest_exn (32768.0 /. (1.0 +. Float.exp (-.v))))
  ;;

  (* The correction term of the softplus, ln(1 + exp(-|v|)), in Q12 over a Q12 magnitude.

     softplus(v) = relu(v) + this. The ramp is exact and carries the whole of a large
     input, thus the table only has to hold a quantity that falls to nothing: at |v| = 8,
     the largest magnitude an int16 Q12 value takes, it is one unit of Q12. 256 buckets of
     128 units cover the range, and the entry is again the centre of its bucket. *)
  let softplus_table =
    Array.init 256 ~f:(fun j ->
      let v = (Float.of_int j +. 0.5) /. 32.0 in
      Float.iround_nearest_exn (4096.0 *. Float.log (1.0 +. Float.exp (-.v))))
  ;;

  let bits16 table = Array.map table ~f:(Hardcaml.Bits.of_unsigned_int ~width:16)
  let sigmoid_bits = bits16 sigmoid_table
  let softplus_bits = bits16 softplus_table

  (* The index rules, stated once for the reference and the two ROMs.

     [sigmoid_index] is the top eight bits of an int16 with the sign bit flipped, which is
     no arithmetic at all in the circuit. [softplus_index] is the magnitude shifted, and
     the clamp catches the one value -32768 whose magnitude does not fit the table. *)
  let sigmoid_index v = ((v asr 8) + 128) land 255
  let softplus_index v = Int.min 255 (Int.abs v asr 7)

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
end

module Tensor = struct
  type t = int array
  type floats = float array

  (* the index of the peak; the compare is strict, thus a tie keeps the first *)
  let peak_index (values : floats) =
    Array.foldi values ~init:0 ~f:(fun i best v ->
      if Float.(v > values.(best)) then i else best)
  ;;

  let dot a b = Array.fold2_exn a b ~init:0.0 ~f:(fun acc x y -> Float.(acc + (x * y)))
  let floats_of (q : t) = Array.map q ~f:Float.of_int
  let same_peak (q : t) (f : floats) = peak_index (floats_of q) = peak_index f

  let cosine (q : t) (f : floats) =
    let q = floats_of q in
    Float.(dot q f / sqrt (dot q q * dot f f))
  ;;
end

module Model = struct
  type quantized =
    { q : Tensor.t
    ; e : int
    }

  (* The tensors the ROM carries, in the order it carries them: the two tables, then the
     three matrices of each layer. It is NOT the order of the checkpoint — the checkpoint
     holds six tensors a layer and three of them never reach the ROM — thus the two orders
     are two structures and neither is implied by the other. *)
  module Rom_data = struct
    type 'a t =
      { seats : 'a
      ; phase : 'a
      ; layers : 'a layer array
      }

    and 'a layer =
      { w_in : 'a
      ; conv : 'a
      ; w_out : 'a
      }

    let to_list { seats; phase; layers } =
      seats
      :: phase
      :: List.concat_map (Array.to_list layers) ~f:(fun { w_in; conv; w_out } ->
        [ w_in; conv; w_out ])
    ;;

    let of_list ~layers items =
      match items with
      | seats :: phase :: rest ->
        let groups =
          List.chunks_of rest ~length:3
          |> List.map ~f:(function
            | [ w_in; conv; w_out ] -> { w_in; conv; w_out }
            | _ -> invalid_arg "a layer takes three tensors in the image")
          |> Array.of_list
        in
        if Array.length groups <> layers
        then
          invalid_arg
            (Printf.sprintf
               "%d image layer groups do not fit %d layers"
               (Array.length groups)
               layers);
        { seats; phase; layers = groups }
      | _ -> invalid_arg "the image starts with the two tables"
    ;;
  end

  (* The weights of one layer as the machine holds them: three tensors in the ROM, and the
     per-head numbers as constants the ops carry.

     [a_log], [dt_bias] and [d_skip] are [heads] values a layer, and an int8 tensor cannot
     hold them. The bias enters a softplus: a step of one part in 127 of its range moves
     [dt] by more than a small [dt] is, and the decay would follow it. They quantize at
     elaboration instead, into the numbers the ops carry — [a * log2(e)] folds into one Q
     constant for each head, exactly as era four folded log2(e) into the temper — thus the
     run time never sees them as tensors. *)
  type layer =
    { w_in : quantized
    ; conv : quantized
    ; w_out : quantized
    ; decay : Constants.scale array (** [a * log2(e)], one for each head *)
    ; dt_bias : Tensor.t (** Q12 *)
    ; d_skip : Tensor.t (** Q12 *)
    }

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
    let { Mamba.Config.d; d_in; heads; state; layers } = t.config in
    (* the rms_norm of the stream divides by [d] and the gated norm by [d_in]: a shift *)
    assert (Int.is_pow2 d);
    assert (Int.is_pow2 d_in);
    (* the state address is (head, lane, n) concatenated, and the tap address (channel,
       tap), thus every field of them is a power of two *)
    assert (Int.is_pow2 heads);
    assert (Int.is_pow2 (Mamba.Config.head t.config));
    assert (Int.is_pow2 state);
    assert (Int.is_pow2 Mamba.conv_taps);
    assert (layers = Array.length t.layers);
    Array.iter t.layers ~f:(fun l ->
      assert (Array.length l.decay = heads);
      assert (Array.length l.dt_bias = heads);
      assert (Array.length l.d_skip = heads));
    (* the seat rows and the phase row add row for row — [Engine.embed] adds them, and the
       Embed op of the circuit walks them as one tensor — thus one exponent covers both *)
    assert (t.phase.e = t.seats.e);
    assert (Array.length t.seats.q = Frame.voices * Vocab.classes * d)
  ;;

  (* the element counts of the ROM tensors in the order of the image *)
  let rom_sizes (config : Mamba.Config.t) =
    let layer =
      [ config.d * Mamba.Config.projection config
      ; Mamba.Config.channels config * Mamba.conv_taps
      ; config.d_in * config.d
      ]
    in
    [ Frame.voices * Vocab.classes * config.d; Jsb.bar_steps * config.d ]
    @ List.concat (List.init config.layers ~f:(fun (_ : int) -> layer))
  ;;

  let rom_tensors t =
    Rom_data.to_list
      { Rom_data.seats = t.seats
      ; phase = t.phase
      ; layers =
          Array.map t.layers ~f:(fun l ->
            { Rom_data.w_in = l.w_in; conv = l.conv; w_out = l.w_out })
      }
  ;;

  (* the running sums of the sizes, handed back through the one definition of the order *)
  let rom_bases t =
    List.folding_map (rom_sizes t.config) ~init:0 ~f:(fun start size ->
      start + size, start)
    |> Rom_data.of_list ~layers:t.config.Mamba.Config.layers
  ;;

  let rom_bits t =
    Array.concat_map
      (Array.of_list (rom_tensors t))
      ~f:(fun { q; e = (_ : int) } ->
        Array.map q ~f:(fun v -> Hardcaml.Bits.of_unsigned_int ~width:8 (v land 255)))
  ;;

  (* The policy in the integer forms of the machine; the rules of the float sampler. The
     temper is log2(e) / T, and its Q is one below the Q of [Constants.log2e]. The extra
     bit is headroom for the temperature: the circuit multiplies by this constant on an
     18-bit signed port, thus the Q of [log2e] would overflow that port under a
     temperature of about 0.36, and this Q holds down to about 0.18. *)
  let policy ~temperature ~min_p =
    Mamba.check_policy ~temperature ~min_p;
    let q = Constants.log2e.q - 1 in
    ( { Constants.q_value =
          Float.iround_nearest_exn (Float.ldexp (1.0 /. Float.log 2.0 /. temperature) q)
      ; q
      }
    , Float.iround_nearest_exn (min_p *. 32768.0) )
  ;;

  (* the quantization arithmetic: pure functions from the float values to the int8 form *)
  let max_abs (floats : Tensor.floats) =
    Array.fold floats ~init:0.0 ~f:(fun acc v -> Float.max acc (Float.abs v))
  ;;

  (* the largest exponent that keeps round(max|w| * 2^e) at 127 or less; 14 caps the
     all-zero tensor *)
  let max_exponent v =
    let fits e = Float.iround_nearest_exn (Float.ldexp v e) <= 127 in
    let rec largest e = if fits e then e else largest (e - 1) in
    if Float.(v <= 0.0) then 14 else largest 14
  ;;

  (* [e] overrides the exponent of the tensor's own peak — the two tables share one,
     because their rows add *)
  let quantize ?e (floats : Tensor.floats) =
    let e = Option.value e ~default:(max_exponent (max_abs floats)) in
    let clamp ft =
      Int.clamp_exn (Float.iround_nearest_exn (Float.ldexp ft e)) ~min:(-127) ~max:127
    in
    { q = Array.map floats ~f:clamp; e }
  ;;

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
      Params_data.of_list ~layers:config.layers tensors
    in
    let e = max_exponent (Float.max (max_abs seats) (max_abs phase)) in
    { config
    ; temper
    ; min_weight
    ; seats = quantize ~e seats
    ; phase = quantize ~e phase
    ; layers =
        Array.map layers ~f:(fun (l : Tensor.floats Params_data.layer) ->
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
          })
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
    (* the shape of a test model: small enough to run in a simulation, and the same
       structure as the model of the era *)
    let config = { Mamba.Config.d = 16; d_in = 32; heads = 2; state = 8; layers = 1 }

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
    (** [layers * d_in * state] values, Q12 — the recurrence, and the one memory this
        machine modifies rather than rewrites *)
    ; taps : Tensor.t (** [layers * channels * conv_taps] values, Q12 *)
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

  let clamp16 v = Int.clamp_exn v ~min:(-32768) ~max:32767
  let clamps16 v = v > 32767 || v < -32768

  (* the reductions of the engine: [sum n f] is the MAC — the sum of [f i] over
     [0 .. n - 1] — and [max_over n f] is the peak scan *)
  let sum n f =
    let rec go acc i = if i = n then acc else go (acc + f i) (i + 1) in
    go 0 0
  ;;

  let max_over n f =
    let rec go acc i = if i = n then acc else go (Int.max acc (f i)) (i + 1) in
    go Int.min_value 0
  ;;

  (* floor of the square root; any correct algorithm gives the one answer the circuit must
     also give *)
  let isqrt n =
    if n <= 0
    then 0
    else (
      let rec shrink g = if g * g > n then shrink (g - 1) else g in
      let rec grow g = if (g + 1) * (g + 1) <= n then grow (g + 1) else g in
      grow (shrink (Float.to_int (Float.sqrt (Float.of_int n)))))
  ;;

  (* exp2 of a nonnegative Q12 magnitude, giving 2^-m in Q15: the integer part shifts, the
     top eight bits of the fraction index the table. The peak — a magnitude of 0 — is
     2^15.

     It is era four's rule read the other way round. That circuit exponentiated a
     nonpositive score and this one a decay that is a magnitude by construction, thus the
     negation stands at the caller there and here it does not stand at all. The gate below
     holds the two to the same table. *)
  let exp2_of_magnitude m =
    let i = m asr 12 in
    if i >= 16 then 0 else Constants.exp2_table.((m asr 4) land 255) asr i
  ;;

  (* the sigmoid of a Q12 value in Q15, and SiLU over it: one table read, one multiply and
     one shift *)
  let sigmoid_q v = Constants.sigmoid_table.(Constants.sigmoid_index v)
  let silu v = clamp16 ((v * sigmoid_q v) asr Constants.alpha_q)

  (* softplus as the ramp and the correction the table holds. The sum rides an int16, thus
     the input clamps before the table reads it and the result clamps after. *)
  let softplus v =
    let v = clamp16 v in
    clamp16 (Int.max 0 v + Constants.softplus_table.(Constants.softplus_index v))
  ;;

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

  (* The tap ring of one layer: a ring of [conv_taps] for each channel, and the position
     names the slot. Tap k reads the step k back, and it reads ZERO while the walk has not
     run k steps — thus the origin needs no clearing walk and the rule is a mux, as era
     four's fill count was. *)
  let tap_slot ~base ~channel ~at =
    base + (channel * Mamba.conv_taps) + (at land (Mamba.conv_taps - 1))
  ;;

  let tap_at ~taps ~base ~position ~channel ~k =
    if position < k then 0 else taps.(tap_slot ~base ~channel ~at:(position - k))
  ;;

  (* One layer of the trunk. It gives the stream after the residual join and the clamps it
     met on the way, and it writes the state and the taps of its layer in place — the two
     arrays are copies the caller made for this step. *)
  let layer t ~index (lay : Model.layer) (h : Tensor.t) ~state ~taps ~clamps =
    let { Mamba.Config.d; d_in; heads; state = n; layers = (_ : int) } = config t in
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
    (* the convolution: the step's input enters the ring, then a row of [conv_taps] terms
       for each channel, then the SiLU chain over the sums *)
    let tap_base = index * channels * Mamba.conv_taps in
    Array.iteri (Array.sub zxbcdt ~pos:d_in ~len:channels) ~f:(fun c v ->
      taps.(tap_slot ~base:tap_base ~channel:c ~at:position) <- v);
    let xbc =
      Array.init channels ~f:(fun c ->
        let acc =
          sum Mamba.conv_taps (fun k ->
            tap_at ~taps ~base:tap_base ~position ~channel:c ~k
            * lay.conv.q.((c * Mamba.conv_taps) + k))
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

  (* One step of the recurrence: the engine after it.

     The two memories are copied once and then written in place through the layer walk.
     This is the local mutation the style rule allows for a mutable structure used as one:
     the state RAM of the circuit is written in place as well, and a fold that rebuilt a
     12 288-element array for each of six layers would model something the machine does
     not do. Nothing outside this function sees a half-written memory. *)
  let forward t ~frame ~phase =
    let state = Array.copy t.state in
    let taps = Array.copy t.taps in
    let h, clamps =
      Array.foldi
        t.model.layers
        ~init:(embed t ~frame ~phase, t.clamps)
        ~f:(fun index (h, clamps) lay -> layer t ~index lay h ~state ~taps ~clamps)
    in
    { t with h; state; taps; position = t.position + 1; clamps }
  ;;

  (* The origin of a walk: a zero state, an empty tap ring and no residual. The lead-in is
     not here — it is the first steps of the walk itself, thus [next_step] states it and a
     caller that counts steps counts the steps the float sampler counts. *)
  let init (model : Model.t) ~seed =
    Model.check_shape model;
    let { Mamba.Config.d; d_in; heads = (_ : int); state = n; layers } = model.config in
    { model
    ; state = Array.create ~len:(layers * d_in * n) 0
    ; taps =
        Array.create
          ~len:(layers * Mamba.Config.channels model.config * Mamba.conv_taps)
          0
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

  (* three PRNG bytes, high first: the walk of [Prng.uniform] *)
  let u24 prng =
    let open Prng in
    run
      (let* high = next in
       let* middle = next in
       let+ low = next in
       (((high * 256) + middle) * 256) + low)
      prng
  ;;

  (* The draw over the logits of one seat. No mask stands before it, because no frame is
     illegal.

     The arithmetic decides the tie the float twin has to argue about: [u] is below 2^24,
     thus the threshold is below the total, thus some running total passes it and the
     class the walk names always holds weight. The walk needs no fallback and states none. *)
  let draw_of_logits t ~logits =
    let peak = max_over classes (fun c -> logits.(c)) in
    let weight c =
      let e = exp2_of_magnitude (-Constants.apply t.model.temper (logits.(c) - peak)) in
      if e >= t.model.min_weight then e else 0
    in
    let weights = Array.init classes ~f:weight in
    let total = sum classes (fun c -> weights.(c)) in
    let prng, u = u24 t.prng in
    let threshold = (u * total) asr 24 in
    let rec walk c running =
      if c = classes - 1
      then c
      else (
        let running = running + weights.(c) in
        if running > threshold then c else walk (c + 1) running)
    in
    { t with prng }, Float.of_int u *. 0x1p-24, walk 0 0
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
      let state = Array.copy t.state in
      let taps = Array.copy t.taps in
      let origin = embed t ~frame ~phase in
      let (_ : Tensor.t * Clamps.t), rows =
        List.fold_map
          (List.init (Array.length t.model.layers) ~f:Fn.id)
          ~init:(origin, t.clamps)
          ~f:(fun (h, clamps) index ->
            let next, clamps =
              layer t ~index t.model.layers.(index) h ~state ~taps ~clamps
            in
            (next, clamps), next)
      in
      (* the embed stands at the head, thus the list is one entry for each write of the
         whole stream that the circuit makes *)
      origin :: rows
    ;;

    let isqrt = isqrt
    let exp2_of_magnitude = exp2_of_magnitude
    let sigmoid_q = sigmoid_q
    let softplus = softplus
    let silu = silu
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
(* The tables and the image the bitstream carries *)
(* ==================================================================== *)

let%expect_test "the exponent of a tensor, and the clamp of the byte" =
  List.iter [ 0.0; 0.02; 0.08; 127.0; 127.49; 127.5; 1e9 ] ~f:(fun v ->
    Stdio.printf "%-6g -> %d\n" v (Model.max_exponent v));
  (* The byte is two's complement and the negative end is not used: the clamp is -127 and
     not -128, thus the image is symmetric and a negated weight is a negated byte. *)
  let { Model.q; e } = Model.quantize ~e:0 [| 200.0; -200.0; 5.4; -5.5; 0.0 |] in
  Stdio.printf "at e %d: %s\n" e (Sexp.to_string ([%sexp_of: int array] q));
  [%expect
    {|
    0      -> 14
    0.02   -> 12
    0.08   -> 10
    127    -> 0
    127.49 -> 0
    127.5  -> -1
    1e+09  -> -23
    at e 0: (127 -127 5 -5 0)
    |}]
;;

let%expect_test "the exp2 table is era four's, read the other way round" =
  (* The decay is a magnitude by construction and the softmax exponent was nonpositive,
     thus this engine negates at the caller and era four's negated inside. One table and
     one shift rule serve both, and this holds them to it over every reading the unit can
     make: 256 entries under 16 shifts is EVERY distinct reading below the zero floor. *)
  let disagrees m =
    Engine.For_test.exp2_of_magnitude m
    <> Mgen_transformer.Quantized.Engine.For_test.exp2_q (-m)
  in
  let readings = List.map (List.range 0 4096) ~f:(fun k -> k * 16) in
  let above = List.map (List.range 0 64) ~f:(fun k -> 65536 + (k * 977)) in
  Stdio.printf
    "%d readings of the table, %d disagree; %d above the floor, %d disagree\n"
    (List.length readings)
    (List.length (List.filter readings ~f:disagrees))
    (List.length above)
    (List.length (List.filter above ~f:disagrees));
  [%expect {| 4096 readings of the table, 0 disagree; 64 above the floor, 0 disagree |}]
;;

let%expect_test "the sigmoid table: the ends, the middle and the symmetry" =
  let show v =
    Stdio.printf
      "  %6d (%.4f) -> %5d (%.4f)\n"
      v
      (Float.of_int v /. 4096.0)
      (Engine.For_test.sigmoid_q v)
      (Float.of_int (Engine.For_test.sigmoid_q v) /. 32768.0)
  in
  List.iter [ -32768; -16384; -4096; 0; 4096; 16384; 32767 ] ~f:show;
  (* the centres of the buckets are symmetric about zero, thus a value and its negative
     weigh 2^15 together — the one property of the sigmoid the quantization can keep *)
  let asymmetric =
    List.count (List.range 0 256) ~f:(fun j ->
      Constants.sigmoid_table.(j) + Constants.sigmoid_table.(255 - j) <> 32768)
  in
  Stdio.printf "%d of 128 mirrored pairs do not sum to 2^15\n" asymmetric;
  [%expect
    {|
      -32768 (-8.0000) ->    11 (0.0003)
      -16384 (-4.0000) ->   608 (0.0186)
       -4096 (-1.0000) ->  9015 (0.2751)
           0 (0.0000) -> 16640 (0.5078)
        4096 (1.0000) -> 24155 (0.7372)
       16384 (4.0000) -> 32196 (0.9825)
       32767 (7.9998) -> 32757 (0.9997)
    0 of 128 mirrored pairs do not sum to 2^15
    |}]
;;

let%expect_test "the softplus is the ramp and its correction" =
  (* against the float function the reference of [Mamba] states: relu(v) + ln(1+exp(-|v|)) *)
  let worst = ref 0.0 in
  let at = ref 0 in
  for v = -32768 to 32767 do
    let float_v = Float.of_int v /. 4096.0 in
    let want =
      Float.max 0.0 float_v +. Float.log (1.0 +. Float.exp (-.Float.abs float_v))
    in
    let gap = Float.abs (want -. (Float.of_int (Engine.For_test.softplus v) /. 4096.0)) in
    if Float.(gap > !worst)
    then (
      worst := gap;
      at := v)
  done;
  Stdio.printf
    "over every int16 input the table stands within %.5f of the float softplus, worst at \
     %.4f\n"
    !worst
    (Float.of_int !at /. 4096.0);
  [%expect
    {| over every int16 input the table stands within 0.00784 of the float softplus, worst at 0.0000 |}]
;;

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
    (0 3072 3328 4640 4832)
    each base is the end of the one before: true, and the image ends at the last: true
    the checkpoint holds 8 tensors and the image 5
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
    Stdio.printf
      "the decay, the bias and the skip agree: %b\n"
      (Array.for_all2_exn read.layers made.layers ~f:(fun a b ->
         Array.equal Int.equal a.dt_bias b.dt_bias
         && Array.equal Int.equal a.d_skip b.d_skip
         && Array.for_all2_exn a.decay b.decay ~f:(fun x y ->
           x.q_value = y.q_value && x.q = y.q)));
    Stdio.printf
      "the file takes the elected policy: %b\n"
      (read.temper.q_value = made.temper.q_value && read.min_weight = made.min_weight));
  [%expect
    {|
    5 image tensors, the file and the tensors quantize alike: true
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

let%expect_test "isqrt floors" =
  List.iter [ 0; 1; 2; 3; 4; 15; 16; 17; 1_000_000 ] ~f:(fun n ->
    Stdio.printf "%d " (Engine.For_test.isqrt n));
  Stdio.printf "\n";
  [%expect {| 0 1 1 1 2 3 4 4 1000 |}]
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
  let taps = Array.init (3 * Mamba.conv_taps) ~f:(fun (_ : int) -> 0) in
  (* one channel, the values 10, 20, 30, 40, 50 entering at positions 0 to 4 *)
  let read position =
    List.init Mamba.conv_taps ~f:(fun k ->
      Engine.tap_at ~taps ~base:0 ~position ~channel:1 ~k)
  in
  List.iteri [ 10; 20; 30; 40; 50 ] ~f:(fun position v ->
    taps.(Engine.tap_slot ~base:0 ~channel:1 ~at:position) <- v;
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
