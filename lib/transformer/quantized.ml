open Base
module Params_data = Transformer.Params_data

module Constants = struct
  (* the scores and the logits are Q12 as well, and stay wide; no constant names them *)
  let h_q = 16
  let y_q = 12
  let kv_q = 12
  let hid_q = 10

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

  (* [apply s v] scales [v] by [s], toward negative infinity — an arithmetic shift, as the
     circuit's. *)
  let apply { q_value; q } v = (v * q_value) asr q

  (* log2(e): the exp2 form of the softmax exponent *)
  let log2e =
    let q = 15 in
    { q_value = Float.iround_nearest_exn (Float.ldexp (1.0 /. Float.log 2.0) q); q }
  ;;

  (* the quantized exponential: exp2 of -j/256 in Q15 — the table of the softmax and the
     sampler, and the same species as the weights *)
  let exp2_table =
    Array.init 256 ~f:(fun j ->
      Float.iround_nearest_exn Float.(32768.0 * (2.0 ** (-of_int j / 256.0))))
  ;;

  let exp2_bits = Array.map exp2_table ~f:(Hardcaml.Bits.of_unsigned_int ~width:16)

  (* The score walk sums the products of two Q[kv_q] values, thus its sum is Q(2 kv_q);
     this brings it to Q[y_q] and applies the 1/sqrt(head_d) of the reference in the same
     shift, thus [head_d] is a power of four. *)
  let score_shift ~head_d = (2 * kv_q) - y_q + (Int.floor_log2 head_d / 2)

  (* The ALiBi slope of head [head] is 2^-(this), thus the penalty is a shift of the age. *)
  let slope_exponent ~span ~heads ~head = span * (head + 1) / heads
end

module Tensor = struct
  type t = int array
  type floats = float array

  let same_peak (q : t) (f : floats) =
    (* the index of the peak; a tie keeps the first *)
    let argmax n value =
      let rec go best i =
        if i = n
        then best
        else go (if Float.(value i > value best) then i else best) (i + 1)
      in
      go 0 1
    in
    argmax (Array.length q) (fun i -> Float.of_int q.(i))
    = argmax (Array.length f) (fun i -> f.(i))
  ;;

  let cosine (q : t) (f : floats) =
    let n = Array.length q in
    let sum g =
      let rec go acc i = if i = n then acc else go Float.(acc + g i) (i + 1) in
      go 0.0 0
    in
    let dot = sum (fun i -> Float.(of_int q.(i) * f.(i))) in
    let qq = sum (fun i -> Float.(of_int q.(i) * of_int q.(i))) in
    let ff = sum (fun i -> Float.(f.(i) * f.(i))) in
    dot /. Float.sqrt (qq *. ff)
  ;;
end

module Model = struct
  type quantized =
    { q : Tensor.t
    ; e : int
    }

  type params = quantized Params_data.t
  type layer = quantized Params_data.layer

  type t =
    { config : Transformer.Config.t
    ; params : params
    ; temper : Constants.scale
    (** the sampling temper, log2(e) / T — folded with the exp2 form *)
    ; min_weight : int (** the min-p share of the peak weight 2^15 *)
    }

  (* The arithmetic of the circuit is shifts, thus the shape obeys the shift rules. The
     reference holds this check because the reference states the rules; the circuit calls
     it at elaboration, where a bad shape must fail loudly. The record is open, thus a
     model that no constructor here made can break a rule, and both consumers of a broken
     model would be silently wrong. *)
  let check_shape t =
    let { Transformer.Config.d; heads; context = slots; layers; slope_span = (_ : int) } =
      t.config
    in
    let { Params_data.seats; phase; layers = tensors } = t.params in
    assert (Int.is_pow2 d);
    assert (Int.is_pow2 slots);
    assert (Int.floor_log2 (d / heads) % 2 = 0);
    assert (layers = Array.length tensors);
    (* the seat rows and the phase row add row for row — [Engine.embed] adds them, and the
       Embed op of the circuit walks them as one tensor — thus one exponent covers both.
       The four seat tables share it for the same reason: they stand in one tensor, and a
       drift report is the instrument that would ask for four. *)
    assert (phase.e = seats.e);
    assert (Array.length seats.q = Frame.voices * Vocab.classes * d)
  ;;

  (* the element counts of the tensors in the flat order, from the one definition of the
     shapes *)
  let sizes config =
    List.map (Transformer.Params.shapes config) ~f:(fun shape ->
      Array.fold shape ~init:1 ~f:( * ))
  ;;

  (* the running sums of the sizes, handed back through the one definition of the order *)
  let rom_bases t =
    List.folding_map (sizes t.config) ~init:0 ~f:(fun start size -> start + size, start)
    |> Params_data.of_list ~layers:t.config.Transformer.Config.layers
  ;;

  let rom_bits t =
    Array.concat_map
      (Array.of_list (Params_data.to_list t.params))
      ~f:(fun { q; e = (_ : int) } ->
        Array.map q ~f:(fun v -> Hardcaml.Bits.of_unsigned_int ~width:8 (v land 255)))
  ;;

  (* The policy in the integer forms of the machine; the rules of the float sampler. The
     temper is log2(e) / T, and its Q is one below the Q of [Constants.log2e]. The extra
     bit is headroom for the temperature: the circuit multiplies by this constant on an
     18-bit signed port, thus the Q of [log2e] would overflow that port under a
     temperature of about 0.36, and this Q holds down to about 0.18. *)
  let policy ~temperature ~min_p =
    if Float.(temperature <= 0.0)
    then invalid_arg "Quantized: the temperature is positive";
    if Float.(min_p < 0.0 || min_p >= 1.0)
    then invalid_arg "Quantized: min_p is 0 up to 1";
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
    let rec grow e = if e < 14 && fits (e + 1) then grow (e + 1) else e in
    let rec shrink e = if fits e then e else shrink (e - 1) in
    if Float.(v <= 0.0) then 14 else shrink (grow 0)
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

  let of_floats (config : Transformer.Config.t) ~temperature ~min_p tensors =
    let temper, min_weight = policy ~temperature ~min_p in
    let { seats; phase; layers } : Tensor.floats Params_data.t =
      Params_data.of_list ~layers:config.layers tensors
    in
    let e = max_exponent (Float.max (max_abs seats) (max_abs phase)) in
    { config
    ; temper
    ; min_weight
    ; params =
        { Params_data.seats = quantize ~e seats
        ; phase = quantize ~e phase
        ; layers =
            Array.map layers ~f:(fun (l : Tensor.floats Params_data.layer) ->
              { Params_data.wq = quantize l.wq
              ; wk = quantize l.wk
              ; wv = quantize l.wv
              ; wo = quantize l.wo
              ; w1 = quantize l.w1
              ; w2 = quantize l.w2
              })
        }
    }
  ;;

  (* the draw the ear elected on 2026-08-18 *)
  let default_temperature = 1.0
  let default_min_p = 0.05

  let of_checkpoint
    ?(temperature = default_temperature)
    ?(min_p = default_min_p)
    (config : Transformer.Config.t)
    path
    =
    let archive = Nx_io.load_safetensors path in
    let tensor name =
      match Stdlib.Hashtbl.find_opt archive name with
      | None -> invalid_arg (Printf.sprintf "%s holds no tensor named %s" path name)
      | Some packed ->
        let t = Nx_io.to_typed Nx.float32 packed in
        let count = Array.fold (Nx.shape t) ~init:1 ~f:( * ) in
        Nx.to_array (Nx.reshape [| count |] t)
    in
    let tensors =
      List.mapi (sizes config) ~f:(fun index size ->
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
end

module Engine = struct
  (* everything in [t] is frozen: an update copies its spine or its row *)
  type t =
    { config : Transformer.Config.t
    ; p : Model.params
    ; temper : Constants.scale
    ; min_weight : int
    ; kc : Tensor.t array (* the K ring: [layer * slots + slot] rows of [d], Q12 int16 *)
    ; vc : Tensor.t array
    ; h : Tensor.t (* the residual stream after the last forwarded step, Q16 *)
    ; position : int (* one forward for each step, thus this counts the steps as well *)
    ; prng : Prng.state
    }

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

  (* exp2 of a Q12 value that is 0 or less: the integer part shifts, the top eight bits of
     the fraction index the table. The result is Q15, and the peak — input 0 — is 2^15. *)
  let exp2_q u =
    let n = -u in
    let i = n asr 12 in
    if i >= 16 then 0 else Constants.exp2_table.((n asr 4) land 255) asr i
  ;;

  (* rms_norm of the residual stream: the sum squares a Q12 copy of the stream — one
     DSP-sized product — then one isqrt, and one division for each element. The division
     is toward zero, as every division of the circuit. *)
  let rms_norm t (h : Tensor.t) : Tensor.t =
    let s = Array.fold h ~init:0 ~f:(fun acc x -> acc + ((x asr 4) * (x asr 4))) in
    (* the mean over [d] elements: the shift is log2 of the width *)
    let m = (s asr Int.floor_log2 t.config.Transformer.Config.d) + Constants.eps_q in
    let g = isqrt m in
    Array.map h ~f:(fun x -> clamp16 (x * 256 / g))
  ;;

  (* The row of one seat inside the seat tensor, which holds the four tables in one, seat
     0 first: the circuit reaches it with a shift and an add from the base of the tensor. *)
  let seat_row ~d ~seat ~index = ((seat * classes) + index) * d

  (* the embedding: the four seat rows and the phase row add in the shared exponent, then
     shift to Q16 *)
  let embed t ~frame ~phase : Tensor.t =
    let d = t.config.Transformer.Config.d in
    let drawn = Array.of_list (Vocab.classes_of_frame frame) in
    Array.init d ~f:(fun i ->
      let v =
        Array.foldi
          drawn
          ~init:t.p.phase.q.((phase * d) + i)
          ~f:(fun seat acc index -> acc + t.p.seats.q.(seat_row ~d ~seat ~index + i))
      in
      rescale ~from:t.p.seats.e ~target:Constants.h_q v)
  ;;

  (* The projections of one step: the query, the key row and the value row. One matvec
     column each; the circuit runs the three separately, on one MAC path. *)
  let projections t (lay : Model.layer) (y : Tensor.t) : Tensor.t * Tensor.t * Tensor.t =
    let d = t.config.Transformer.Config.d in
    let project (w : Model.quantized) o =
      let acc = sum d (fun i -> y.(i) * w.q.((i * d) + o)) in
      clamp16 (rescale ~from:(Constants.y_q + w.e) ~target:Constants.kv_q acc)
    in
    ( Array.init d ~f:(project lay.wq)
    , Array.init d ~f:(project lay.wk)
    , Array.init d ~f:(project lay.wv) )
  ;;

  (* Attention of layer [l] over the newest [n] steps of the rings: the merged context of
     the query [q], head by head. Age [a] reads slot [(cur - a) & (slots - 1)], thus the
     ALiBi distance is the age itself and the causal wall is the walk. *)
  let attend t (kc : Tensor.t array) (vc : Tensor.t array) ~l ~cur ~n (q : Tensor.t)
    : Tensor.t
    =
    let { Transformer.Config.d
        ; heads
        ; context = slots
        ; slope_span = span
        ; layers = (_ : int)
        }
      =
      t.config
    in
    let head_d = d / heads in
    (* the ring row that age [a] reads. The rows depend on neither the head nor the lane,
       thus they are named once: the weighted sum below would otherwise walk the ring for
       every (head, lane, age). *)
    let row ring a = ring.((l * slots) + ((cur - a) land (slots - 1))) in
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
      (* the exp2 weight of each age, Q15: the peak weighs 2^15 *)
      let age_weight =
        Array.init n ~f:(fun a ->
          exp2_q (Constants.apply Constants.log2e (scores.(a) - peak)))
      in
      let den = sum n (fun a -> age_weight.(a)) in
      Array.init head_d ~f:(fun j ->
        clamp16 (sum n (fun a -> age_weight.(a) * v_rows.(a).(hb + j)) / den))
    in
    (* head [k] gives the lanes [k * head_d] up to the next head; the heads are
       independent, thus the order of [List.init] moves nothing *)
    List.init heads ~f:head_context |> Array.concat
  ;;

  (* the feed-forward hidden vector of the normed stream [y]: one matvec and a ReLU, Q10 *)
  let hidden t (lay : Model.layer) (y : Tensor.t) : Tensor.t =
    let d = t.config.Transformer.Config.d in
    let dff = 4 * d in
    Array.init dff ~f:(fun o ->
      let acc = sum d (fun i -> y.(i) * lay.w1.q.((i * dff) + o)) in
      clamp16
        (max 0 (rescale ~from:(Constants.y_q + lay.w1.e) ~target:Constants.hid_q acc)))
  ;;

  (* a residual join: [values] times [w] lands on the stream [h] — the stream after; the
     exponent of [w] folds into the shift with [from], the format of [values] *)
  let join t (h : Tensor.t) (w : Model.quantized) ~(values : Tensor.t) ~len ~from
    : Tensor.t
    =
    let d = t.config.Transformer.Config.d in
    Array.mapi h ~f:(fun o above ->
      let acc = sum len (fun i -> values.(i) * w.q.((i * d) + o)) in
      above + rescale ~from:(from + w.e) ~target:Constants.h_q acc)
  ;;

  (* The ring keeps the top byte of a Q12 row: the circuit stores eight bits and restores
     eight zero low bits at the read, thus the granularity is 2^-4 and the format stays
     Q12. The query does not pass here — only the stored rows coarsen. *)
  (* [asr] and [lsl] associate to the right; the parentheses are the expression *)
  let coarse_to_ring (row : Tensor.t) : Tensor.t =
    Array.map row ~f:(fun v -> (v asr 8) lsl 8)
  ;;

  (* one step through the engine: the next engine *)
  let forward t ~frame ~phase =
    let d = t.config.Transformer.Config.d in
    let slots = t.config.Transformer.Config.context in
    let cur = t.position land (slots - 1) in
    let n = min (t.position + 1) slots in
    let kc = Array.copy t.kc in
    let vc = Array.copy t.vc in
    let layer l h (lay : Model.layer) =
      let y = rms_norm t h in
      let q, k_row, v_row = projections t lay y in
      kc.((l * slots) + cur) <- coarse_to_ring k_row;
      vc.((l * slots) + cur) <- coarse_to_ring v_row;
      let ctx = attend t kc vc ~l ~cur ~n q in
      let h = join t h lay.wo ~values:ctx ~len:d ~from:Constants.kv_q in
      let y = rms_norm t h in
      let hid = hidden t lay y in
      join t h lay.w2 ~values:hid ~len:(4 * d) ~from:Constants.hid_q
    in
    let h = Array.foldi t.p.layers ~init:(embed t ~frame ~phase) ~f:layer in
    { t with h; kc; vc; position = t.position + 1 }
  ;;

  (* The origin of a walk: an empty ring and no residual. The lead-in is not here — it is
     the first sixteen steps of the walk itself, thus [next_step] states it and a caller
     that counts steps counts the same steps the float sampler counts. *)
  let init (model : Model.t) ~seed =
    let { Model.config; params = p; temper; min_weight } = model in
    Model.check_shape model;
    let { Transformer.Config.d
        ; heads = (_ : int)
        ; context = slots
        ; layers
        ; slope_span = (_ : int)
        }
      =
      config
    in
    { config
    ; p
    ; temper
    ; min_weight
    ; (* a walk never reads an unwritten slot, thus one zero row serves them all *)
      kc = Array.create ~len:(layers * slots) (Array.create ~len:d 0)
    ; vc = Array.create ~len:(layers * slots) (Array.create ~len:d 0)
    ; h = Array.create ~len:d 0
    ; position = 0
    ; prng = Prng.create ~seed
    }
  ;;

  (* the tied head of one seat: rms_norm of the stream the chain has written so far, then
     that seat's table read backward; Q12 logits over the classes *)
  let seat_logits t (stream : Tensor.t) ~seat =
    let y = rms_norm t stream in
    let d = t.config.Transformer.Config.d in
    Array.init classes ~f:(fun index ->
      sum d (fun i -> y.(i) * t.p.seats.q.(seat_row ~d ~seat ~index + i)) asr t.p.seats.e)
  ;;

  (* what the chain adds after a seat draws: the drawn row, in the format of the stream *)
  let add_row t (stream : Tensor.t) ~seat ~index =
    let d = t.config.Transformer.Config.d in
    let base = seat_row ~d ~seat ~index in
    Array.mapi stream ~f:(fun i above ->
      above + rescale ~from:t.p.seats.e ~target:Constants.h_q t.p.seats.q.(base + i))
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
    (* the tempered weight of one class: exp2, and refused under min-p *)
    let weight c =
      let e = exp2_q (Constants.apply t.temper (logits.(c) - peak)) in
      if e >= t.min_weight then e else 0
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
    (* The boot of docs/transformer_model.md: a lead-in of silence, one bar of it, drawing
       nothing and taking no number from the generator. The model opens the music itself
       after it, thus the walk needs no pitch and no table to begin. *)
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

  (* the scalar rules the L0 circuit units must reproduce; their gate tests read them here
     rather than restate them *)
  module For_test = struct
    let isqrt = isqrt
    let exp2_q = exp2_q
  end
end

module Drift = struct
  type stats =
    { steps : int
    ; draws : int
    ; same_peak : int
    ; same_draw : int
    ; mean_cosine : float
    }

  (* the flat float array of one checkpoint tensor: what the quantization reads *)
  let flatten tensor =
    let count = Array.fold (Nx.shape tensor) ~init:1 ~f:( * ) in
    Nx.to_array (Nx.reshape [| count |] tensor)
  ;;

  (* One weights source and one policy: the walk quantizes the float tensors itself, under
     the draw of the era, thus the pairing cannot slip. The loop is state at the edge of
     the module, as the float sampler's loop is. *)
  let walk (config : Transformer.Config.t) params ~steps ~seed =
    let model =
      Model.of_floats
        config
        ~temperature:Model.default_temperature
        ~min_p:Model.default_min_p
        (List.map (Transformer.Params.to_list params) ~f:flatten)
    in
    let engine = ref (Engine.init model ~seed) in
    (* the history of the quantized walk, newest first: the float pass reads it, thus the
       two models are compared over one history and never over two *)
    let frames = ref [] in
    let positions = ref [] in
    let window history =
      List.take history config.Transformer.Config.context |> List.rev |> Array.of_list
    in
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
      (* the float logits of the same position, over the same chain: teacher-forcing
         inside the step, thus what the report measures is the quantization alone *)
      if not (List.is_empty chain)
      then (
        let floated =
          Transformer.logits
            config
            params
            ~frames:(window !frames)
            ~positions:(window !positions)
            ~drawn:(drawn_classes chain)
        in
        List.iter chain ~f:(fun (d : Engine.draw) ->
          let float_row = floated.(d.seat) in
          if Tensor.same_peak d.logits float_row then Int.incr same_peak;
          cosine_sum := !cosine_sum +. Tensor.cosine d.logits float_row;
          let float_class =
            Transformer.draw_class
              float_row
              ~temperature:Model.default_temperature
              ~min_p:Model.default_min_p
              ~uniform:d.uniform
          in
          if float_class = d.drawn then Int.incr same_draw;
          Int.incr draws));
      engine := next;
      frames := frame :: !frames;
      positions := step :: !positions
    done;
    { steps
    ; draws = !draws
    ; same_peak = !same_peak
    ; same_draw = !same_draw
    ; mean_cosine = !cosine_sum /. Float.of_int (max 1 !draws)
    }
  ;;
end

(* the shapes of a test model: small enough to run in a test, and the same structure *)
let test_config =
  { Transformer.Config.baseline with d = 32; layers = 1; heads = 2; context = 16 }
;;

(* a model of drawn weights: the tests read no file that git ignores *)
let test_model ~seed =
  let (_ : Prng.state), tensors =
    Prng.run
      (Prng.all
         (List.map (Model.sizes test_config) ~f:(fun count ->
            Prng.normals ~count ~scale:0.02)))
      (Prng.create_folded ~seed)
  in
  Model.of_floats
    test_config
    ~temperature:Model.default_temperature
    ~min_p:Model.default_min_p
    tensors
;;

let%expect_test "the exp2 table: the peak, the floor and the halving" =
  (* entry 0 is the peak 2^15; a full fractional step halves; the last entry sits one
     table step above one half *)
  Stdio.printf
    "%d %d %d  half at one: %d\n"
    Constants.exp2_table.(0)
    Constants.exp2_table.(128)
    Constants.exp2_table.(255)
    (Engine.exp2_q (-4096));
  [%expect {| 32768 23170 16428  half at one: 16384 |}]
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
  (* the lead-in is silence and takes no draw; every step after it takes four *)
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
    step 16  ceb0c5a4  4 draws
    step 17  c3adaeb7  4 draws
    step 18  c0b3adb7  4 draws
    step 19  c9bcb0bd  4 draws
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
  (* the order the draws happened in: the soprano first, and every seat one time *)
  Stdio.printf
    "%d drawn steps, the order %s\n"
    (List.length drawn)
    (Sexp.to_string
       ([%sexp_of: int list]
          (List.map (List.hd_exn drawn).draws ~f:(fun (d : Engine.draw) -> d.seat))));
  (* The class a seat drew is the class that seat holds in the frame. This is the join the
     chain could silently invert — the chain runs down and a frame reads up — thus the
     test states it and does not trust it. *)
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
      b5b0aeb2  drawn (15 11 13 18)  frame (15 11 13 18)
      bbb1c1b2  drawn (15 30 14 24)  frame (15 30 14 24)
      a6cabfb2  drawn (15 28 39 3)  frame (15 28 39 3)
      cbccb3ce  drawn (43 16 41 40)  frame (43 16 41 40)
    |}]
;;
