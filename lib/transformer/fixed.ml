open Base
module Params_data = Transformer.Params_data

(* The design constants of the fixed-point scheme: the formats of
   docs/transformer_rtl_proto.md, and the values derived from them and from the
   mathematics. The sampling policy is not here — the model carries it. *)
module Constants = struct
  (* the formats: the residual stream is Q16 in int32; the normed vector, the query, the
     keys, the values and the context are Q12 in int16; the FFN hidden is Q10 in int16;
     the scores and the logits are Q12 and stay wide *)
  let h_q = 16
  let y_q = 12
  let kv_q = 12
  let hid_q = 10

  (* the rms epsilon of the float model, in the Q of the squared stream: the sum squares a
     Q12 copy, thus the mean is Q(2 y_q) *)
  let eps_q = Float.iround_nearest_exn (Float.ldexp 1e-6 (2 * y_q))

  (* log2(e) in Q15: the exp2 form of the softmax exponent *)
  let log2e_q15 = Float.iround_nearest_exn (32768.0 /. Float.log 2.0)

  (* the quantized exponential: exp2 of -j/256 in Q15 — the table of the softmax and the
     sampler, and the same species as the weights *)
  let exp2_table =
    Array.init 256 ~f:(fun j ->
      Float.iround_nearest_exn Float.(32768.0 * (2.0 ** (-of_int j / 256.0))))
  ;;

  let exp2_bits = Array.map exp2_table ~f:(Hardcaml.Bits.of_unsigned_int ~width:16)
end

type float_tensor = float array

(* the quantization arithmetic of one tensor: pure functions from the float values to the
   int8 form *)
module Tensor = struct
  (* one weight tensor: w ~ q * 2^-e *)
  type t =
    { q : int array
    ; e : int
    }

  let max_abs (tensor : float_tensor) =
    Array.fold tensor ~init:0.0 ~f:(fun acc v -> Float.max acc (Float.abs v))
  ;;

  (* the largest exponent that keeps round(max|w| * 2^e) at 127 or less; 14 caps the
     all-zero tensor *)
  let max_exponent v =
    let fits e = Float.iround_nearest_exn (Float.ldexp v e) <= 127 in
    let rec grow e = if e < 14 && fits (e + 1) then grow (e + 1) else e in
    let rec shrink e = if fits e then e else shrink (e - 1) in
    if Float.(v <= 0.0) then 14 else shrink (grow 0)
  ;;

  (* [e] overrides the exponent of the tensor's own peak — the tables share one, because
     their rows add *)
  let quantize ?e (tensor : float_tensor) =
    let e = Option.value e ~default:(max_exponent (max_abs tensor)) in
    let clamp ft =
      Int.clamp_exn (Float.iround_nearest_exn (Float.ldexp ft e)) ~min:(-127) ~max:127
    in
    { q = Array.map tensor ~f:clamp; e }
  ;;
end

(* The quantized model: the configuration and the tensors quantized under it, one value —
   the unit the engine and the circuit consume. The pairing invariant lives here: after
   the constructor, no caller can mispair a configuration with another model's tensors. *)
module Model = struct
  (* the structure of [Params_data], over the quantized tensor: one definition holds the
     shape and the flat order of the checkpoint *)
  type params = Tensor.t Params_data.t
  type layer = Tensor.t Params_data.layer

  type t =
    { config : Transformer.Config.t
    ; params : params
    ; temper_q14 : int
    (** the sampling temper, log2(e) / T in Q14 — folded with the exp2 form *)
    ; min_weight : int (** the min-p share of the peak weight 2^15 *)
    }

  (* the element counts of the tensors in the flat order, from the one definition of the
     shapes *)
  let sizes config =
    List.map (Transformer.Params.shapes config) ~f:(fun shape ->
      Array.fold shape ~init:1 ~f:( * ))
  ;;

  (* The base of each tensor inside the ROM, in the shape of the parameters: the running
     sums of the sizes, handed back through the one definition of the flat order. *)
  let rom_bases t =
    List.folding_map (sizes t.config) ~init:0 ~f:(fun start size -> start + size, start)
    |> Params_data.of_list ~layers:t.config.Transformer.Config.layers
  ;;

  (* The ROM image of the circuit: every tensor in the checkpoint order, one byte for each
     weight, two's complement, padded to the next power of two — the depth of the address. *)
  let rom_bits t =
    let image =
      Array.concat_map
        (Array.of_list (Params_data.to_list t.params))
        ~f:(fun { q; e = (_ : int) } -> Array.map q ~f:(fun v -> v land 255))
    in
    let depth = Int.ceil_pow2 (Array.length image) in
    Array.init depth ~f:(fun at ->
      if at < Array.length image
      then Hardcaml.Bits.of_unsigned_int ~width:8 image.(at)
      else Hardcaml.Bits.zero 8)
  ;;

  (* the policy in the integer forms of the machine; the rules of the float sampler *)
  let policy ~temperature ~min_p =
    if Float.(temperature <= 0.0) then invalid_arg "Fixed: the temperature is positive";
    if Float.(min_p < 0.0 || min_p >= 1.0) then invalid_arg "Fixed: min_p is 0 up to 1";
    ( Float.iround_nearest_exn (32768.0 /. Float.log 2.0 /. temperature /. 2.0)
    , Float.iround_nearest_exn (min_p *. 32768.0) )
  ;;

  (* the three tables add row for row, thus they share one exponent *)
  let of_floats (config : Transformer.Config.t) ~temperature ~min_p tensors =
    let temper_q14, min_weight = policy ~temperature ~min_p in
    let { embed; phase; progress; layers } : float_tensor Params_data.t =
      Params_data.of_list ~layers:config.layers tensors
    in
    let e =
      Tensor.max_exponent
        (Float.max
           (Tensor.max_abs embed)
           (Float.max (Tensor.max_abs phase) (Tensor.max_abs progress)))
    in
    { config
    ; temper_q14
    ; min_weight
    ; params =
        { Params_data.embed = Tensor.quantize ~e embed
        ; phase = Tensor.quantize ~e phase
        ; progress = Tensor.quantize ~e progress
        ; layers =
            Array.map layers ~f:(fun (l : float_tensor Params_data.layer) ->
              { Params_data.wq = Tensor.quantize l.wq
              ; wk = Tensor.quantize l.wk
              ; wv = Tensor.quantize l.wv
              ; wo = Tensor.quantize l.wo
              ; w1 = Tensor.quantize l.w1
              ; w2 = Tensor.quantize l.w2
              })
        }
    }
  ;;

  (* the settled sampling defaults of the era *)
  let default_temperature = 0.9
  let default_min_p = 1.0 /. 256.0

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

  module For_test = struct
    let init
      ?(temperature = default_temperature)
      ?(min_p = default_min_p)
      (config : Transformer.Config.t)
      ~seed
      =
      let (_ : Prng.state), tensors =
        Prng.run
          (Prng.all
             (List.map (sizes config) ~f:(fun count -> Prng.normals ~count ~scale:0.02)))
          (Prng.create_folded ~seed)
      in
      of_floats config ~temperature ~min_p tensors
    ;;
  end
end

module Engine = struct
  (* the peak magnitudes before any clamp: the calibration of the circuit widths *)
  type peaks =
    { mutable h : int
    ; mutable rms_sum : int
    ; mutable y : int
    ; mutable qkv : int
    ; mutable score : int
    ; mutable ctx : int
    ; mutable hid : int
    ; mutable logit : int
    }

  (* The engine as a value: an operation gives the next engine, and the arrays of [t] are
     frozen — an update copies its spine or row. The one exception is [pk], the meter of
     the walk: engines of one walk share it. *)
  type t =
    { config : Transformer.Config.t
    ; p : Model.params
    ; temper_q14 : int
    ; min_weight : int
    ; kc : int array array (* the K ring: [layer * slots + slot] rows of [d], Q12 int16 *)
    ; vc : int array array
    ; h : int array (* the residual stream after the last forwarded token, Q16 *)
    ; position : int
    ; prng : Prng.state
    ; sounding : Sounding_state.t
    ; seats : int option array
    ; step_index : int
    ; pk : peaks
    }

  (* one socket event of a drawn sentence: the voice, the pitch, and On or Off *)
  type event =
    { voice : int
    ; pitch : int
    ; on : bool
    }
  [@@deriving sexp_of]

  let vocab = Token.vocab

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

  let note_peak { pk } field v =
    let v = abs v in
    match field with
    | `H -> if v > pk.h then pk.h <- v
    | `Rms -> if v > pk.rms_sum then pk.rms_sum <- v
    | `Y -> if v > pk.y then pk.y <- v
    | `Qkv -> if v > pk.qkv then pk.qkv <- v
    | `Score -> if v > pk.score then pk.score <- v
    | `Ctx -> if v > pk.ctx then pk.ctx <- v
    | `Hid -> if v > pk.hid then pk.hid <- v
    | `Logit -> if v > pk.logit then pk.logit <- v
  ;;

  (* rms_norm of the residual stream: the sum squares a Q12 copy of the stream — one
     DSP-sized product — then one isqrt, and one division for each element. The division
     is toward zero, as every division of the circuit. *)
  let rms_norm t h =
    let s = Array.fold h ~init:0 ~f:(fun acc x -> acc + ((x asr 4) * (x asr 4))) in
    note_peak t `Rms s;
    (* the mean over [d] elements: the shift is log2 of the width *)
    let m = (s asr Int.floor_log2 t.config.Transformer.Config.d) + Constants.eps_q in
    let g = isqrt m in
    Array.map h ~f:(fun x ->
      let y = x * 256 / g in
      note_peak t `Y y;
      clamp16 y)
  ;;

  (* the embedding: the three rows add in the shared exponent, then shift to Q16 *)
  let embed t ~code ~phase ~bucket =
    let d = t.config.Transformer.Config.d in
    Array.init d ~f:(fun i ->
      let v =
        t.p.embed.q.((code * d) + i)
        + t.p.phase.q.((phase * d) + i)
        + t.p.progress.q.((bucket * d) + i)
      in
      rescale ~from:t.p.embed.e ~target:Constants.h_q v)
  ;;

  (* The projections of one token: the query, the key row and the value row. One matvec
     column each; the circuit runs the three separately, on one MAC path. *)
  let projections t (lay : Model.layer) y =
    let d = t.config.Transformer.Config.d in
    let project (w : Tensor.t) o =
      let acc = sum d (fun i -> y.(i) * w.q.((i * d) + o)) in
      let v = rescale ~from:(Constants.y_q + w.e) ~target:Constants.kv_q acc in
      note_peak t `Qkv v;
      clamp16 v
    in
    ( Array.init d ~f:(project lay.wq)
    , Array.init d ~f:(project lay.wk)
    , Array.init d ~f:(project lay.wv) )
  ;;

  (* Attention of layer [l] over the newest [n] tokens, one head at a time: the merged
     context of the query [q]. Age [a] reads slot [(cur - a) & 255], thus the ALiBi
     distance is the age itself and the causal wall is the walk. *)
  let attend t kc vc ~l ~cur ~n q =
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
    let scores = Array.create ~len:n 0 in
    let ctx = Array.create ~len:d 0 in
    (* the ring row that age [a] reads *)
    let row ring a = ring.((l * slots) + ((cur - a) land (slots - 1))) in
    for head = 0 to heads - 1 do
      let hb = head * head_d in
      let slope_exponent = span * (head + 1) / heads in
      (* Q(2 kv_q) to Q12, then the 1/sqrt(head_d) of the reference — a shift, thus head_d
         is a power of four *)
      let score_shift =
        (2 * Constants.kv_q) - Constants.y_q + (Int.floor_log2 head_d / 2)
      in
      let score a =
        let k = row kc a in
        (sum head_d (fun j -> q.(hb + j) * k.(hb + j)) asr score_shift)
        - (a lsl (Constants.y_q - slope_exponent))
      in
      for a = 0 to n - 1 do
        let s = score a in
        note_peak t `Score s;
        scores.(a) <- s
      done;
      let peak = max_over n (fun a -> scores.(a)) in
      (* the exp2 weight of each age, Q15: the peak weighs 2^15 *)
      let age_weight =
        Array.init n ~f:(fun a ->
          exp2_q (((scores.(a) - peak) * Constants.log2e_q15) asr 15))
      in
      let den = sum n (fun a -> age_weight.(a)) in
      for j = 0 to head_d - 1 do
        let c = sum n (fun a -> age_weight.(a) * (row vc a).(hb + j)) / den in
        note_peak t `Ctx c;
        ctx.(hb + j) <- clamp16 c
      done
    done;
    ctx
  ;;

  (* the feed-forward hidden vector of the normed stream [y]: one matvec and a ReLU, Q10 *)
  let hidden t (lay : Model.layer) y =
    let d = t.config.Transformer.Config.d in
    let dff = 4 * d in
    Array.init dff ~f:(fun o ->
      let acc = sum d (fun i -> y.(i) * lay.w1.q.((i * dff) + o)) in
      let v =
        max 0 (rescale ~from:(Constants.y_q + lay.w1.e) ~target:Constants.hid_q acc)
      in
      note_peak t `Hid v;
      clamp16 v)
  ;;

  (* a residual join: [values] times [w] adds into the stream [h]; [from] is the format of
     [values], and the exponent of [w] folds into the shift *)
  let join t h (w : Tensor.t) ~values ~len ~from =
    let d = t.config.Transformer.Config.d in
    for o = 0 to d - 1 do
      let acc = sum len (fun i -> values.(i) * w.q.((i * d) + o)) in
      let v = h.(o) + rescale ~from:(from + w.e) ~target:Constants.h_q acc in
      note_peak t `H v;
      h.(o) <- v
    done
  ;;

  (* one token through the engine: the next engine *)
  let forward t ~code ~phase ~bucket =
    let d = t.config.Transformer.Config.d in
    let slots = t.config.Transformer.Config.context in
    let cur = t.position land (slots - 1) in
    let n = min (t.position + 1) slots in
    let h = embed t ~code ~phase ~bucket in
    let kc = Array.copy t.kc in
    let vc = Array.copy t.vc in
    Array.iteri t.p.layers ~f:(fun l lay ->
      let y = rms_norm t h in
      let q, k_row, v_row = projections t lay y in
      kc.((l * slots) + cur) <- k_row;
      vc.((l * slots) + cur) <- v_row;
      let ctx = attend t kc vc ~l ~cur ~n q in
      join t h lay.wo ~values:ctx ~len:d ~from:Constants.kv_q;
      let y = rms_norm t h in
      let hid = hidden t lay y in
      join t h lay.w2 ~values:hid ~len:(4 * d) ~from:Constants.hid_q);
    { t with
      h
    ; kc
    ; vc
    ; position = t.position + 1
    ; (* the model state of the token lands with the token: the mask of the next draw can
         never run ahead of or behind the engine *)
      sounding = Sounding_state.step t.sounding (Token.of_code code)
    }
  ;;

  let init (model : Model.t) ~seed =
    let { Model.config; params = p; temper_q14; min_weight } = model in
    let { Transformer.Config.d; heads; context = slots; layers; slope_span = (_ : int) } =
      config
    in
    (* the arithmetic of the circuit is shifts, thus the shape obeys the shift rules *)
    assert (Int.is_pow2 d);
    assert (Int.is_pow2 slots);
    assert (Int.floor_log2 (d / heads) % 2 = 0);
    assert (layers = Array.length p.Params_data.layers);
    let engine =
      { config
      ; p
      ; temper_q14
      ; min_weight
      ; (* a walk never reads an unwritten slot, thus one zero row serves them all *)
        kc = Array.create ~len:(layers * slots) (Array.create ~len:d 0)
      ; vc = Array.create ~len:(layers * slots) (Array.create ~len:d 0)
      ; h = Array.create ~len:d 0
      ; position = 0
      ; prng = Prng.create ~seed
      ; sounding = Sounding_state.silence
      ; seats = Array.create ~len:Token.seats None
      ; step_index = 0
      ; pk =
          { h = 0; rms_sum = 0; y = 0; qkv = 0; score = 0; ctx = 0; hid = 0; logit = 0 }
      }
    in
    forward engine ~code:(Token.to_code Token.Start) ~phase:0 ~bucket:0
  ;;

  (* the tied head: rms_norm, then the token table read backward; Q12 logits *)
  let logits t =
    let y = rms_norm t t.h in
    let d = t.config.Transformer.Config.d in
    let e = t.p.embed.e in
    Array.init vocab ~f:(fun c ->
      let l = sum d (fun i -> y.(i) * t.p.embed.q.((c * d) + i)) asr e in
      note_peak t `Logit l;
      l)
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

  let next_code t =
    let logits = logits t in
    let mask = Sounding_state.legal_mask t.sounding in
    let peak = max_over vocab (fun c -> if mask.(c) then logits.(c) else Int.min_value) in
    (* the tempered weight of one code: masked, exp2, and refused under min-p *)
    let weight c =
      if mask.(c)
      then (
        let u = ((logits.(c) - peak) * t.temper_q14) asr 14 in
        let e = exp2_q u in
        if e >= t.min_weight then e else 0)
      else 0
    in
    let weights = Array.init vocab ~f:weight in
    let total = sum vocab (fun c -> weights.(c)) in
    let prng, u = u24 t.prng in
    let threshold = (u * total) asr 24 in
    (* the code whose running total passes the threshold; the fallback of the float
       sampler when no weight remains on the walk *)
    let rec walk c total =
      if c = vocab - 1
      then c
      else (
        let total = total + weights.(c) in
        if total > threshold then c else walk (c + 1) total)
    in
    let chosen = walk 0 0 in
    { t with prng }, if weights.(chosen) > 0 then chosen else 0
  ;;

  (* an On takes the highest free seat — the melody sits high; an Off names the seat that
     holds its pitch. The mask guarantees both exist. *)
  let highest_free seats =
    let rec go s =
      assert (s >= 0);
      if Option.is_none seats.(s) then s else go (s - 1)
    in
    go (Token.seats - 1)
  ;;

  let seat_of seats pitch =
    let rec go s =
      assert (s >= 0);
      match seats.(s) with
      | Some held when held = pitch -> s
      | _ -> go (s - 1)
    in
    go (Token.seats - 1)
  ;;

  let next_step t =
    let phase = t.step_index % Transformer.phase_buckets in
    let bucket =
      t.step_index / Transformer.progress_stride % Transformer.progress_buckets
    in
    let sit t seat holds =
      let seats = Array.copy t.seats in
      seats.(seat) <- holds;
      { t with seats }
    in
    let rec go t events count =
      (* the mask bounds a sentence at four Offs, four Ons and the End *)
      assert (count < 16);
      let t, code = next_code t in
      let token = Token.of_code code in
      let t = forward t ~code ~phase ~bucket in
      match token with
      | Start -> assert false
      | End -> { t with step_index = t.step_index + 1 }, List.rev events
      | On pitch ->
        let seat = highest_free t.seats in
        go
          (sit t seat (Some pitch))
          ({ voice = seat; pitch; on = true } :: events)
          (count + 1)
      | Off pitch ->
        let seat = seat_of t.seats pitch in
        go (sit t seat None) ({ voice = seat; pitch; on = false } :: events) (count + 1)
    in
    go t [] 0
  ;;

  let peaks { pk } =
    [ "h", pk.h
    ; "rms_sum", pk.rms_sum
    ; "y", pk.y
    ; "qkv", pk.qkv
    ; "score", pk.score
    ; "ctx", pk.ctx
    ; "hid", pk.hid
    ; "logit", pk.logit
    ]
  ;;
end

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
  Stdio.printf
    "%d %d %d %d %d\n"
    (Engine.isqrt 0)
    (Engine.isqrt 15)
    (Engine.isqrt 16)
    (Engine.isqrt 4295)
    (Engine.isqrt (1 lsl 50));
  [%expect {| 0 3 4 65 33554432 |}]
;;

(* The walk with drawn weights: the music is noise, but the grammar and the seats must
   hold, and the same seed must repeat. The replay walks the events back through
   [Sounding_state], as the sampler test of [Transformer] does. *)
(* the fold threads the engine through the steps in the drawn order *)
let collect_steps n engine =
  let (_ : Engine.t), steps =
    List.fold (List.range 0 n) ~init:(engine, []) ~f:(fun (engine, acc) (_ : int) ->
      let engine, events = Engine.next_step engine in
      engine, events :: acc)
  in
  List.rev steps
;;

let%expect_test "a drawn walk keeps the grammar, the seats and the seed" =
  let model = Model.For_test.init Transformer.Config.baseline ~seed:11 in
  let engine = Engine.init model ~seed:42 in
  let steps = collect_steps 12 engine in
  let replay (state, violations) { Engine.voice = (_ : int); pitch; on } =
    let token = if on then Token.On pitch else Token.Off pitch in
    let legal = Sounding_state.legal_mask state in
    let violations = violations + Bool.to_int (not legal.(Token.to_code token)) in
    Sounding_state.step state token, violations
  in
  let close state = Sounding_state.step state Token.End in
  let (_ : Sounding_state.t), violations =
    List.fold steps ~init:(Sounding_state.silence, 0) ~f:(fun acc events ->
      let state, violations = List.fold events ~init:acc ~f:replay in
      close state, violations)
  in
  let events = List.length (List.concat steps) in
  List.iter
    (List.take (List.filter steps ~f:(Fn.non List.is_empty)) 4)
    ~f:(fun step -> Stdio.print_s ([%sexp_of: Engine.event list] step));
  let again =
    let engine = Engine.init model ~seed:42 in
    collect_steps 12 engine
  in
  Stdio.printf
    "12 steps  %d events  %d illegal  the seed repeats: %b\n"
    events
    violations
    ([%compare.equal: (int * int * bool) list list]
       (List.map
          steps
          ~f:(List.map ~f:(fun { Engine.voice; pitch; on } -> voice, pitch, on)))
       (List.map
          again
          ~f:(List.map ~f:(fun { Engine.voice; pitch; on } -> voice, pitch, on))));
  [%expect
    {|
    (((voice 3) (pitch 17) (on true)) ((voice 2) (pitch 12) (on true))
     ((voice 1) (pitch 9) (on true)) ((voice 0) (pitch 1) (on true)))
    (((voice 3) (pitch 17) (on false)) ((voice 3) (pitch 113) (on true)))
    (((voice 0) (pitch 1) (on false)) ((voice 0) (pitch 101) (on true)))
    (((voice 3) (pitch 113) (on false)) ((voice 3) (pitch 10) (on true)))
    12 steps  22 events  0 illegal  the seed repeats: true
    |}]
;;
