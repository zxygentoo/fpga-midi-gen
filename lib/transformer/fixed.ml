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
end

(* the quantization arithmetic of one tensor: pure functions from the float values to the
   int8 form *)
module Tensor = struct
  type t =
    { q : int array
    ; e : int
    }

  (* the quantized exponential: exp2 of -j/256 in Q15 — the table of the softmax and the
     sampler, and the same species as the weights *)
  let exp2_table =
    Array.init 256 ~f:(fun j ->
      Float.iround_nearest_exn Float.(32768.0 * (2.0 ** (-of_int j / 256.0))))
  ;;

  let exp2_bits = Array.map exp2_table ~f:(Hardcaml.Bits.of_unsigned_int ~width:16)

  let maxabs values =
    Array.fold values ~init:0.0 ~f:(fun acc v -> Float.max acc (Float.abs v))
  ;;

  (* the largest exponent that keeps round(max|w| * 2^e) at 127 or less; 14 caps the
     all-zero tensor *)
  let exponent_for maxabs =
    let fits e = Float.iround_nearest_exn (Float.ldexp maxabs e) <= 127 in
    let rec grow e = if e < 14 && fits (e + 1) then grow (e + 1) else e in
    let rec shrink e = if fits e then e else shrink (e - 1) in
    if Float.(maxabs <= 0.0) then 14 else shrink (grow 0)
  ;;

  (* [e] overrides the exponent of the tensor's own peak — the tables share one, because
     their rows add *)
  let quantize ?e values =
    let e = Option.value e ~default:(exponent_for (maxabs values)) in
    { q =
        Array.map values ~f:(fun v ->
          Int.clamp_exn (Float.iround_nearest_exn (Float.ldexp v e)) ~min:(-127) ~max:127)
    ; e
    }
  ;;
end

(* The quantized model: the configuration and the tensors quantized under it, one value —
   the unit the engine and the circuit consume. The pairing invariant lives here: after
   the constructor, no caller can mispair a configuration with another model's tensors. *)
module Model = struct
  type tensor = Tensor.t =
    { q : int array
    ; e : int
    }

  (* the structure of [Params_data], over the quantized tensor: one definition holds the
     shape and the flat order of the checkpoint *)
  type params = tensor Params_data.t
  type layer = tensor Params_data.layer

  type t =
    { config : Transformer.Config.t
    ; params : params
    ; temper_q14 : int
    (** the sampling temper, log2(e) / T in Q14 — folded with the exp2 form *)
    ; min_weight : int (** the min-p share of the peak weight 2^15 *)
    }

  let exp2_bits = Tensor.exp2_bits

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
        ~f:(fun { Tensor.q; e = (_ : int) } -> Array.map q ~f:(fun v -> v land 255))
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
    let floats : float array Params_data.t =
      Params_data.of_list ~layers:config.layers tensors
    in
    let e =
      Tensor.exponent_for
        (Float.max
           (Tensor.maxabs floats.embed)
           (Float.max (Tensor.maxabs floats.phase) (Tensor.maxabs floats.progress)))
    in
    { config
    ; temper_q14
    ; min_weight
    ; params =
        { Params_data.embed = Tensor.quantize ~e floats.embed
        ; phase = Tensor.quantize ~e floats.phase
        ; progress = Tensor.quantize ~e floats.progress
        ; layers =
            Array.map floats.layers ~f:(fun (l : float array Params_data.layer) ->
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
    if i >= 16 then 0 else Tensor.exp2_table.((n asr 4) land 255) asr i
  ;;

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

  type t =
    { config : Transformer.Config.t
    ; p : Model.params
    ; temper_q14 : int
    ; min_weight : int
    ; kc : int array (* [layer; slot; dim], Q12 int16 *)
    ; vc : int array
    ; h : int array (* the residual stream, Q16 *)
    ; y : int array (* the normed vector; the merged context reuses it *)
    ; q : int array
    ; scores : int array (* one head at a time; the FFN hidden reuses it *)
    ; logits : int array
    ; weights : int array
    ; mutable fed : int
    ; mutable prng : Prng.state
    ; mutable sounding : Sounding_state.t
    ; seats : int option array
    ; mutable step_index : int
    ; pk : peaks
    }

  let note_peak t field v =
    let v = abs v in
    match field with
    | `H -> if v > t.pk.h then t.pk.h <- v
    | `Rms -> if v > t.pk.rms_sum then t.pk.rms_sum <- v
    | `Y -> if v > t.pk.y then t.pk.y <- v
    | `Qkv -> if v > t.pk.qkv then t.pk.qkv <- v
    | `Score -> if v > t.pk.score then t.pk.score <- v
    | `Ctx -> if v > t.pk.ctx then t.pk.ctx <- v
    | `Hid -> if v > t.pk.hid then t.pk.hid <- v
    | `Logit -> if v > t.pk.logit then t.pk.logit <- v
  ;;

  (* rms_norm of the residual stream into [y]: the sum squares a Q12 copy of the stream —
     one DSP-sized product — then one isqrt, and one division for each element. The
     division is toward zero, as every division of the circuit. *)
  let rms_into t =
    let s = Array.fold t.h ~init:0 ~f:(fun acc x -> acc + ((x asr 4) * (x asr 4))) in
    note_peak t `Rms s;
    (* the mean over [d] elements: the shift is log2 of the width *)
    let m = (s asr Int.floor_log2 t.config.Transformer.Config.d) + Constants.eps_q in
    let g = isqrt m in
    Array.iteri t.h ~f:(fun i x ->
      let y = x * 256 / g in
      note_peak t `Y y;
      t.y.(i) <- clamp16 y)
  ;;

  let feed t ~code ~phase ~bucket =
    let p = t.p in
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
    (* the embedding: the three rows add in the shared exponent, then shift to Q16 *)
    for i = 0 to d - 1 do
      let v =
        p.embed.q.((code * d) + i)
        + p.phase.q.((phase * d) + i)
        + p.progress.q.((bucket * d) + i)
      in
      t.h.(i) <- rescale ~from:p.embed.e ~target:Constants.h_q v
    done;
    let cur = t.fed land (slots - 1) in
    let n = min (t.fed + 1) slots in
    Array.iteri p.layers ~f:(fun l (lay : Model.layer) ->
      rms_into t;
      (* the projections; k and v enter the ring at the slot of this token *)
      let ring_base = ((l * slots) + cur) * d in
      for o = 0 to d - 1 do
        (* one matvec column; the circuit runs the three separately, on one MAC path *)
        let project (w : Tensor.t) =
          let acc = sum d (fun i -> t.y.(i) * w.q.((i * d) + o)) in
          let v = rescale ~from:(Constants.y_q + w.e) ~target:Constants.kv_q acc in
          note_peak t `Qkv v;
          clamp16 v
        in
        t.q.(o) <- project lay.wq;
        t.kc.(ring_base + o) <- project lay.wk;
        t.vc.(ring_base + o) <- project lay.wv
      done;
      (* attention, one head at a time; age [a] reads slot [(cur - a) & 255], thus the
         ALiBi distance is the age itself and the causal wall is the walk *)
      for head = 0 to heads - 1 do
        let hb = head * head_d in
        let slope_exponent = span * (head + 1) / heads in
        (* the base of the ring slot that age [a] reads, at this head *)
        let slot_base a = (((l * slots) + ((cur - a) land (slots - 1))) * d) + hb in
        (* Q(2 kv_q) to Q12, then the 1/sqrt(head_d) of the reference — a shift, thus
           head_d is a power of four *)
        let score_shift =
          (2 * Constants.kv_q) - Constants.y_q + (Int.floor_log2 head_d / 2)
        in
        let score a =
          let sb = slot_base a in
          (sum head_d (fun j -> t.q.(hb + j) * t.kc.(sb + j)) asr score_shift)
          - (a lsl (Constants.y_q - slope_exponent))
        in
        for a = 0 to n - 1 do
          let s = score a in
          note_peak t `Score s;
          t.scores.(a) <- s
        done;
        let peak = max_over n (fun a -> t.scores.(a)) in
        (* the exp2 weight of each age, Q15: the peak weighs 2^15 *)
        let age_weight =
          Array.init n ~f:(fun a ->
            exp2_q (((t.scores.(a) - peak) * Constants.log2e_q15) asr 15))
        in
        let den = sum n (fun a -> age_weight.(a)) in
        (* the merged context overwrites [y]: the projections took what they needed *)
        for j = 0 to head_d - 1 do
          let c = sum n (fun a -> age_weight.(a) * t.vc.(slot_base a + j)) / den in
          note_peak t `Ctx c;
          t.y.(hb + j) <- clamp16 c
        done
      done;
      (* the attention branch joins the stream *)
      for o = 0 to d - 1 do
        let acc = sum d (fun i -> t.y.(i) * lay.wo.q.((i * d) + o)) in
        let v =
          t.h.(o) + rescale ~from:(Constants.kv_q + lay.wo.e) ~target:Constants.h_q acc
        in
        note_peak t `H v;
        t.h.(o) <- v
      done;
      (* the feed-forward; the hidden vector reuses the score RAM *)
      rms_into t;
      let dff = 4 * d in
      for o = 0 to dff - 1 do
        let acc = sum d (fun i -> t.y.(i) * lay.w1.q.((i * dff) + o)) in
        let v =
          max 0 (rescale ~from:(Constants.y_q + lay.w1.e) ~target:Constants.hid_q acc)
        in
        note_peak t `Hid v;
        t.scores.(o) <- clamp16 v
      done;
      for o = 0 to d - 1 do
        let acc = sum dff (fun i -> t.scores.(i) * lay.w2.q.((i * d) + o)) in
        let v =
          t.h.(o) + rescale ~from:(Constants.hid_q + lay.w2.e) ~target:Constants.h_q acc
        in
        note_peak t `H v;
        t.h.(o) <- v
      done);
    t.fed <- t.fed + 1;
    (* the model state of the token lands with the token: the mask of the next draw can
       never run ahead of or behind the engine *)
    t.sounding <- Sounding_state.step t.sounding (Token.of_code code)
  ;;

  (* the tied head: rms_norm, then the token table read backward; Q12 logits *)
  let next_logits t =
    rms_into t;
    let d = t.config.Transformer.Config.d in
    let e = t.p.embed.e in
    for c = 0 to vocab - 1 do
      let l = sum d (fun i -> t.y.(i) * t.p.embed.q.((c * d) + i)) asr e in
      note_peak t `Logit l;
      t.logits.(c) <- l
    done;
    Array.copy t.logits
  ;;

  (* three PRNG bytes, high first: the walk of [Prng.uniform] *)
  let u24 t =
    let open Prng in
    let prng, draw =
      run
        (let* high = next in
         let* middle = next in
         let+ low = next in
         (((high * 256) + middle) * 256) + low)
        t.prng
    in
    t.prng <- prng;
    draw
  ;;

  let draw_code t =
    let (_ : int array) = next_logits t in
    let mask = Sounding_state.legal_mask t.sounding in
    let peak =
      max_over vocab (fun c -> if mask.(c) then t.logits.(c) else Int.min_value)
    in
    (* the tempered weight of one code: masked, exp2, and refused under min-p *)
    let weight c =
      if mask.(c)
      then (
        let u = ((t.logits.(c) - peak) * t.temper_q14) asr 14 in
        let e = exp2_q u in
        if e >= t.min_weight then e else 0)
      else 0
    in
    for c = 0 to vocab - 1 do
      t.weights.(c) <- weight c
    done;
    let total = sum vocab (fun c -> t.weights.(c)) in
    let threshold = (u24 t * total) asr 24 in
    (* the code whose running total passes the threshold; the fallback of the float
       sampler when no weight remains on the walk *)
    let rec walk c total =
      if c = vocab - 1
      then c
      else (
        let total = total + t.weights.(c) in
        if total > threshold then c else walk (c + 1) total)
    in
    let chosen = walk 0 0 in
    if t.weights.(chosen) > 0 then chosen else 0
  ;;

  let create (model : Model.t) ~seed =
    let { Model.config; params = p; temper_q14; min_weight } = model in
    let { Transformer.Config.d; heads; context = slots; layers; slope_span = (_ : int) } =
      config
    in
    (* the arithmetic of the circuit is shifts, thus the shape obeys the shift rules *)
    assert (Int.is_pow2 d);
    assert (Int.is_pow2 slots);
    assert (Int.floor_log2 (d / heads) % 2 = 0);
    assert (layers = Array.length p.Params_data.layers);
    let t =
      { config
      ; p
      ; temper_q14
      ; min_weight
      ; kc = Array.create ~len:(layers * slots * d) 0
      ; vc = Array.create ~len:(layers * slots * d) 0
      ; h = Array.create ~len:d 0
      ; y = Array.create ~len:d 0
      ; q = Array.create ~len:d 0
      ; (* one array serves the scores and the feed-forward hidden vector *)
        scores = Array.create ~len:(max slots (4 * d)) 0
      ; logits = Array.create ~len:vocab 0
      ; weights = Array.create ~len:vocab 0
      ; fed = 0
      ; prng = Prng.create ~seed
      ; sounding = Sounding_state.silence
      ; seats = Array.create ~len:Token.seats None
      ; step_index = 0
      ; pk =
          { h = 0; rms_sum = 0; y = 0; qkv = 0; score = 0; ctx = 0; hid = 0; logit = 0 }
      }
    in
    feed t ~code:(Token.to_code Token.Start) ~phase:0 ~bucket:0;
    t
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

  let step_events t =
    let phase = t.step_index % Transformer.phase_buckets in
    let bucket =
      t.step_index / Transformer.progress_stride % Transformer.progress_buckets
    in
    let rec go events count =
      (* the mask bounds a sentence at four Offs, four Ons and the End *)
      assert (count < 16);
      let code = draw_code t in
      let token = Token.of_code code in
      feed t ~code ~phase ~bucket;
      match token with
      | Start -> assert false
      | End ->
        t.step_index <- t.step_index + 1;
        List.rev events
      | On pitch ->
        let seat = highest_free t.seats in
        t.seats.(seat) <- Some pitch;
        go ({ voice = seat; pitch; on = true } :: events) (count + 1)
      | Off pitch ->
        let seat = seat_of t.seats pitch in
        t.seats.(seat) <- None;
        go ({ voice = seat; pitch; on = false } :: events) (count + 1)
    in
    go [] 0
  ;;

  let peaks t =
    [ "h", t.pk.h
    ; "rms_sum", t.pk.rms_sum
    ; "y", t.pk.y
    ; "qkv", t.pk.qkv
    ; "score", t.pk.score
    ; "ctx", t.pk.ctx
    ; "hid", t.pk.hid
    ; "logit", t.pk.logit
    ]
  ;;
end

let%expect_test "the exp2 table: the peak, the floor and the halving" =
  (* entry 0 is the peak 2^15; a full fractional step halves; the last entry sits one
     table step above one half *)
  Stdio.printf
    "%d %d %d  half at one: %d\n"
    Tensor.exp2_table.(0)
    Tensor.exp2_table.(128)
    Tensor.exp2_table.(255)
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
(* [List.init] applies [f] in the reverse index order, thus it cannot collect from the
   mutable engine; the fold steps in the true order *)
let collect_steps n engine =
  List.rev
    (List.fold (List.range 0 n) ~init:[] ~f:(fun acc (_ : int) ->
       Engine.step_events engine :: acc))
;;

let%expect_test "a drawn walk keeps the grammar, the seats and the seed" =
  let model = Model.For_test.init Transformer.Config.baseline ~seed:11 in
  let engine = Engine.create model ~seed:42 in
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
    let engine = Engine.create model ~seed:42 in
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
