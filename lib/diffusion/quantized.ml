(* The integer twin of the masked canvas — see quantized.mli for the contract and
   docs/diffusion_rtl.md for the formats. Every rule here is one the circuit of the next
   round will read rather than restate. *)

open Core
module Nn = Mgen_nn.Quantized
module Policy = Mgen_nn.Policy

let rows = Diffusion.rows
let voices = Diffusion.voices

(* THE ACTIVATION FORMAT IS Q6 IN INT16, AND IT IS MEASURED. The trunk is a residual stack
   with no norm on the stream, thus a trained model's activations GROW with depth: the
   golden candidate peaks at 184 on half-masked corpus canvases and at 313 on the seeded
   openings the walk really visits — measured 2026-08-25, and the reason the opening
   design of the counters exists. Q6 holds 512 with a 1.6 margin over that peak; what its
   1/64 resolution costs is the drift report's to say, and a per-layer exponent (which the
   gain shift would absorb for free) is the refinement of the next round if the verdict is
   poor. *)
let activation_q = 6

(* the one in the activation format, and the fixed point of the biases *)
let activation_one = 1 lsl activation_q

module Model = struct
  type quantized = Mgen_nn.Quantized.quantized

  type layer =
    { kernel : quantized
    ; gain : Nn.Constants.scale array
    ; bias : int array
    ; inputs : int
    ; outputs : int
    }

  type t =
    { layers : layer array
    ; temper : Nn.Constants.scale
    }

  (* The 16-bit form of the exponent rule of the eras: the largest exponent that keeps the
     rounded gain in int16. The shift of the scale retires the weight exponent in the same
     move, thus the accumulator goes from Q(activation_q + e) to Q(activation_q) in one
     multiply. 30 caps the all-zero gain, as 14 caps the all-zero tensor in
     [Nn.max_exponent]. *)
  let gain_scale value ~weight_exponent =
    let magnitude = Float.abs value in
    let fits e = Float.iround_nearest_exn (Float.ldexp magnitude e) <= 32767 in
    let rec largest e = if fits e then e else largest (e - 1) in
    let e = if Float.(magnitude <= 0.0) then 30 else largest 30 in
    { Nn.Constants.q_value = Float.iround_nearest_exn (Float.ldexp value e)
    ; q = e + weight_exponent
    }
  ;;

  (* The norm folds into the two per-channel rows: at inference batch normalization is the
     affine [a * gain + bias] with [gain = scale * rsqrt (variance + eps)] and
     [bias = shift - mean * gain]. The fold is the same affine — its rounding is part of
     what [Drift] measures. A bias outside the activation format clamps; a trained norm
     that puts one there is a format fault the drift report would shout about. *)
  let fold_layer kernel_t scale_t shift_t mean_t variance_t =
    let kernel = Nn.quantize (Nx.to_array kernel_t) in
    let shape = Nx.shape kernel_t in
    let scale = Nx.to_array scale_t
    and shift = Nx.to_array shift_t
    and mean = Nx.to_array mean_t
    and variance = Nx.to_array variance_t in
    let gain_of channel =
      scale.(channel) /. Float.sqrt (variance.(channel) +. Diffusion.norm_epsilon)
    in
    { kernel
    ; gain =
        Array.init (Array.length scale) ~f:(fun channel ->
          gain_scale (gain_of channel) ~weight_exponent:kernel.e)
    ; bias =
        Array.init (Array.length scale) ~f:(fun channel ->
          Nn.clamp16
            (Float.iround_nearest_exn
               ((shift.(channel) -. (mean.(channel) *. gain_of channel))
                *. Float.of_int activation_one)))
    ; inputs = shape.(2)
    ; outputs = shape.(3)
    }
  ;;

  let of_params ?(temperature = 1.0) params =
    let layers =
      Diffusion.Params.to_list params
      |> List.chunks_of ~length:5
      |> List.map ~f:(function
        | [ kernel; scale; shift; mean; variance ] ->
          fold_layer kernel scale shift mean variance
        | group -> invalid_argf "a layer holds 5 tensors, not %d" (List.length group) ())
      |> Array.of_list
    in
    { layers; temper = fst (Nn.policy ~temperature ~min_p:0.0) }
  ;;

  let of_checkpoint ?temperature config path =
    of_params ?temperature (Diffusion.Params.load config ~path)
  ;;

  (* The int32 accumulator of the machine is exact below this width: 9 C products of int8
     by int16 reach 9 * 57 * 127 * 32767, which stands under 2^31, and one channel more
     can pass it. The elected shapes stand far under; the rule stands so the prose cannot
     rot. *)
  let widest_inputs = 57

  let check_shape { layers; temper = (_ : Nn.Constants.scale) } =
    let count = Array.length layers in
    if count < 3 then invalid_argf "%d layers is no canvas model" count ();
    if count % 2 <> 0 then invalid_argf "%d layers hold no whole residual pairs" count ();
    if layers.(0).inputs <> 2 * voices
    then invalid_argf "the stem reads %d planes, not %d" layers.(0).inputs (2 * voices) ();
    if layers.(count - 1).outputs <> voices
    then
      invalid_argf
        "the head states %d channels, not the %d voices"
        layers.(count - 1).outputs
        voices
        ();
    Array.iteri layers ~f:(fun at { kernel; gain; bias; inputs; outputs } ->
      if at > 0 && inputs <> layers.(at - 1).outputs
      then
        invalid_argf
          "layer %d reads %d channels and the layer before it wrote %d"
          at
          inputs
          layers.(at - 1).outputs
          ();
      if Array.length kernel.q <> 9 * inputs * outputs
      then
        invalid_argf
          "the kernel of layer %d holds %d weights, not %d"
          at
          (Array.length kernel.q)
          (9 * inputs * outputs)
          ();
      if inputs > widest_inputs
      then
        invalid_argf
          "layer %d reads %d channels and the int32 accumulator holds %d"
          at
          inputs
          widest_inputs
          ();
      if Array.length gain <> outputs || Array.length bias <> outputs
      then invalid_argf "the constants of layer %d do not cover its channels" at ())
  ;;

  let rom_tensors { layers; temper = (_ : Nn.Constants.scale) } =
    Array.to_list (Array.map layers ~f:(fun { kernel; _ } -> kernel))
  ;;

  let rom_bits model = Nn.rom_bits (rom_tensors model)

  let rom_bases model =
    let sizes =
      List.map (rom_tensors model) ~f:(fun { q; e = (_ : int) } -> Array.length q)
    in
    Array.of_list (List.folding_map sizes ~init:0 ~f:(fun base size -> base + size, base))
  ;;

  module For_test = struct
    let config = { Diffusion.Config.layers = 4; width = 6 }
    let init config ~seed = of_params (Diffusion.Params.init config ~seed)
  end
end

module Clamps = struct
  type t =
    { activations : int
    ; activations_seen : int
    ; peak : int
    }

  let share hit seen = if seen = 0 then 0.0 else Float.of_int hit /. Float.of_int seen
end

(* the running counters of a walk, folded into [Clamps.t] when a caller reads them *)
type counters =
  { mutable hits : int
  ; mutable seen : int
  ; mutable peak : int
  }

(* Every activation write goes through here: the clamp is counted and the peak is kept,
   never assumed away. The peak reads BEFORE the clamp, thus it answers the format
   question directly — the Q6 election was made on exactly this number, measured then with
   a throwaway probe, and the counter is what makes it re-measurable when the checkpoint
   changes. *)
let write counters out index value =
  if Nn.clamps16 value then counters.hits <- counters.hits + 1;
  counters.seen <- counters.seen + 1;
  counters.peak <- max counters.peak (abs value);
  out.(index) <- Nn.clamp16 value
;;

(* the input planes in the activation format: a cell of the masked roll is 0 or one, exact *)
let plane_activations canvas hidden ~steps =
  let planes = 2 * voices in
  let x = Array.create ~len:(steps * rows * planes) 0 in
  for step = 0 to steps - 1 do
    for voice = 0 to voices - 1 do
      if hidden.(step).(voice)
      then
        for row = 0 to rows - 1 do
          x.((((step * rows) + row) * planes) + voices + voice) <- activation_one
        done
      else
        x.((((step * rows) + canvas.(step).(voice)) * planes) + voice) <- activation_one
    done
  done;
  x
;;

(* One layer: the convolution into the int32 accumulator, the folded norm, the optional
   ReLU, and the counted clamp of every write.

   The accumulator is exact below [Model.widest_inputs] input channels — 9 C products of
   int8 by int16 stand under 2^31, and [check_shape] refuses a wider layer — thus the tap
   order cannot matter and the circuit may take its own. The gain multiply rides the wide
   host int and the RTL will size its own product. *)
let layer_forward counters (layer : Model.layer) ~steps ~relu x =
  let { Model.kernel = { q = weights; e = (_ : int) }; gain; bias; inputs; outputs } =
    layer
  in
  let out = Array.create ~len:(steps * rows * outputs) 0 in
  for step = 0 to steps - 1 do
    for row = 0 to rows - 1 do
      for channel = 0 to outputs - 1 do
        let acc = ref 0 in
        for dy = 0 to 2 do
          let source_step = step + dy - 1 in
          if source_step >= 0 && source_step < steps
          then
            for dx = 0 to 2 do
              let source_row = row + dx - 1 in
              if source_row >= 0 && source_row < rows
              then (
                let x_base = ((source_step * rows) + source_row) * inputs in
                let k_base = ((dy * 3) + dx) * inputs in
                for input = 0 to inputs - 1 do
                  acc
                  := !acc
                     + (x.(x_base + input)
                        * weights.(((k_base + input) * outputs) + channel))
                done)
            done
        done;
        let value = Nn.Constants.apply gain.(channel) !acc + bias.(channel) in
        let value = if relu then max value 0 else value in
        write counters out ((((step * rows) + row) * outputs) + channel) value
      done
    done
  done;
  out
;;

(* the trunk: the stem, the residual pairs, the head — the structure of the float model,
   join for join. The skip adds two activation-format values and the sum is written
   through the same counted clamp. *)
let forward counters (model : Model.t) canvas hidden ~steps =
  let layers = model.layers in
  let last = Array.length layers - 1 in
  let stem =
    layer_forward
      counters
      layers.(0)
      ~steps
      ~relu:true
      (plane_activations canvas hidden ~steps)
  in
  let trunk =
    List.fold (List.range ~stride:2 1 last) ~init:stem ~f:(fun x at ->
      let first = layer_forward counters layers.(at) ~steps ~relu:true x in
      let second = layer_forward counters layers.(at + 1) ~steps ~relu:false first in
      let joined = Array.create ~len:(Array.length x) 0 in
      Array.iteri x ~f:(fun index held ->
        write counters joined index (max 0 (held + second.(index))));
      joined)
  in
  layer_forward counters layers.(last) ~steps ~relu:false trunk
;;

(* the logits of one cell over the rows, in the activation format *)
let column said ~step ~voice =
  Array.init rows ~f:(fun row -> said.((((step * rows) + row) * voices) + voice))
;;

(* the draw of one cell: the logits temper against their peak, exp2 gives Q15 weights, and
   the shared 24-bit pick takes the class — era four's pipeline, rule for rule *)
let draw_cell (model : Model.t) raw prng =
  let peak = Array.fold raw ~init:Int.min_value ~f:max in
  let weights =
    Array.map raw ~f:(fun logit ->
      (* the logits carry Q[activation_q] and the exp2 unit reads Q12, thus the difference
         shifts up by the gap first — exact, because a left shift of an int is. A
         difference read at the wrong Q is silently wrong music: unshifted, every weight
         stands within a fraction of a nat of the peak and the draw is uniform — the fault
         the drift report caught at 3.4 percent same-draw. *)
      Nn.exp2_q (Nn.Constants.apply model.temper ((logit - peak) lsl (12 - activation_q))))
  in
  Nn.draw ~weights prng
;;

module Engine = struct
  type draw =
    { step : int
    ; voice : int
    ; logits : int array
    ; uniform : float
    ; drawn : int
    }

  type pass =
    { before : int array array
    ; hidden : bool array array
    ; draws : draw list
    }

  type t =
    { model : Model.t
    ; steps : int
    ; walk : int
    ; pass : int
    ; canvas : int array array
    ; prng : Prng.state
    ; clamps : Clamps.t
    }

  let init model ~steps ~walk ~seed =
    Model.check_shape model;
    let prng, canvas = Diffusion.opening_canvas (Prng.create ~seed) ~steps in
    { model
    ; steps
    ; walk
    ; pass = 0
    ; canvas
    ; prng
    ; clamps = { Clamps.activations = 0; activations_seen = 0; peak = 0 }
    }
  ;;

  let canvas t = t.canvas
  let clamps t = t.clamps

  let next_pass t =
    if t.pass >= t.walk then invalid_arg "the walk is finished";
    let threshold = Diffusion.anneal_threshold ~step:t.pass ~walk:t.walk in
    let prng, hidden = Diffusion.hidden_cells t.prng ~steps:t.steps ~threshold in
    (* the engine never writes a canvas in place, thus the record shares it and one copy
       serves the successor *)
    let before = t.canvas in
    let counters = { hits = 0; seen = 0; peak = 0 } in
    let said = forward counters t.model before hidden ~steps:t.steps in
    let canvas = Array.map before ~f:Array.copy in
    (* one draw for each hidden cell, in the cell order the float walk takes: a cell the
       mask left standing takes no uniform *)
    let draw_hidden_cell (prng, draws) (step, voice) =
      if not hidden.(step).(voice)
      then prng, draws
      else (
        let logits = column said ~step ~voice in
        let next, uniform, drawn = draw_cell t.model logits prng in
        canvas.(step).(voice) <- drawn;
        next, { step; voice; logits; uniform; drawn } :: draws)
    in
    let prng, draws =
      List.fold (Diffusion.cell_order ~steps:t.steps) ~init:(prng, []) ~f:draw_hidden_cell
    in
    ( { t with
        pass = t.pass + 1
      ; canvas
      ; prng
      ; clamps =
          { Clamps.activations = t.clamps.activations + counters.hits
          ; activations_seen = t.clamps.activations_seen + counters.seen
          ; peak = max t.clamps.peak counters.peak
          }
      }
    , { before; hidden; draws = List.rev draws } )
  ;;

  let rec run t = if t.pass >= t.walk then t.canvas else run (fst (next_pass t))
end

module Drift = struct
  type stats =
    { passes : int
    ; cells : int
    ; same_peak : int
    ; same_draw : int
    ; mean_cosine : float
    ; activations_clamped : float
    ; activation_peak : float
    }

  (* what the comparison counts over the cells it has seen *)
  type tally =
    { cells : int
    ; same_peak : int
    ; same_draw : int
    ; cosine_sum : float
    }

  let counted_nothing = { cells = 0; same_peak = 0; same_draw = 0; cosine_sum = 0.0 }

  (* the state the report folds: the engine, and what the comparison has counted *)
  type walk =
    { engine : Engine.t
    ; tally : tally
    }

  let walk params ~steps ~walk ~seed =
    let model = Model.of_params params in
    (* one cell of a pass against the float logits of the same cell, on the very uniform
       the engine took *)
    let count_cell float_said tally { Engine.step; voice; logits; uniform; drawn } =
      let raw =
        Array.init rows ~f:(fun row ->
          float_said.((((step * rows) + row) * voices) + voice))
      in
      let float_class = Policy.draw_class raw ~temperature:1.0 ~min_p:0.0 ~uniform in
      { cells = tally.cells + 1
      ; same_peak = (tally.same_peak + if Nn.Tensor.same_peak logits raw then 1 else 0)
      ; same_draw = (tally.same_draw + if float_class = drawn then 1 else 0)
      ; cosine_sum = tally.cosine_sum +. Nn.Tensor.cosine logits raw
      }
    in
    let take_pass w (_ : int) =
      let engine, (pass : Engine.pass) = Engine.next_pass w.engine in
      (* the float model, teacher-forced on the engine's canvas and the engine's mask: one
         context, thus what parts the two is the arithmetic alone *)
      let float_said =
        Nx.to_array (Diffusion.logits params ~classes:pass.before ~hidden:pass.hidden)
      in
      { engine; tally = List.fold pass.draws ~init:w.tally ~f:(count_cell float_said) }
    in
    let opening =
      { engine = Engine.init model ~steps ~walk ~seed; tally = counted_nothing }
    in
    let walked = List.fold (List.range 0 walk) ~init:opening ~f:take_pass in
    let { cells; same_peak; same_draw; cosine_sum } = walked.tally in
    let clamps = Engine.clamps walked.engine in
    { passes = walk
    ; cells
    ; same_peak
    ; same_draw
    ; mean_cosine = (if cells = 0 then 1.0 else cosine_sum /. Float.of_int cells)
    ; activations_clamped = Clamps.share clamps.activations clamps.activations_seen
    ; activation_peak = Float.of_int clamps.peak /. Float.of_int activation_one
    }
  ;;
end

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the quantized model holds its shape" =
  let model = Model.For_test.init Model.For_test.config ~seed:11 in
  printf
    "check_shape: %s\n"
    (Mgen_nn.Checkpoint.refusal (fun () -> Model.check_shape model));
  let sizes =
    List.map (Model.rom_tensors model) ~f:(fun { q; e = (_ : int) } -> Array.length q)
  in
  print_s ([%sexp_of: int list] sizes);
  print_s ([%sexp_of: int array] (Model.rom_bases model));
  printf "rom bytes: %d\n" (Array.length (Model.rom_bits model));
  [%expect
    {|
    check_shape: no raise
    (432 324 324 216)
    (0 432 756 1080)
    rom bytes: 1296
    |}]
;;

let%expect_test "a broken model refuses loudly" =
  let model = Model.For_test.init Model.For_test.config ~seed:11 in
  let refuse broken =
    printf "%s\n" (Mgen_nn.Checkpoint.refusal (fun () -> Model.check_shape broken))
  in
  refuse { model with layers = Array.sub model.layers ~pos:0 ~len:3 };
  [%expect {| 3 layers hold no whole residual pairs |}];
  let head = model.layers.(3) in
  let chopped = { head with gain = Array.sub head.gain ~pos:0 ~len:2 } in
  refuse
    { model with
      layers = Array.append (Array.sub model.layers ~pos:0 ~len:3) [| chopped |]
    };
  [%expect {| the constants of layer 3 do not cover its channels |}]
;;

let%expect_test "the two openings are one opening" =
  (* a seed inside 32 bits folds to itself, thus the float walk and the engine start on
     the same state, consume the same uniforms, and state the same opening canvas *)
  let params = Diffusion.Params.init Model.For_test.config ~seed:3 in
  let model = Model.of_params params in
  let engine = Engine.init model ~steps:16 ~walk:1 ~seed:7 in
  let float_opening = Diffusion.gibbs params ~steps:16 ~walk:0 ~temperature:1.0 ~seed:7 in
  printf
    "the engine opens where the float walk opens: %b\n"
    (Array.equal (Array.equal ( = )) (Engine.canvas engine) float_opening);
  [%expect {| the engine opens where the float walk opens: true |}]
;;

let%expect_test "the engine walk is the seed and nothing else" =
  let model = Model.For_test.init Model.For_test.config ~seed:3 in
  let draw seed = Engine.run (Engine.init model ~steps:8 ~walk:3 ~seed) in
  let same a b = Array.equal (Array.equal ( = )) a b in
  printf "one seed, one canvas: %b\n" (same (draw 5) (draw 5));
  printf "another seed, another canvas: %b\n" (not (same (draw 5) (draw 6)));
  [%expect
    {|
    one seed, one canvas: true
    another seed, another canvas: true
    |}]
;;

let%expect_test "the drift of a drawn model: the twin tracks the float reference" =
  (* an init model at norm scale 1.0, thus no checkpoint enters and the drawn trunk holds
     the O(1) activations a trained norm holds — at the trainer's opening tenth a drawn
     trunk decays tenfold at every layer and the report reads the resolution floor of the
     activation format instead of the arithmetic. The era's numbers on the elected
     checkpoint are the drift tool's; these pin the mechanics. *)
  let params = Diffusion.Params.init ~norm_scale:1.0 Model.For_test.config ~seed:3 in
  let { Drift.passes
      ; cells
      ; same_peak
      ; same_draw
      ; mean_cosine
      ; activations_clamped
      ; activation_peak
      }
    =
    Drift.walk params ~steps:16 ~walk:4 ~seed:7
  in
  let share count = Float.of_int count /. Float.of_int (max 1 cells) in
  printf "%d passes redrew %d cells\n" passes cells;
  printf
    "top1 %.3f same_draw %.3f cosine %.4f clamped %.4f peak %.2f\n"
    (share same_peak)
    (share same_draw)
    mean_cosine
    activations_clamped
    activation_peak;
  (* MEASURED NUMBERS AND NOT THRESHOLDS, the rule of the drift gates: a diff here says
     the integers moved — judge whether it is a re-measurement or a bug. The larger sweep
     is [test/test_diffusion_drift.ml]; the era's numbers on the elected checkpoint are
     the drift tool's. *)
  [%expect
    {|
    4 passes redrew 120 cells
    top1 0.967 same_draw 0.958 cosine 0.9980 clamped 0.0000 peak 3.95
    |}]
;;
