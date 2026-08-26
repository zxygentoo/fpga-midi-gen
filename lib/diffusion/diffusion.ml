(* The float reference of the masked canvas — see diffusion.mli for the contract and
   docs/diffusion_rtl.md for the design. Keep the exact arithmetic: Gate A holds this
   forward against the JAX forward, and Gate C holds this walk against the JAX walk. *)

open Core
module Checkpoint = Mgen_nn.Checkpoint
module Policy = Mgen_nn.Policy

type tensor = Checkpoint.tensor

let rows = Vocab.classes
let voices = Frame.voices

(* the reach of one convolution: three by three over time and pitch *)
let kernel = 3
let norm_epsilon = 1e-7

(* the 24-bit grid of the generator: a uniform is [k * 2 ** -24], thus [u * grid] is the
   integer [k] and a threshold compare over it is exact in a double *)
let grid = Float.of_int (1 lsl 24)

(* The register of each seat: the lowest and the highest pitch it sings anywhere in this
   corpus, seat 0 the bass — [Jsb.voice_ranges] turned around, thus the corpus library's
   own test pins it. The corpus table gives the soprano first, as the file does.

   [opening_canvas] draws inside these, thus a cell the first Bernoulli leaves standing
   states a note a chorale could hold. They are the RANGES of [jax/measure.py], stated as
   pitches. *)
let seat_ranges = Array.of_list (List.rev (Array.to_list Jsb.voice_ranges))

(* the top of the corpus: the classes 1 to 46 cover [Vocab.pitch_low] to here, and the
   spare class stands one above it *)
let pitch_high = Vocab.pitch_low + rows - 3

(* The pitch of one corpus cell becomes its class, and the reader is DELIBERATELY NARROWER
   THAN [Vocab]: the window holds one spare class above the corpus — pitch 82, the draw's
   to state and never the corpus's — thus a file that names it is corrupt and refuses
   loudly rather than filing the fault under the spare. The twin reader of [jax/data.py]
   refuses the same range with the same reason. *)
let class_of_cell pitch =
  if pitch < 0
  then Vocab.silence
  else if pitch < Vocab.pitch_low || pitch > pitch_high
  then
    invalid_argf
      "the pitch %d is outside the corpus's %d to %d"
      pitch
      Vocab.pitch_low
      pitch_high
      ()
  else pitch - Vocab.pitch_low + 1
;;

type opening =
  { low : int
  ; width : int
  }

(* The register of each seat as classes: [seat_ranges] read through [class_of_cell]
   itself, thus the map has ONE home and a circuit that restated [low - pitch_low + 1]
   could not part from it. [opening_canvas] draws inside these and the elaboration of the
   circuit carries them, thus the opening of the walk and the opening of the board are one
   rule. *)
let seat_openings =
  Array.map seat_ranges ~f:(fun (low, high) ->
    { low = class_of_cell low; width = high - low + 1 })
;;

(* the canvas of one piece of [Jsb]: [steps] rows of [voices] class indices, seat 0 the
   bass, and a rest is the silence class. [class_of_cell] states what a pitch outside the
   corpus does, and why. *)
let classes_of_chorale { Jsb.cells; legal_shifts = _ } =
  (* the file gives the soprano first; the reverse lands the bass at seat 0 *)
  Array.map cells ~f:(fun step -> Array.of_list (List.rev_map step ~f:class_of_cell))
;;

module Config = struct
  type t =
    { layers : int
    ; width : int
    }

  let of_checkpoint path =
    let archive = Nx_io.load_safetensors path in
    let count = Stdlib.Hashtbl.length archive in
    let layers, spare = count / 5, count % 5 in
    if spare <> 0 || layers < 3
    then invalid_argf "%s: %d tensors is no canvas model" path count ();
    if layers % 2 <> 0
    then invalid_argf "%s: %d layers hold no whole residual pairs" path layers ();
    let shape_at index =
      match Stdlib.Hashtbl.find_opt archive (Int.to_string index) with
      | None -> invalid_argf "%s has no tensor %d" path index ()
      | Some packed -> Nx.shape (Nx_io.to_typed Nx.float32 packed)
    in
    let refuse name index wanted =
      invalid_argf
        "the %s of %s is %s at tensor %d, and not %s"
        name
        path
        (Sexp.to_string ([%sexp_of: int array] (shape_at index)))
        index
        wanted
        ()
    in
    let width =
      match shape_at 0 with
      | [| k1; k2; planes; width |] when k1 = kernel && k2 = kernel && planes = 2 * voices
        -> width
      | _ ->
        refuse
          "stem kernel"
          0
          (sprintf "%d by %d over %d planes" kernel kernel (2 * voices))
    in
    (match shape_at (5 * (layers - 1)) with
     | [| k1; k2; inputs; outputs |]
       when k1 = kernel && k2 = kernel && inputs = width && outputs = voices -> ()
     | _ ->
       refuse
         "head kernel"
         (5 * (layers - 1))
         (sprintf "%d by %d from %d channels to %d voices" kernel kernel width voices));
    { layers; width }
  ;;
end

module Params = struct
  (* one layer of the checkpoint: the kernel, the two norm terms, and the two population
     statistics. Every pass below the seam reads the population — the reference has no
     training mode. *)
  type layer =
    { kernel : tensor
    ; scale : tensor
    ; shift : tensor
    ; mean : tensor
    ; variance : tensor
    }

  type t = layer array

  (* the input and output channels of each layer: the stem widens the planes, the trunk
     holds the width, and the head narrows to the voices *)
  let channels { Config.layers; width } =
    List.init layers ~f:(fun at ->
      (if at = 0 then 2 * voices else width), if at = layers - 1 then voices else width)
  ;;

  (* the shapes of the tensors in the flat order of the checkpoint *)
  let shapes config =
    List.concat_map (channels config) ~f:(fun (inputs, outputs) ->
      [ [| kernel; kernel; inputs; outputs |]
      ; [| outputs |]
      ; [| outputs |]
      ; [| outputs |]
      ; [| outputs |]
      ])
  ;;

  let init ?(norm_scale = 0.1) config ~seed =
    (* the He normal of the trainer, one draw for each kernel: sqrt (2 / fan_in) over the
       reach and the input channels. [Prng.all] runs left to right, thus the stream is the
       layer order and no state travels by hand. *)
    let draws =
      List.map (channels config) ~f:(fun (inputs, outputs) ->
        Prng.normals
          ~count:(kernel * kernel * inputs * outputs)
          ~scale:(Float.sqrt (2.0 /. Float.of_int (kernel * kernel * inputs))))
    in
    let (_ : Prng.state), weights =
      Prng.run (Prng.all draws) (Prng.create_folded ~seed)
    in
    (* the norm scale opens at the trainer's tenth unless a gate asks for the trained
       regime, and the population opens at the prior *)
    let layer (inputs, outputs) drawn =
      { kernel = Nx.create Nx.float32 [| kernel; kernel; inputs; outputs |] drawn
      ; scale = Nx.full Nx.float32 [| outputs |] norm_scale
      ; shift = Nx.full Nx.float32 [| outputs |] 0.0
      ; mean = Nx.full Nx.float32 [| outputs |] 0.0
      ; variance = Nx.full Nx.float32 [| outputs |] 1.0
      }
    in
    Array.of_list (List.map2_exn (channels config) weights ~f:layer)
  ;;

  let to_list params =
    Array.to_list params
    |> List.concat_map ~f:(fun { kernel; scale; shift; mean; variance } ->
      [ kernel; scale; shift; mean; variance ])
  ;;

  let of_list tensors =
    List.chunks_of tensors ~length:5
    |> List.map ~f:(function
      | [ kernel; scale; shift; mean; variance ] ->
        { kernel; scale; shift; mean; variance }
      | group -> invalid_argf "a layer holds 5 tensors, not %d" (List.length group) ())
    |> Array.of_list
  ;;

  let load config ~path =
    let archive = Nx_io.load_safetensors path in
    let tensors =
      List.mapi (shapes config) ~f:(fun index wanted ->
        match Stdlib.Hashtbl.find_opt archive (Int.to_string index) with
        | None -> invalid_argf "%s has no tensor %d" path index ()
        | Some packed ->
          let tensor = Nx_io.to_typed Nx.float32 packed in
          if not (Array.equal ( = ) (Nx.shape tensor) wanted)
          then
            invalid_argf
              "tensor %d of %s is %s, and not %s"
              index
              path
              (Sexp.to_string ([%sexp_of: int array] (Nx.shape tensor)))
              (Sexp.to_string ([%sexp_of: int array] wanted))
              ();
          tensor)
    in
    of_list tensors
  ;;
end

(* ==================================================================== *)
(* The forward pass *)
(* ==================================================================== *)

(* the paper's input: the masked one-hot roll beside the mask, [steps; rows; 2 voices]. A
   hidden cell shows zero in every row of its roll column and one in every row of its mask
   plane. *)
let planes ~classes ~hidden =
  let steps = Array.length classes in
  Nx.init
    Nx.float32
    [| steps; rows; 2 * voices |]
    (fun index ->
      let step = index.(0)
      and row = index.(1)
      and channel = index.(2) in
      if channel < voices
      then
        if (not hidden.(step).(channel)) && classes.(step).(channel) = row
        then 1.0
        else 0.0
      else if hidden.(step).(channel - voices)
      then 1.0
      else 0.0)
;;

(* One convolution over the step axis and the pitch axis, zero at all four edges, no bias.
   It runs as im2col against one matmul: the nine shifted views of the padded canvas
   concatenate along the channels in the kernel's own (time, pitch, input) order, thus the
   reshaped kernel reads them row for row and the pass is one BLAS call.

   EVERY TAP IS MATERIALIZED BEFORE THE CONCATENATE, and the [Nx.contiguous] is load
   bearing: [Nx.concatenate] misreads a strided slice of a padded tensor at some shapes —
   measured at [4; 48; 2] the result rolls the step axis by one — and the expect test
   below holds this function against a direct loop at the shapes that caught it. *)
let conv x weights =
  let shape = Nx.shape x in
  let steps = shape.(0)
  and pitches = shape.(1)
  and inputs = shape.(2) in
  let outputs = (Nx.shape weights).(3) in
  let padded = Nx.pad [| 1, 1; 1, 1; 0, 0 |] 0.0 x in
  let taps =
    List.concat_map [ 0; 1; 2 ] ~f:(fun dy ->
      List.map [ 0; 1; 2 ] ~f:(fun dx ->
        Nx.contiguous (Nx.slice [ R (dy, dy + steps); R (dx, dx + pitches); A ] padded)))
  in
  let columns = Nx.contiguous (Nx.concatenate ~axis:2 taps) in
  let flat = Nx.reshape [| steps * pitches; kernel * kernel * inputs |] columns in
  let table =
    Nx.reshape [| kernel * kernel * inputs; outputs |] (Nx.contiguous weights)
  in
  Nx.reshape [| steps; pitches; outputs |] (Nx.matmul flat table)
;;

(* the trainer's norm, expression for expression: (a - mean) * rsqrt (variance + eps),
   then scale and shift. Gate A holds the two forwards to a tolerance, thus no algebraic
   rearrangement. *)
let normed_conv x (layer : Params.layer) =
  let a = conv x layer.kernel in
  let width = (Nx.shape a).(2) in
  let vec v = Nx.reshape [| 1; 1; width |] v in
  let normed =
    Nx.mul
      (Nx.sub a (vec layer.mean))
      (Nx.rsqrt (Nx.add_s (vec layer.variance) norm_epsilon))
  in
  Nx.add (Nx.mul normed (vec layer.scale)) (vec layer.shift)
;;

let logits (params : Params.t) ~classes ~hidden =
  let last = Array.length params - 1 in
  let stem = Nx.relu (normed_conv (planes ~classes ~hidden) params.(0)) in
  (* the residual pairs: two layers and the skip past both, activated once on the sum *)
  let trunk =
    List.fold (List.range ~stride:2 1 last) ~init:stem ~f:(fun x at ->
      let first = normed_conv x params.(at) in
      let second = normed_conv (Nx.relu first) params.(at + 1) in
      Nx.relu (Nx.add x second))
  in
  (* the head keeps its norm and takes no activation: the norm is a learned scale on the
     logits *)
  normed_conv trunk params.(last)
;;

let column said ~step ~voice =
  Array.init rows ~f:(fun p -> said.((((step * rows) + p) * voices) + voice))
;;

let masked_nll (params : Params.t) ~classes ~hidden =
  let said = Nx.to_array (logits params ~classes ~hidden) in
  (* the cross entropy of one cell: the log-sum-exp of its column, less the logit of the
     label the canvas holds *)
  let cell_nll ~step ~voice label =
    let raw = column said ~step ~voice in
    let peak = Array.fold raw ~init:Float.neg_infinity ~f:Float.max in
    let sum =
      Array.fold raw ~init:0.0 ~f:(fun acc logit -> acc +. Float.exp (logit -. peak))
    in
    peak +. Float.log sum -. raw.(label)
  in
  let over_step step total_and_count cells =
    Array.foldi cells ~init:total_and_count ~f:(fun voice (total, count) label ->
      if hidden.(step).(voice)
      then total +. cell_nll ~step ~voice label, count + 1
      else total, count)
  in
  let total, count = Array.foldi classes ~init:(0.0, 0) ~f:over_step in
  if count = 0 then invalid_arg "the masked loss takes one hidden cell or more";
  total /. Float.of_int count
;;

(* ==================================================================== *)
(* The walk *)
(* ==================================================================== *)

let anneal_threshold ~step ~walk =
  let low = 0.1
  and high = 0.9 in
  (* the paper's schedule, in the trainer's expression: both language sides evaluate the
     same doubles, thus the thresholds agree bit for bit *)
  let alpha =
    Float.max
      low
      (high -. ((high -. low) *. Float.of_int step /. (0.7 *. Float.of_int walk)))
  in
  Int.of_float (Float.round_down (alpha *. grid))
;;

(* a threshold compare on the 24-bit grid: exact, because [u * grid] is an integer *)
let under ~threshold u = Float.(u *. grid < of_int threshold)

(* The cell order of the walk: one step at a time, and the seats inside a step. Every
   uniform of the walk is drawn in this order and every hidden cell is drawn in it, thus
   the integer twin takes the same walk by taking the same order. *)
let cell_order ~steps = List.cartesian_product (List.range 0 steps) (List.range 0 voices)

(* one uniform for each cell in the cell order, folded into [f] *)
let over_cells state ~steps ~f =
  List.fold (cell_order ~steps) ~init:state ~f:(fun state (step, voice) ->
    let next, value = Prng.run Prng.uniform state in
    f ~step ~voice value;
    next)
;;

let opening_canvas state ~steps =
  let canvas = Array.make_matrix ~dimx:steps ~dimy:voices 0 in
  (* the product [u * width] is exact on the grid, thus the twin and the circuit state the
     same class from the same uniform *)
  let state =
    over_cells state ~steps ~f:(fun ~step ~voice u ->
      let { low; width } = seat_openings.(voice) in
      canvas.(step).(voice)
      <- low + Int.of_float (Float.round_down (u *. Float.of_int width)))
  in
  state, canvas
;;

let hidden_cells state ~steps ~threshold =
  let hidden = Array.make_matrix ~dimx:steps ~dimy:voices false in
  let state =
    over_cells state ~steps ~f:(fun ~step ~voice u ->
      hidden.(step).(voice) <- under ~threshold u)
  in
  state, hidden
;;

let gibbs (params : Params.t) ~steps ~walk ~temperature ~seed =
  Policy.check_policy ~temperature ~min_p:0.0;
  (* the canvas is the walk's own state: local mutation, because the walk IS a sequence of
     writes and a rebuilt canvas at every pass would say less *)
  let state, canvas = opening_canvas (Prng.create_folded ~seed) ~steps in
  (* one uniform for each hidden cell, in the cell order: a cell the mask left standing
     takes no uniform, thus the walk of the twin lands on the same draws *)
  let draw_hidden_cells state ~hidden =
    let said = Nx.to_array (logits params ~classes:canvas ~hidden) in
    List.fold (cell_order ~steps) ~init:state ~f:(fun state (step, voice) ->
      if not hidden.(step).(voice)
      then state
      else (
        let raw = column said ~step ~voice in
        let next, uniform = Prng.run Prng.uniform state in
        canvas.(step).(voice) <- Policy.draw_class raw ~temperature ~min_p:0.0 ~uniform;
        next))
  in
  let take_pass state pass =
    let threshold = anneal_threshold ~step:pass ~walk in
    let state, hidden = hidden_cells state ~steps ~threshold in
    draw_hidden_cells state ~hidden
  in
  let (_ : Prng.state) = List.fold (List.range 0 walk) ~init:state ~f:take_pass in
  canvas
;;

let frames_of_canvas canvas =
  Array.map canvas ~f:(fun step -> Vocab.frame_of_classes (Array.to_list step))
;;

let gate_canvases chorales ~steps =
  List.filter_map chorales ~f:(fun chorale ->
    let canvas = classes_of_chorale chorale in
    if Array.length canvas >= steps
    then Some (Array.sub canvas ~pos:0 ~len:steps)
    else None)
  |> Array.of_list
;;

(* a Bernoulli half on the grid: hidden exactly when u * 2^24 < 2^23 *)
let gate_mask ~index ~steps =
  snd (hidden_cells (Prng.create ~seed:(index + 1)) ~steps ~threshold:(1 lsl 23))
;;

(* the checkpoint seam of the expect tests below. The interface states nothing of it: the
   tests that read it stand in this file, thus nothing outside needs the names. *)
module For_test = struct
  let with_checkpoint = Checkpoint.with_checkpoint
  let refusal ~path f = Checkpoint.scrubbed_refusal ~path f
end

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "a checkpoint states its own shape, and the tensors survive the seam" =
  let config = { Config.layers = 4; width = 6 } in
  let params = Params.init config ~seed:11 in
  For_test.with_checkpoint (Params.to_list params) ~f:(fun path ->
    let { Config.layers; width } = Config.of_checkpoint path in
    printf "the file states %d layers of %d channels\n" layers width;
    let back = Params.load config ~path in
    let same =
      List.for_all2_exn
        (Params.to_list params)
        (Params.to_list back)
        ~f:(fun ours theirs ->
          Array.equal Float.equal (Nx.to_array ours) (Nx.to_array theirs))
    in
    printf "every tensor reads back equal: %b\n" same);
  [%expect
    {|
    the file states 4 layers of 6 channels
    every tensor reads back equal: true
    |}]
;;

let%expect_test "a file that is not a canvas model refuses, and the message says why" =
  let dummy count shape =
    List.init count ~f:(fun (_ : int) -> Nx.full Nx.float32 shape 0.0)
  in
  let refuse tensors =
    For_test.with_checkpoint tensors ~f:(fun path ->
      printf
        "%s\n"
        (For_test.refusal ~path (fun () -> ignore (Config.of_checkpoint path))))
  in
  refuse (dummy 7 [| 1 |]);
  [%expect {| <file>: 7 tensors is no canvas model |}];
  refuse (dummy 15 [| 1 |]);
  [%expect {| <file>: 3 layers hold no whole residual pairs |}];
  refuse (dummy 20 [| 1 |]);
  [%expect
    {| the stem kernel of <file> is (1) at tensor 0, and not 3 by 3 over 8 planes |}];
  (* a right stem and a wrong head: the head check names its tensor *)
  let stem = Nx.full Nx.float32 [| 3; 3; 8; 6 |] 0.0 in
  let vector = Nx.full Nx.float32 [| 6 |] 0.0 in
  let middle = Nx.full Nx.float32 [| 3; 3; 6; 6 |] 0.0 in
  refuse
    ([ stem; vector; vector; vector; vector ]
     @ List.concat
         (List.init 3 ~f:(fun (_ : int) -> [ middle; vector; vector; vector; vector ])));
  [%expect
    {| the head kernel of <file> is (3 3 6 6) at tensor 15, and not 3 by 3 from 6 channels to 4 voices |}]
;;

let%expect_test "an untrained model states the uniform prior" =
  (* the population opens at the prior and the norm scale at a tenth, thus the logits of a
     drawn model are near flat and the loss near log 48. A loss far from it at step zero
     is a scale fault — the same gate the trainer's tests state. *)
  let config = { Config.layers = 4; width = 6 } in
  let params = Params.init config ~seed:3 in
  let classes = gibbs params ~steps:8 ~walk:0 ~temperature:1.0 ~seed:5 in
  let hidden = gate_mask ~index:0 ~steps:8 in
  let value = masked_nll params ~classes ~hidden in
  printf
    "within 0.3 of log 48: %b\n"
    Float.(abs (value -. Float.log (Float.of_int rows)) < 0.3);
  [%expect {| within 0.3 of log 48: true |}]
;;

let%expect_test "the walk is the seed and nothing else" =
  let config = { Config.layers = 4; width = 6 } in
  let params = Params.init config ~seed:3 in
  let draw seed = gibbs params ~steps:8 ~walk:3 ~temperature:1.0 ~seed in
  let same a b = Array.equal (Array.equal ( = )) a b in
  printf "one seed, one canvas: %b\n" (same (draw 5) (draw 5));
  printf "another seed, another canvas: %b\n" (not (same (draw 5) (draw 6)));
  [%expect
    {|
    one seed, one canvas: true
    another seed, another canvas: true
    |}]
;;

let%expect_test "the opening puts every voice inside the register of its seat" =
  (* a walk of zero passes is the opening itself: the draw is over the seat's range and
     never the whole roll, thus a cell the first masks leave standing states a note a
     chorale could hold *)
  let config = { Config.layers = 4; width = 6 } in
  let params = Params.init config ~seed:3 in
  let canvas = gibbs params ~steps:64 ~walk:0 ~temperature:1.0 ~seed:9 in
  let inside =
    Array.for_all canvas ~f:(fun step ->
      Array.for_alli step ~f:(fun voice index ->
        let low, high = seat_ranges.(voice) in
        let pitch = index + Vocab.pitch_low - 1 in
        index <> Vocab.silence && pitch >= low && pitch <= high))
  in
  printf "64 steps of 4 seats, every cell in register: %b\n" inside;
  [%expect {| 64 steps of 4 seats, every cell in register: true |}]
;;

let%expect_test "the anneal thresholds: the paper's schedule on the 24-bit grid" =
  let walk = 512 in
  let at step = anneal_threshold ~step ~walk in
  printf "step 0: %d = floor of 0.9 * 2^24\n" (at 0);
  printf "the floor: %d = floor of 0.1 * 2^24\n" (at (walk - 1));
  let thresholds = Array.init walk ~f:(fun step -> at step) in
  let falling =
    Array.for_alli thresholds ~f:(fun step value ->
      value <= if step = 0 then value else thresholds.(step - 1))
  in
  printf "the schedule only falls: %b\n" falling;
  (* it reaches the floor after the span share of the walk, not at its end *)
  let settles = Int.of_float (Float.round_up (0.7 *. Float.of_int walk)) in
  printf "settled at step %d: %b\n" settles (at settles = at (walk - 1));
  [%expect
    {|
    step 0: 15099494 = floor of 0.9 * 2^24
    the floor: 1677721 = floor of 0.1 * 2^24
    the schedule only falls: true
    settled at step 359: true
    |}]
;;

let%expect_test "a canvas becomes the frames of the wire" =
  (* pitch 60 at the bass and pitch 81 at the soprano: the bass lands in the low byte, the
     two silent seats carry no pitch, and the events follow the rule of the frame *)
  let canvas =
    [| [| class_of_cell 60; Vocab.silence; Vocab.silence; class_of_cell 81 |] |]
  in
  let frames = frames_of_canvas canvas in
  printf "%08x\n" frames.(0);
  [%expect {| d10000bc |}]
;;

let%expect_test "the chorale turns around: the file gives the soprano first" =
  let chorale = { Jsb.cells = [| [ 74; 70; 65; 58 ] |]; legal_shifts = [ 0 ] } in
  print_s ([%sexp_of: int array array] (classes_of_chorale chorale));
  [%expect {| ((23 30 35 39)) |}];
  printf
    "%s\n"
    (Checkpoint.refusal (fun () ->
       ignore
         (classes_of_chorale
            { Jsb.cells = [| [ 20; -1; -1; -1 ] |]; legal_shifts = [ 0 ] })));
  [%expect {| the pitch 20 is outside the corpus's 36 to 81 |}]
;;

let%expect_test "the canvases and the masks of Gate A are deterministic" =
  let piece length =
    { Jsb.cells = Array.create ~len:length [ 74; 70; 65; 58 ]; legal_shifts = [ 0 ] }
  in
  (* a piece shorter than the gate's crop is dropped, as the JAX side drops it *)
  let canvases = gate_canvases [ piece 4; piece 2; piece 5 ] ~steps:3 in
  printf "3 pieces hold %d gate canvases of 3 steps\n" (Array.length canvases);
  let mask = gate_mask ~index:0 ~steps:128 in
  let again = gate_mask ~index:0 ~steps:128 in
  let other = gate_mask ~index:1 ~steps:128 in
  let same = Array.equal (Array.equal Bool.equal) in
  printf "one index, one mask: %b\n" (same mask again);
  printf "another index, another mask: %b\n" (not (same mask other));
  let hidden = Array.sum (module Int) mask ~f:(fun row -> Array.count row ~f:Fn.id) in
  let share = Float.of_int hidden /. Float.of_int (128 * voices) in
  printf "a Bernoulli half hides near half: %b\n" Float.(share > 0.4 && share < 0.6);
  [%expect
    {|
    3 pieces hold 2 gate canvases of 3 steps
    one index, one mask: true
    another index, another mask: true
    a Bernoulli half hides near half: true
    |}]
;;

let%expect_test "the loss refuses a canvas with nothing hidden" =
  let config = { Config.layers = 4; width = 6 } in
  let params = Params.init config ~seed:3 in
  let classes = Array.make_matrix ~dimx:4 ~dimy:voices 0 in
  let hidden = Array.make_matrix ~dimx:4 ~dimy:voices false in
  printf
    "%s\n"
    (Checkpoint.refusal (fun () -> ignore (masked_nll params ~classes ~hidden)));
  [%expect {| the masked loss takes one hidden cell or more |}]
;;

let%expect_test "the convolution against a direct loop, at the shapes that caught it" =
  (* THE TRAP THIS PINS: [Nx.concatenate] misread a strided slice of a padded tensor and
     rolled the step axis by one — at [4; 48] and not at [4; 5], thus a small hand test
     passed while the real roll shifted. The [Nx.contiguous] on each tap of [conv] is the
     fix, and this loop is the referee. *)
  let check ~steps ~pitches ~inputs ~outputs =
    let rand = Random.State.make [| 99 |] in
    let x =
      Nx.init Nx.float32 [| steps; pitches; inputs |] (fun (_ : int array) ->
        Random.State.float_range rand (-1.0) 1.0)
    in
    let k =
      Nx.init Nx.float32 [| kernel; kernel; inputs; outputs |] (fun (_ : int array) ->
        Random.State.float_range rand (-1.0) 1.0)
    in
    let y = conv x k in
    let xa = Nx.to_array x
    and ka = Nx.to_array k in
    let worst = ref 0.0 in
    for t = 0 to steps - 1 do
      for p = 0 to pitches - 1 do
        for co = 0 to outputs - 1 do
          let acc = ref 0.0 in
          for dy = 0 to 2 do
            for dx = 0 to 2 do
              let ts = t + dy - 1
              and ps = p + dx - 1 in
              if ts >= 0 && ts < steps && ps >= 0 && ps < pitches
              then
                for ci = 0 to inputs - 1 do
                  acc
                  := !acc
                     +. (xa.((((ts * pitches) + ps) * inputs) + ci)
                         *. ka.((((((dy * 3) + dx) * inputs) + ci) * outputs) + co))
                done
            done
          done;
          worst := Float.max !worst (Float.abs (!acc -. Nx.item [ t; p; co ] y))
        done
      done
    done;
    printf
      "%d by %d, %d to %d channels: within 1e-5 of the loop: %b\n"
      steps
      pitches
      inputs
      outputs
      Float.(!worst < 1e-5)
  in
  check ~steps:4 ~pitches:5 ~inputs:2 ~outputs:3;
  check ~steps:4 ~pitches:48 ~inputs:2 ~outputs:3;
  check ~steps:4 ~pitches:48 ~inputs:8 ~outputs:6;
  [%expect
    {|
    4 by 5, 2 to 3 channels: within 1e-5 of the loop: true
    4 by 48, 2 to 3 channels: within 1e-5 of the loop: true
    4 by 48, 8 to 6 channels: within 1e-5 of the loop: true
    |}]
;;
