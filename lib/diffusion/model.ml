(* The model as the circuit reads it — see model.mli, and docs/diffusion_rtl.md for the
   design. *)

open Core
module Nn_quantized = Mgen_nn.Quantized

(* ==================================================================== *)
(* The roll *)
(* ==================================================================== *)

let rows = Vocab.classes
let voices = Frame.voices
let planes = 2 * voices

(* ==================================================================== *)
(* The formats *)
(* ==================================================================== *)

(* THE ACTIVATION FORMAT IS Q6 IN INT16, AND IT IS MEASURED. The trunk is a residual stack
   with no norm on the stream, thus a trained model's activations GROW with depth: the
   golden candidate peaks at 184 on half-masked corpus sheets and at 313 on the seeded
   openings the walk really visits — measured 2026-08-25, and the reason the opening
   design of the counters exists. Q6 holds 512 with a 1.6 margin over that peak; what its
   1/64 resolution costs is the drift report's to say, and a per-layer exponent (which the
   gain shift would absorb for free) is the refinement of the next round if the verdict is
   poor. *)
let activation_q = 6

(* the widths of the format, beside its Q: an activation is int16 and the products of one
   dwell sum into int32 *)
let activation_bits = 16
let accumulator_bits = 32

(* the bottom rail of the format, which is also the opening value of the draw's peak scan:
   nothing a logit can be stands below it. The clamp's own rails live in [Nn_quantized] —
   [int16_high] and [int16_low] — because the clamp is the repository's and not this
   era's. *)
let activation_low = -(1 lsl (activation_bits - 1))

(* the accumulator is exact below this width: 9 C products of int8 by int16 reach 9 * 57 *
   127 * 32767, under 2^31, and one channel more can pass it *)
let widest_inputs = 57

(* ==================================================================== *)
(* The model *)
(* ==================================================================== *)

type quantized = Nn_quantized.quantized

type layer =
  { kernel : quantized
  ; gain : Nn_quantized.Constants.scale array
  ; bias : int array
  ; inputs : int
  ; outputs : int
  }

type t =
  { layers : layer array
  ; temper : Nn_quantized.Constants.scale
  }

(* The two tensors of the file that are not layers. They stand beside the numbered layers
   and not inside [__metadata__] BECAUSE [Nx_io] GIVES NO ACCESS TO IT. *)
let temper_tensor = "temper"
let activation_tensor = "activation_q"

(* the tensors one layer holds, in the order of the file *)
let layer_tensors = 5

let check_shape { layers; temper = (_ : Nn_quantized.Constants.scale) } =
  let count = Array.length layers in
  if count < 3 then invalid_argf "%d layers is no sheet model" count ();
  if count % 2 <> 0 then invalid_argf "%d layers hold no whole residual pairs" count ();
  if layers.(0).inputs <> planes
  then invalid_argf "the stem reads %d planes, not %d" layers.(0).inputs planes ();
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

(* THE CONTRACT FILE, READ. [jax/diffusion/quantized.py] writes it and its docstring holds
   the layout.

   EVERY TENSOR IS INT32, INCLUDING THE KERNEL, because [Nx_io.load_safetensors] SKIPS
   every dtype it does not hold — an int8 kernel would arrive as a hole and the model
   would refuse for the wrong reason. The values are the int8 image all the same. *)
let of_int8_checkpoint path =
  let archive = Nx_io.load_safetensors path in
  let packed name =
    match Stdlib.Hashtbl.find_opt archive name with
    | None -> invalid_argf "%s has no tensor %s" path name ()
    | Some packed -> packed
  in
  let values name =
    Array.map (Nx.to_array (Nx_io.to_typed Nx.int32 (packed name))) ~f:Int32.to_int_exn
  in
  let only name =
    match values name with
    | [| value |] -> value
    | row ->
      invalid_argf "%s: %s holds %d values, not one" path name (Array.length row) ()
  in
  let count = Stdlib.Hashtbl.length archive in
  let layers, spare = (count - 2) / layer_tensors, (count - 2) % layer_tensors in
  if spare <> 0 || layers < 1
  then invalid_argf "%s: %d tensors is no quantized sheet model" path count ();
  let stated = only activation_tensor in
  if stated <> activation_q
  then
    invalid_argf
      "%s is quantized at Q%d and this reader states Q%d"
      path
      stated
      activation_q
      ();
  let layer at =
    let name index = Int.to_string ((layer_tensors * at) + index) in
    (* the reach of one convolution is three by three over time and pitch *)
    let inputs, outputs =
      match Nx_io.packed_shape (packed (name 0)) with
      | [| 3; 3; inputs; outputs |] -> inputs, outputs
      | shape ->
        invalid_argf
          "the kernel of layer %d of %s is %s, and not a 3 by 3 kernel"
          at
          path
          (Sexp.to_string ([%sexp_of: int array] shape))
          ()
    in
    { kernel = { Nn_quantized.q = values (name 0); e = only (name 1) }
    ; gain =
        Array.map2_exn
          (values (name 2))
          (values (name 3))
          ~f:(fun q_value q -> { Nn_quantized.Constants.q_value; q })
    ; bias = values (name 4)
    ; inputs
    ; outputs
    }
  in
  let temper =
    match values temper_tensor with
    | [| q_value; q |] -> { Nn_quantized.Constants.q_value; q }
    | row ->
      invalid_argf "%s: the temper holds %d values, not two" path (Array.length row) ()
  in
  let model = { layers = Array.init layers ~f:layer; temper } in
  check_shape model;
  model
;;

let rom_tensors { layers; temper = (_ : Nn_quantized.Constants.scale) } =
  Array.to_list (Array.map layers ~f:(fun { kernel; _ } -> kernel))
;;

let rom_bits model = Nn_quantized.rom_bits (rom_tensors model)

(* the exclusive prefix scan: the ROM's bases are one reading of it and the elaboration's
   banks are the others *)
let bases_of sizes =
  Array.folding_map sizes ~init:0 ~f:(fun base size -> base + size, base)
;;

let rom_bases model =
  bases_of
    (Array.of_list_map (rom_tensors model) ~f:(fun { q; e = (_ : int) } -> Array.length q))
;;

(* ==================================================================== *)
(* The walk *)
(* ==================================================================== *)

(* the grid of the generator: a uniform is [k * 2 ** -24], thus [u * grid] is the integer
   [k] and every compare over it is exact in a double *)
let grid = Float.of_int (1 lsl Prng.uniform_bits)

(* the register of each seat, seat 0 the bass: [Jsb.voice_ranges] turned around, because
   the corpus table gives the soprano first. They are the RANGES of [jax/measure.py]. *)
let seat_ranges = Array.of_list (List.rev (Array.to_list Jsb.voice_ranges))

type opening =
  { low : int
  ; width : int
  }

(* [seat_ranges] as classes: the row of a pitch is its distance above [Vocab.pitch_low],
   and row 0 is silence *)
let seat_openings =
  Array.map seat_ranges ~f:(fun (low, high) ->
    { low = low - Vocab.pitch_low + 1; width = high - low + 1 })
;;

let anneal_threshold ~step ~walk =
  let low = 0.1
  and high = 0.9 in
  (* the trainer's own expression, thus both language sides evaluate the same doubles *)
  let alpha =
    Float.max
      low
      (high -. ((high -. low) *. Float.of_int step /. (0.7 *. Float.of_int walk)))
  in
  Int.of_float (Float.round_down (alpha *. grid))
;;

(* a threshold compare on the 24-bit grid: exact, because [u * grid] is an integer *)
let under ~threshold u = Float.(u *. grid < of_int threshold)

(* one step at a time, and the seats inside a step. Every uniform of the walk is drawn in
   this order, thus the twin takes the same walk by taking the same order. *)
let cell_order ~steps = List.cartesian_product (List.range 0 steps) (List.range 0 voices)

(* one uniform for each cell in the cell order, folded into [f] *)
let over_cells state ~steps ~f =
  List.fold (cell_order ~steps) ~init:state ~f:(fun state (step, voice) ->
    let next, value = Prng.run Prng.uniform state in
    f ~step ~voice value;
    next)
;;

let opening_sheet state ~steps =
  let sheet = Array.make_matrix ~dimx:steps ~dimy:voices 0 in
  (* [u * width] is exact on the grid, thus the twin and the circuit state one class from
     one uniform *)
  let state =
    over_cells state ~steps ~f:(fun ~step ~voice u ->
      let { low; width } = seat_openings.(voice) in
      sheet.(step).(voice)
      <- low + Int.of_float (Float.round_down (u *. Float.of_int width)))
  in
  state, sheet
;;

let hidden_cells state ~steps ~threshold =
  let hidden = Array.make_matrix ~dimx:steps ~dimy:voices false in
  let state =
    over_cells state ~steps ~f:(fun ~step ~voice u ->
      hidden.(step).(voice) <- under ~threshold u)
  in
  state, hidden
;;

(* ==================================================================== *)
(* The frames *)
(* ==================================================================== *)

let frames_of_sheet sheet =
  Array.map sheet ~f:(fun step -> Vocab.frame_of_classes (Array.to_list step))
;;

(* ==================================================================== *)
(* The drawn model *)
(* ==================================================================== *)

module For_test = struct
  (* Nothing below the seam reads a kernel's exponent — the ROM carries [q] alone — thus
     one constant serves every drawn layer. *)
  let drawn_exponent = 14

  (* a third of the byte, thus three sigma still fits and the clamp is rare *)
  let kernel_spread = 42.0

  (* a quarter of the one, thus the channels differ and none of them dominates *)
  let gain_spread = 0.25
  let bias_spread = Float.of_int (1 lsl activation_q) /. 4.0

  (* THE DRAWN MODEL HOLDS ITS TRUNK AT O(1) ACTIVATIONS, and that is the whole of the
     arithmetic here. A kernel byte of spread [s] over the [9 * C] taps of a dwell carries
     an activation of magnitude A into an accumulator of about [sqrt (9 C) * s * A], thus
     the gain must be [1 / (sqrt (9 C) * s)] to take it back. A gain drawn flat inside
     int16 would clamp every write of the trunk or zero it. *)
  let multiplier ~inputs = 1.0 /. (Float.sqrt (Float.of_int (9 * inputs)) *. kernel_spread)

  (* the multiplier at a quarter of int16: the largest shift that keeps four times it
     inside the format, thus a drawn gain varies around it and no [q_value] leaves *)
  let gain_shift value =
    let rec largest e =
      if Float.iround_nearest_exn (Float.ldexp value e) <= 8191 then e else largest (e - 1)
    in
    largest 30
  ;;

  let clamp16 = Nn_quantized.clamp16
  let clamp_byte v = Int.clamp_exn v ~min:(-127) ~max:127

  let drawn ~layers ~width ~seed =
    (* the stem widens the planes, the trunk holds the width, the head narrows to the
       voices *)
    let channels =
      List.init layers ~f:(fun at ->
        (if at = 0 then 2 * voices else width), if at = layers - 1 then voices else width)
    in
    let layer (inputs, outputs) =
      let open Prng in
      let scale = multiplier ~inputs in
      let shift = gain_shift scale in
      let* weights = normals ~count:(9 * inputs * outputs) ~scale:kernel_spread in
      let* gains = normals ~count:outputs ~scale:gain_spread in
      let+ biases = normals ~count:outputs ~scale:bias_spread in
      { kernel =
          { Nn_quantized.q =
              Array.map weights ~f:(fun w -> clamp_byte (Float.iround_nearest_exn w))
          ; e = drawn_exponent
          }
      ; gain =
          Array.map gains ~f:(fun g ->
            { Nn_quantized.Constants.q_value =
                clamp16
                  (Float.iround_nearest_exn (Float.ldexp (scale *. (1.0 +. g)) shift))
            ; q = shift
            })
      ; bias = Array.map biases ~f:(fun b -> clamp16 (Float.iround_nearest_exn b))
      ; inputs
      ; outputs
      }
    in
    let (_ : Prng.state), built =
      Prng.run (Prng.all (List.map channels ~f:layer)) (Prng.create_folded ~seed)
    in
    let model =
      { layers = Array.of_list built
      ; temper = fst (Nn_quantized.policy ~temperature:1.0 ~min_p:0.0)
      }
    in
    check_shape model;
    model
  ;;

  let rom_tensors = rom_tensors
  let over_cells = over_cells
end

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the drawn model holds its shape" =
  (* L 4 by H 6: the smallest shape with the era's structure — the stem, one residual pair
     and the head *)
  let model = For_test.drawn ~layers:4 ~width:6 ~seed:11 in
  printf "check_shape: %s\n" (Mgen_nn.Checkpoint.refusal (fun () -> check_shape model));
  let sizes =
    List.map (For_test.rom_tensors model) ~f:(fun { q; e = (_ : int) } -> Array.length q)
  in
  print_s ([%sexp_of: int list] sizes);
  print_s ([%sexp_of: int array] (rom_bases model));
  printf "rom bytes: %d\n" (Array.length (rom_bits model));
  [%expect
    {|
    check_shape: no raise
    (432 324 324 216)
    (0 432 756 1080)
    rom bytes: 1296
    |}]
;;

let%expect_test "a broken model refuses loudly" =
  let model = For_test.drawn ~layers:4 ~width:6 ~seed:11 in
  let refuse broken =
    printf "%s\n" (Mgen_nn.Checkpoint.refusal (fun () -> check_shape broken))
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

let%expect_test "the opening puts every voice inside the register of its seat" =
  (* the draw is over the seat's range and never the whole roll *)
  let (_ : Prng.state), sheet = opening_sheet (Prng.create_folded ~seed:9) ~steps:64 in
  let inside =
    Array.for_all sheet ~f:(fun step ->
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

let%expect_test "a sheet becomes the frames of the wire" =
  (* pitch 60 at the bass and pitch 81 at the soprano: the bass lands in the low byte and
     the two silent seats carry no pitch *)
  let class_of_pitch pitch = pitch - Vocab.pitch_low + 1 in
  let sheet =
    [| [| class_of_pitch 60; Vocab.silence; Vocab.silence; class_of_pitch 81 |] |]
  in
  let frames = frames_of_sheet sheet in
  printf "%08x\n" frames.(0);
  [%expect {| d10000bc |}]
;;
