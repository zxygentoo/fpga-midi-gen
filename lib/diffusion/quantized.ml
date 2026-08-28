(* The int8 checkpoint as data — see quantized.mli for the contract and
   docs/diffusion_rtl.md for the formats. The arithmetic of the twin is
   jax/diffusion/quantized.py; what stands here is the reader of the file it writes, the
   ROM image the bitstream carries, and the formats every unit of the circuit slices on. *)

open Core
module Nn_quantized = Mgen_nn.Quantized

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

(* The WIDTHS of the format, beside its Q: an activation is int16, and the products of one
   dwell sum into int32. They stand here because they are the other half of the sentence
   [activation_q] opens — a unit that stated 16 of its own could part from the twin and no
   compiler would say so — thus every unit of the circuit reads the format in one place. *)
let activation_bits = 16
let accumulator_bits = 32

module Model = struct
  type quantized = Mgen_nn.Quantized.quantized

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

  (* The tensors of the contract file that are not layers: the sampling temper, and the Q
     the file was quantized at. They stand beside the numbered layers and not inside
     [__metadata__] BECAUSE [Nx_io] GIVES NO ACCESS TO IT — the loader hands back the
     tensors alone — and the elaboration needs both numbers. *)
  let temper_tensor = "temper"
  let activation_tensor = "activation_q"

  (* the tensors one layer holds, in the order of the file *)
  let layer_tensors = 5

  (* The int32 accumulator of the machine is exact below this width: 9 C products of int8
     by int16 reach 9 * 57 * 127 * 32767, which stands under 2^31, and one channel more
     can pass it. The elected shapes stand far under; the rule stands so the prose cannot
     rot. *)
  let widest_inputs = 57

  let check_shape { layers; temper = (_ : Nn_quantized.Constants.scale) } =
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

  (* THE CONTRACT FILE, READ. [jax/diffusion/quantized.py] writes it and this is its only
     reader: the quantization happens above the seam, one time, and the file carries the
     result. Its layout and the reasons behind it are that module's docstring.

     EVERY TENSOR IS INT32, INCLUDING THE KERNEL. It is a fact of the reader and not a
     taste: [Nx_io.load_safetensors] holds F32, F64, I32, F16 and BF16 and SKIPS every
     other dtype with a warning on stderr, thus an int8 kernel would arrive here as a hole
     and the model would refuse for the wrong reason. The values are the int8 image all
     the same, and [check_shape] and [Elaboration.norm_word] hold every range. *)
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
    then invalid_argf "%s: %d tensors is no quantized canvas model" path count ();
    let stated = only activation_tensor in
    if stated <> activation_q
    then
      invalid_argf
        "%s is quantized at Q%d and this twin reads Q%d"
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

  let rom_bases model =
    let sizes =
      List.map (rom_tensors model) ~f:(fun { q; e = (_ : int) } -> Array.length q)
    in
    Array.of_list (List.folding_map sizes ~init:0 ~f:(fun base size -> base + size, base))
  ;;

  module For_test = struct
    let config = { Diffusion.Config.layers = 4; width = 6 }

    (* THE EXPONENT OF A DRAWN KERNEL. Nothing below the seam reads it — the ROM carries
       [q] alone and the norm word carries the gain's own shift — thus one constant serves
       every drawn layer and its bytes stand for [q * 2^-14]. *)
    let drawn_exponent = 14

    (* the spread of a drawn kernel byte: a third of the byte, thus a draw of three sigma
       still fits and the clamp of [quantize] is rare *)
    let kernel_spread = 42.0

    (* the spread of a drawn channel gain around one, and of a drawn bias in the
       activation format: a quarter of the one, thus the channels differ and none of them
       dominates *)
    let gain_spread = 0.25
    let bias_spread = Float.of_int (1 lsl activation_q) /. 4.0

    (* THE DRAWN MODEL HOLDS ITS TRUNK AT O(1) ACTIVATIONS, AND THAT IS THE WHOLE OF THE
       ARITHMETIC HERE. A kernel byte of spread [s] over the [9 * C] taps of a dwell
       carries an activation of magnitude A into an accumulator of about
       [sqrt (9 C) * s * A]; the gain has to take it back to A, thus the multiplier is
       [1 / (sqrt (9 C) * s)].

       A GAIN DRAWN FLAT INSIDE INT16 WOULD SAY NOTHING. It would clamp every write of the
       trunk or zero it, and the pictures, the frames and the cycle counts of the thirteen
       tests that read a drawn model would all read a machine that no checkpoint makes.
       This is the argument [jax/diffusion/model.py]'s [drawn_params] makes in floats, in
       the integers the file now carries. *)
    let multiplier ~inputs =
      1.0 /. (Float.sqrt (Float.of_int (9 * inputs)) *. kernel_spread)
    ;;

    (* The shift that puts the multiplier at a quarter of int16: the largest that keeps
       four times it inside the format, thus a drawn channel gain varies around it and no
       [q_value] leaves the width. It is the 16-bit exponent rule of the eras with the
       headroom the draw wants. *)
    let gain_shift value =
      let rec largest e =
        if Float.iround_nearest_exn (Float.ldexp value e) <= 8191
        then e
        else largest (e - 1)
      in
      largest 30
    ;;

    let clamp16 = Nn_quantized.clamp16
    let clamp_byte v = Int.clamp_exn v ~min:(-127) ~max:127

    let drawn { Diffusion.Config.layers = count; width } ~seed =
      (* the input and output channels of each layer: the stem widens the planes, the
         trunk holds the width, and the head narrows to the voices *)
      let channels =
        List.init count ~f:(fun at ->
          (if at = 0 then 2 * voices else width), if at = count - 1 then voices else width)
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
      let (_ : Prng.state), layers =
        Prng.run (Prng.all (List.map channels ~f:layer)) (Prng.create_folded ~seed)
      in
      let model =
        { layers = Array.of_list layers
        ; temper = fst (Nn_quantized.policy ~temperature:1.0 ~min_p:0.0)
        }
      in
      check_shape model;
      model
    ;;

    let rom_tensors = rom_tensors
  end
end

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the quantized model holds its shape" =
  let model = Model.For_test.drawn Model.For_test.config ~seed:11 in
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
  let model = Model.For_test.drawn Model.For_test.config ~seed:11 in
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
