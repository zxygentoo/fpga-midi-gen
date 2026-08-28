(* The shared integer rules — see quantized.mli for the contract and the design. *)

open Base

module Constants = struct
  let h_q = 16
  let y_q = 12

  (* the feed-forward hidden after its ReLU *)
  let hid_q = 10

  (* the rms epsilon of the float models, in the Q of the squared stream: the sum squares
     a Q12 copy, thus the mean is Q(2 y_q) *)
  let eps_q = Float.iround_nearest_exn (Float.ldexp 1e-6 (2 * y_q))

  (* A fixed-point multiplier: the value stands for [q_value * 2^-q]. The Q travels with
     the value because the two are one fact — a multiply that takes the wrong shift is
     silently wrong, and both the twins and the circuits apply these scales. *)
  type scale =
    { q_value : int
    ; q : int
    }

  (* [apply s v] scales [v] by [s], toward negative infinity — an arithmetic shift, as the
     circuits'. *)
  let apply { q_value; q } v = (v * q_value) asr q

  (* log2(e): the exp2 form of an exponential *)
  let log2e =
    let q = 15 in
    { q_value = Float.iround_nearest_exn (Float.ldexp (1.0 /. Float.log 2.0) q); q }
  ;;

  (* THE TEMPER AT TEMPERATURE 1: log2(e) at the temper's own Q, one below [log2e]'s.

     The temper is log2(e) / T, and the spare bit is headroom for the temperature: the
     circuits multiply by this constant on an 18-bit signed port, thus [log2e]'s own Q
     would overflow that port under a temperature of about 0.36, and this Q holds down to
     about 0.18. [jax/nn.py]'s [temper_of] states the rule for every temperature and
     [jax/tests/test_quantized.py] pins this reading of it.

     A model of a CONTRACT FILE reads its temper from the file. A DRAWN model has no
     training run behind it, thus it states this one. *)
  let temper_at_one = { q_value = 23637; q = log2e.q - 1 }

  (* the quantized exponential: exp2 of -j/256 in Q15 — one table serves the softmax and
     the sampler of era four and the decay of era five *)
  let exp2_table =
    Array.init 256 ~f:(fun j ->
      Float.iround_nearest_exn Float.(32768.0 * (2.0 ** (-of_int j / 256.0))))
  ;;

  (* The sigmoid of a Q12 value, in Q15. The input is int16, thus its range is |v| < 8
     exactly and a clamp costs nothing: 256 buckets of 256 Q12 units cover it, and the
     index is the top eight bits with the sign flipped.

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
  let exp2_bits = bits16 exp2_table
  let sigmoid_bits = bits16 sigmoid_table
  let softplus_bits = bits16 softplus_table

  (* The index rules, stated once for the twins and the ROMs.

     [sigmoid_index] is the top eight bits of an int16 with the sign bit flipped, which is
     no arithmetic at all in a circuit. [softplus_index] is the magnitude shifted, and the
     clamp catches the one value -32768 whose magnitude does not fit the table. *)
  let sigmoid_index v = ((v asr 8) + 128) land 255
  let softplus_index v = Int.min 255 (Int.abs v asr 7)

  (* The two rules of the attention head. A raw score is a product of two Q[row_q] rows;
     the shift brings it to Q[y_q] and applies the 1/sqrt(head_d) of the twins in the same
     move, thus the scale costs no multiply and [head_d] is a power of four. [row_q] is
     the Q of the scored rows — both eras store their rings in Q12 and each names that
     format itself.

     The ALiBi slope of head [head] is 2^-(this), thus the penalty of an age is a shift of
     the age and no multiply pays for it either. *)
  let score_shift ~row_q ~head_d = (2 * row_q) - y_q + (Int.floor_log2 head_d / 2)
  let slope_exponent ~span ~heads ~head = span * (head + 1) / heads
end

module Tensor = struct
  type t = int array
end

(* ==================================================================== *)
(* The scalar rules of the engines *)
(* ==================================================================== *)

(* THE RAILS OF INT16, NAMED ONE TIME. The two scalar clamps here, the circuit below and
   era six's clamps read them from these two values, thus a unit that wrote a rail of its
   own could not part from the twin in silence. The frozen eras still write 32767 out at
   [transformer/source.ml] and [mamba/source.ml]; adoption there moves their netlists, and
   it belongs to their own round. *)
let int16_high = 32767
let int16_low = -32768
let clamp16 v = Int.clamp_exn v ~min:int16_low ~max:int16_high

(* the MAC as a reduction: the sum of [f i] over [0 .. n - 1] *)
let sum n f =
  let rec go acc i = if i = n then acc else go (acc + f i) (i + 1) in
  go 0 0
;;

(* floor of the square root; any correct algorithm gives the one answer the circuits must
   also give *)
let isqrt n =
  if n <= 0
  then 0
  else (
    let rec shrink g = if g * g > n then shrink (g - 1) else g in
    let rec grow g = if (g + 1) * (g + 1) <= n then grow (g + 1) else g in
    grow (shrink (Float.to_int (Float.sqrt (Float.of_int n)))))
;;

(* Exp2 of a nonnegative Q12 magnitude, giving 2^-m in Q15: the integer part shifts, the
   top eight bits of the fraction index the table. The peak — a magnitude of 0 — is 2^15.

   [exp2_q] is the same rule over a Q12 value that is 0 or less: era four exponentiates a
   nonpositive score and era five a decay that is a magnitude by construction, thus the
   negation stands at the caller there and does not stand at all here. One definition
   holds the two readings to one table. *)
let exp2_of_magnitude m =
  let i = m asr 12 in
  if i >= 16 then 0 else Constants.exp2_table.((m asr 4) land 255) asr i
;;

let exp2_q u = exp2_of_magnitude (-u)

(* the sigmoid of a Q12 value in Q15, and SiLU over it: one table read, one multiply and
   one shift by the Q of the sigmoid *)
let sigmoid_q v = Constants.sigmoid_table.(Constants.sigmoid_index v)
let silu v = clamp16 ((v * sigmoid_q v) asr 15)

(* softplus as the ramp and the correction the table holds. The sum rides an int16, thus
   the input clamps before the table reads it and the result clamps after. *)
let softplus v =
  let v = clamp16 v in
  clamp16 (Int.max 0 v + Constants.softplus_table.(Constants.softplus_index v))
;;

(* ==================================================================== *)
(* The rules as circuits *)
(* ==================================================================== *)

module Rtl = struct
  open Hardcaml.Signal

  (* the width the clamp writes: the int16 its rails name *)
  let bits = 16

  (* [clamp16] of the twin, as a circuit: the value saturates and never wraps. A wrap here
     would be silently wrong music, and the clamp is what the format election stands on.
     THE COMPARE STANDS AT THE OPERAND'S OWN WIDTH — an [sresize ~width:32] before it
     would truncate a 48-bit product and hand a wrapped value to a clamp that then sees
     nothing to clamp. *)
  let clamp16 wide =
    (* the rails twice over: at the width of the value for the compare, and at 16 bits for
       what the clamp writes *)
    let clamped value = of_signed_int ~width:bits value in
    let high = of_signed_int ~width:(width wide) int16_high in
    let low = of_signed_int ~width:(width wide) int16_low in
    mux2
      (wide >+ high)
      (clamped int16_high)
      (mux2 (wide <+ low) (clamped int16_low) (sresize wide ~width:bits))
  ;;
end

(* ==================================================================== *)
(* The quantization of a checkpoint *)
(* ==================================================================== *)

type quantized =
  { q : Tensor.t
  ; e : int
  }

let rom_bits tensors =
  Array.concat_map (Array.of_list tensors) ~f:(fun { q; e = (_ : int) } ->
    Array.map q ~f:(fun v -> Hardcaml.Bits.of_unsigned_int ~width:8 (v land 255)))
;;

(* the exclusive prefix scan: a ROM's bases are one reading of it and the elaborations'
   banks are the others *)
let bases_of sizes =
  Array.folding_map sizes ~init:0 ~f:(fun base size -> base + size, base)
;;

(* ==================================================================== *)
(* The integer draw *)
(* ==================================================================== *)

(* the 24-bit uniform of a draw: three bytes of the generator, high first *)
let u24 prng =
  let open Prng in
  run
    (let* high = next in
     let* middle = next in
     let+ low = next in
     (((high * 256) + middle) * 256) + low)
    prng
;;

let draw ~weights prng =
  let count = Array.length weights in
  let total = sum count (fun c -> weights.(c)) in
  let prng, u = u24 prng in
  let threshold = (u * total) asr 24 in
  let rec walk c running =
    if c = count - 1
    then c
    else (
      let running = running + weights.(c) in
      if running > threshold then c else walk (c + 1) running)
  in
  prng, Float.of_int u *. 0x1p-24, walk 0 0
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

(* These rules decide every byte the boards hold. The frame gates of the eras read them
   only through walks of tens of thousands of cycles, thus a break there says "the frames
   disagree" and says nothing about which rule broke. *)
let%expect_test "the clamp saturates at both rails, at a narrow width and at a wide one" =
  (* THE 48-BIT CASE IS THE ONE THIS FORM EXISTS FOR. Era six's epilogue clamps a gain
     product 48 bits wide; a clamp that resized to 32 before the compare would wrap 2^47
     down to something inside the rails and pass it. Width 20 is the other end: the rails
     still fit the operand, thus the compare is the same compare. *)
  let clamped ~width:w value =
    let wide = Hardcaml.Signal.input "wide" w in
    let circuit =
      Hardcaml.Circuit.create_exn
        ~name:"clamp16"
        [ Hardcaml.Signal.output "clamped" (Rtl.clamp16 wide) ]
    in
    let sim = Hardcaml.Cyclesim.create circuit in
    Hardcaml.Cyclesim.in_port sim "wide" := Hardcaml.Bits.of_signed_int ~width:w value;
    Hardcaml.Cyclesim.cycle sim;
    Hardcaml.Bits.to_signed_int !(Hardcaml.Cyclesim.out_port sim "clamped")
  in
  let at ~width:w =
    let show value = Stdio.printf "%d -> %d\n" value (clamped ~width:w value) in
    Stdio.printf "at width %d:\n" w;
    (* the rails themselves pass, and one step past each saturates. The extremes of the
       operand are the wrap the compare must see. *)
    List.iter
      [ int16_high
      ; int16_high + 1
      ; int16_low
      ; int16_low - 1
      ; 0
      ; (1 lsl (w - 1)) - 1
      ; -(1 lsl (w - 1))
      ]
      ~f:show
  in
  at ~width:20;
  at ~width:48;
  [%expect
    {|
    at width 20:
    32767 -> 32767
    32768 -> 32767
    -32768 -> -32768
    -32769 -> -32768
    0 -> 0
    524287 -> 32767
    -524288 -> -32768
    at width 48:
    32767 -> 32767
    32768 -> 32767
    -32768 -> -32768
    -32769 -> -32768
    0 -> 0
    140737488355327 -> 32767
    -140737488355328 -> -32768
    |}]
;;

let%expect_test "the exp2 table: the peak, the floor and the halving" =
  (* entry 0 is the peak 2^15; a full fractional step halves; the last entry sits one
     table step above one half *)
  Stdio.printf
    "%d %d %d  half at one: %d\n"
    Constants.exp2_table.(0)
    Constants.exp2_table.(128)
    Constants.exp2_table.(255)
    (exp2_q (-4096));
  [%expect {| 32768 23170 16428  half at one: 16384 |}]
;;

let%expect_test "isqrt floors" =
  List.iter [ 0; 1; 2; 3; 4; 15; 16; 17; 1_000_000 ] ~f:(fun n ->
    Stdio.printf "%d " (isqrt n));
  Stdio.printf "\n";
  [%expect {| 0 1 1 1 2 3 4 4 1000 |}]
;;

let%expect_test "the sigmoid table: the ends, the middle and the symmetry" =
  let show v =
    Stdio.printf
      "  %6d (%.4f) -> %5d (%.4f)\n"
      v
      (Float.of_int v /. 4096.0)
      (sigmoid_q v)
      (Float.of_int (sigmoid_q v) /. 32768.0)
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
  (* against the float function it stands for: relu(v) + ln(1+exp(-|v|)) *)
  let wider (worst, at) v =
    let float_v = Float.of_int v /. 4096.0 in
    let want =
      Float.max 0.0 float_v +. Float.log (1.0 +. Float.exp (-.Float.abs float_v))
    in
    let gap = Float.abs (want -. (Float.of_int (softplus v) /. 4096.0)) in
    if Float.(gap > worst) then gap, v else worst, at
  in
  let worst, at =
    Sequence.range (-32768) 32768 |> Sequence.fold ~init:(0.0, 0) ~f:wider
  in
  Stdio.printf
    "over every int16 input the table stands within %.5f of the float softplus, worst at \
     %.4f\n"
    worst
    (Float.of_int at /. 4096.0);
  [%expect
    {| over every int16 input the table stands within 0.00784 of the float softplus, worst at 0.0000 |}]
;;
