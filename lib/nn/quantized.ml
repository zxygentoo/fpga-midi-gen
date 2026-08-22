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
     silently wrong, and both the references and the circuits apply these scales. *)
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

  (* The index rules, stated once for the references and the ROMs.

     [sigmoid_index] is the top eight bits of an int16 with the sign bit flipped, which is
     no arithmetic at all in a circuit. [softplus_index] is the magnitude shifted, and the
     clamp catches the one value -32768 whose magnitude does not fit the table. *)
  let sigmoid_index v = ((v asr 8) + 128) land 255
  let softplus_index v = Int.min 255 (Int.abs v asr 7)

  (* The two rules of the attention head. A raw score is a product of two Q[row_q] rows;
     the shift brings it to Q[y_q] and applies the 1/sqrt(head_d) of the references in the
     same move, thus the scale costs no multiply and [head_d] is a power of four. [row_q]
     is the Q of the scored rows — both eras store their rings in Q12 and each names that
     format itself.

     The ALiBi slope of head [head] is 2^-(this), thus the penalty of an age is a shift of
     the age and no multiply pays for it either. *)
  let score_shift ~row_q ~head_d = (2 * row_q) - y_q + (Int.floor_log2 head_d / 2)
  let slope_exponent ~span ~heads ~head = span * (head + 1) / heads
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

(* ==================================================================== *)
(* The scalar rules of the engines *)
(* ==================================================================== *)

let clamp16 v = Int.clamp_exn v ~min:(-32768) ~max:32767
let clamps16 v = v > 32767 || v < -32768

(* the reductions of the engines: [sum n f] is the MAC — the sum of [f i] over
   [0 .. n - 1] — and [max_over n f] is the peak scan *)
let sum n f =
  let rec go acc i = if i = n then acc else go (acc + f i) (i + 1) in
  go 0 0
;;

let max_over n f =
  let rec go acc i = if i = n then acc else go (Int.max acc (f i)) (i + 1) in
  go Int.min_value 0
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
(* The quantization of a checkpoint *)
(* ==================================================================== *)

type quantized =
  { q : Tensor.t
  ; e : int
  }

let max_abs (floats : Tensor.floats) =
  Array.fold floats ~init:0.0 ~f:(fun acc v -> Float.max acc (Float.abs v))
;;

(* the largest exponent that keeps round(max|w| * 2^e) at 127 or less; 14 caps the
   all-zero tensor *)
let max_exponent v =
  let fits e = Float.iround_nearest_exn (Float.ldexp v e) <= 127 in
  (* [fits] falls monotonically in [e], thus the first [e] that fits is the largest *)
  let rec largest e = if fits e then e else largest (e - 1) in
  if Float.(v <= 0.0) then 14 else largest 14
;;

(* [e] overrides the exponent of the tensor's own peak — tensors whose rows add share one *)
let quantize ?e (floats : Tensor.floats) =
  let e = Option.value e ~default:(max_exponent (max_abs floats)) in
  let clamp ft =
    Int.clamp_exn (Float.iround_nearest_exn (Float.ldexp ft e)) ~min:(-127) ~max:127
  in
  { q = Array.map floats ~f:clamp; e }
;;

let rom_bits tensors =
  Array.concat_map (Array.of_list tensors) ~f:(fun { q; e = (_ : int) } ->
    Array.map q ~f:(fun v -> Hardcaml.Bits.of_unsigned_int ~width:8 (v land 255)))
;;

(* The policy in the integer forms of the machines; the rules of the float sampler. The
   temper is log2(e) / T, and its Q is one below the Q of [Constants.log2e]. The extra bit
   is headroom for the temperature: the circuits multiply by this constant on an 18-bit
   signed port, thus the Q of [log2e] would overflow that port under a temperature of
   about 0.36, and this Q holds down to about 0.18. *)
let policy ~temperature ~min_p =
  Policy.check_policy ~temperature ~min_p;
  let q = Constants.log2e.q - 1 in
  ( { Constants.q_value =
        Float.iround_nearest_exn (Float.ldexp (1.0 /. Float.log 2.0 /. temperature) q)
    ; q
    }
  , Float.iround_nearest_exn (min_p *. 32768.0) )
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
let%expect_test "the exponent of a tensor, and the clamp of the byte" =
  (* the largest e that keeps the peak at 127 or less. 14 caps the all-zero tensor, where
     every exponent fits, and 127.5 is the rounding boundary: it rounds to 128 and the
     exponent has to step down. *)
  List.iter [ 0.0; 0.02; 0.08; 127.0; 127.49; 127.5; 1e9 ] ~f:(fun v ->
    Stdio.printf "%-6g -> %d\n" v (max_exponent v));
  (* The byte is two's complement and the negative end is not used: the clamp is -127 and
     not -128, thus the image is symmetric and a negated weight is a negated byte. A tie
     rounds up and never away from zero, thus -5.5 is -5. *)
  let { q; e } = quantize ~e:0 [| 200.0; -200.0; 5.4; -5.5; 0.0 |] in
  Stdio.printf "at e %d: %s\n" e (Sexp.to_string ([%sexp_of: int array] q));
  (* with no exponent given, the tensor's own peak states it *)
  let { q; e } = quantize [| 0.02; -0.01; 0.0 |] in
  Stdio.printf "at its own e %d: %s\n" e (Sexp.to_string ([%sexp_of: int array] q));
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
    at its own e 12: (82 -41 0)
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
  (* against the float function the references state: relu(v) + ln(1+exp(-|v|)) *)
  let worst = ref 0.0 in
  let at = ref 0 in
  for v = -32768 to 32767 do
    let float_v = Float.of_int v /. 4096.0 in
    let want =
      Float.max 0.0 float_v +. Float.log (1.0 +. Float.exp (-.Float.abs float_v))
    in
    let gap = Float.abs (want -. (Float.of_int (softplus v) /. 4096.0)) in
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
