(* L3 of the diffusion source — see epilogue.mli for the contract and
   docs/diffusion_rtl.md, "The epilogue", for the design. What stands here is the WHY of
   each rule.

   The unit is the tail of the twin's [layer_forward] and nothing else: [lanes] identical
   lanes, each one a norm, a ReLU, a clamp and — where the layer closes a pair — a
   residual add under a second clamp. It holds no memory and no walk. *)

open Core
open Hardcaml
open Signal
module Nn_quantized = Mgen_nn.Quantized

module type Shape = sig
  val rows : int
  val lanes : int
end

(* The pipe: the multiply, then the shift with the bias, then the ReLU with the first
   clamp, then the residual with the second. A 32 by 16 multiply and a 48-bit variable
   shift do not stand in one cycle at 100 MHz beside each other — and ring 3 read the
   shift, the bias and the clamp together at 19 levels, failing in context — thus the
   stages are five and the tag rides beside them. The split is the round's licensed
   reserve: no caller counts the depth, because the tag travels.

   THE MULTIPLY ITSELF TOOK TWO OF THE FIVE at the fused rung 3. In LUTs — the array owns
   every DSP — a 32 by 16 product is the widest cone of this unit, and the census of the
   golden candidate named it at −0.090 over ten endpoints. It stands in two stages now,
   split by the GAIN'S BYTES, and the product is the same product. *)
let latency = 5

(* the activation format the twin states, and the accumulator the array hands over *)
let activation_bits = Model.activation_bits
let accumulator_bits = Model.accumulator_bits

(* the whole gain product: an int32 sum by the gain the norm word carries *)
let product_bits = accumulator_bits + Elaboration.gain_bits

(* [clamp16] of the twin, as a circuit: the value saturates and never wraps. A wrap here
   would be silently wrong music, and the clamp is what the format election stands on. *)
let clamp16 wide =
  (* the rails twice over: at the width of the value for the compare, and at the
     activation format for what the clamp writes *)
  let clamped value = of_signed_int ~width:activation_bits value in
  let high = of_signed_int ~width:(width wide) Model.activation_high in
  let low = of_signed_int ~width:(width wide) Model.activation_low in
  mux2
    (wide >+ high)
    (clamped Model.activation_high)
    (mux2
       (wide <+ low)
       (clamped Model.activation_low)
       (sresize wide ~width:activation_bits))
;;

module Make (Shape : Shape) = struct
  let lanes = Shape.lanes
  let row_bits = address_bits_for Shape.rows

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; drained : 'a
      ; row : 'a [@bits row_bits]
      ; sums : 'a [@bits lanes * accumulator_bits]
      ; residual : 'a [@bits lanes * activation_bits]
      ; norms : 'a [@bits lanes * Elaboration.norm_bits]
      ; relu : 'a
      ; join : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { valid : 'a
      ; activation_row : 'a [@bits row_bits]
      ; activations : 'a [@bits lanes * activation_bits]
      }
    [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    (* the datapath registers have no clear: what is real is what [drained] marks, and the
       tag that clears is what says so *)
    let dspec = Reg_spec.create ~clock:i.clock () in
    let lane at =
      let slice signal bits =
        select signal ~high:((at * bits) + bits - 1) ~low:(at * bits)
      in
      (* THE FIELD ORDER IS THE ELABORATION'S, SLICED BY THE ELABORATION. The packer and
         this slicing are one rule over [Comb], thus a word that changed shape could not
         be packed one way and read another. *)
      let { Elaboration.gain; shift; bias } =
        Elaboration.Rtl.norm_fields (slice i.norms Elaboration.norm_bits)
      in
      (* STAGES 1 AND 2 — the gain multiply, split by the gain's bytes. The product of an
         int32 sum and an int16 gain is 48 bits, and it is LUTs and never a DSP: the array
         owns those.

         THE SPLIT IS EXACT OVER THE INTEGERS. A signed gain is its signed high byte times
         256 plus its UNSIGNED low byte, thus [s * g] is [s *+ g_high] shifted eight plus
         [s *+ g_low] — the low half carried as a non-negative signed value, which is what
         the leading zero states. No value moves; one register does. *)
      let accumulated = slice i.sums accumulator_bits in
      let gain_high =
        reg dspec (Column_array.no_dsp (accumulated *+ sel_top gain ~width:8))
      in
      let gain_low =
        reg dspec (Column_array.no_dsp (accumulated *+ (gnd @: sel_bottom gain ~width:8)))
      in
      let product =
        reg
          dspec
          (sresize (gain_high @: zero 8) ~width:product_bits
           +: sresize gain_low ~width:product_bits)
      in
      (* STAGE 3 — [Constants.apply] with the bias. The shift is VARIABLE because a gain
         carries its own q, and it goes toward negative infinity as the twin's arithmetic
         shift does. *)
      let scaled = log_shift ~f:sra product ~by:(pipeline dspec ~n:2 shift) in
      let biased =
        reg dspec (scaled +: sresize (pipeline dspec ~n:2 bias) ~width:(width scaled))
      in
      (* STAGE 4 — the ReLU and the first clamp *)
      let ramped =
        mux2
          (pipeline dspec ~n:3 i.relu)
          (mux2 (biased <+. 0) (zero (width biased)) biased)
          biased
      in
      let conv = reg dspec (clamp16 ramped) in
      (* STAGE 5 — the join. THE TWIN CLAMPS TWICE: it writes the convolution through its
         counted clamp and then writes the sum through it again, thus a value that rode
         the first clamp and then meets a residual gives a different answer under one
         clamp than under two. Gate B is bit for bit. *)
      let with_residual =
        sresize conv ~width:(activation_bits + 1)
        +: sresize
             (pipeline dspec ~n:4 (slice i.residual activation_bits))
             ~width:(activation_bits + 1)
      in
      let joined =
        clamp16 (mux2 (with_residual <+. 0) (zero (width with_residual)) with_residual)
      in
      reg dspec (mux2 (pipeline dspec ~n:4 i.join) joined conv)
    in
    (* the tag clears and the datapath does not: what is real is what [valid] marks *)
    { O.valid = pipeline spec ~n:latency i.drained
    ; activation_row = pipeline spec ~n:latency i.row
    ; activations = concat_lsb (List.init lanes ~f:lane)
    }
  ;;
end

(* ==================================================================== *)
(* The bench *)
(* ==================================================================== *)

(* THE REFERENCE IS THE TWIN, CALLED: [Constants.apply] and [clamp16] are the twin's own
   functions, thus the gate compares the circuit against the arithmetic itself and never
   against a second copy of it. *)
module Bench (Shape : Shape) = struct
  module Lane = Make (Shape)
  module Sim = Cyclesim.With_interface (Lane.I) (Lane.O)

  let lanes = Shape.lanes
  let rows = Shape.rows

  (* one row of work: the accumulators, the norms and the residual of each lane *)
  type row =
    { sums : int array
    ; norms : Nn_quantized.Constants.scale array
    ; biases : int array
    ; residual : int array
    }

  let expected { sums; norms; biases; residual } ~relu ~join =
    Array.init lanes ~f:(fun at ->
      let value = Nn_quantized.Constants.apply norms.(at) sums.(at) + biases.(at) in
      let value = if relu then Int.max value 0 else value in
      let conv = Nn_quantized.clamp16 value in
      if join then Nn_quantized.clamp16 (Int.max 0 (residual.(at) + conv)) else conv)
  ;;

  (* THE BENCH PACKS WITH THE ELABORATION'S OWN PACKER and never with a copy of it. A
     mirror of the epilogue's slicing would agree with the epilogue whichever order the
     image really takes, thus it would pass a flipped word — the lesson the weight image's
     gate already learned. *)
  let pack_norms norms biases =
    Bits.concat_lsb
      (List.init lanes ~f:(fun at -> Elaboration.norm_word norms.(at) ~bias:biases.(at)))
  ;;

  (* [run rows ~relu ~join] drives the rows back to back and gives what left the pipe,
     beside the tag each row entered with. *)
  let run work ~relu ~join =
    let sim = Sim.create Lane.create in
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs sim in
    let given = ref [] in
    let cycle () =
      Cyclesim.cycle sim;
      if Bits.to_bool !(out.valid)
      then (
        let activations = Harness.unpack !(out.activations) ~width:activation_bits in
        given := (Bits.to_unsigned_int !(out.activation_row), activations) :: !given);
      inp.drained := Bits.gnd
    in
    inp.relu := if relu then Bits.vdd else Bits.gnd;
    inp.join := if join then Bits.vdd else Bits.gnd;
    (* the tag is a row of the drain, thus it walks 0 to [rows] - 1 and begins again *)
    List.iteri work ~f:(fun at { sums; norms; biases; residual } ->
      inp.drained := Bits.vdd;
      Harness.set inp.row (at % rows);
      inp.sums := Harness.pack sums ~width:accumulator_bits;
      inp.residual := Harness.pack residual ~width:activation_bits;
      inp.norms := pack_norms norms biases;
      cycle ());
    for _ = 1 to latency + 1 do
      cycle ()
    done;
    List.rev !given
  ;;

  (* [check rows ~relu ~join] names what disagreed: the tags out of order, the wrong count
     of rows, or an activation the twin does not state. *)
  let check work ~relu ~join =
    let given = run work ~relu ~join in
    let wanted = List.map work ~f:(expected ~relu ~join) in
    let tags = List.mapi work ~f:(fun at (_ : row) -> at % rows) in
    let counted = List.length given = List.length wanted in
    Harness.verdict
      [ "tags", not (List.equal Int.equal (List.map given ~f:fst) tags)
      ; "count", not counted
      ; ( "activations"
        , not
            (counted
             && List.for_all2_exn (List.map given ~f:snd) wanted ~f:(fun a b ->
               Array.equal Int.equal a b)) )
      ]
  ;;
end

let%expect_test "the epilogue states what the twin's layer tail states" =
  (* THE FUZZ, over the three shapes a layer can take. The sums run the whole int32 range
     the array can hand over, the gains and the shifts the whole range the quantizer can
     state, and the biases and the residual the whole int16 — thus the clamps fire, which
     is the point: a value that never rides a clamp tests nothing about the clamps.

     [relu] with [join] is not among them: a pair-closing convolution runs with no ReLU of
     its own and the head takes neither, thus the pair never stands. *)
  (* [work_rows] is how many drained rows the case drives, and it is NOT the shape's
     [rows] two lines below: the tag walks 0 to 47 and begins again, thus a case may drive
     more rows than a drain holds. *)
  let case ~lanes ~relu ~join ~work_rows =
    let module B =
      Bench (struct
        let rows = 48
        let lanes = lanes
      end)
    in
    let draw_row state =
      (* the rail the array's own gate proved it reaches, not a comfortable fraction of
         it: the multiply's top operand bit is exercised and not argued *)
      let state, sums =
        Prng.For_test.draw_array state ~len:lanes ~limit:((1 lsl 31) - 1)
      in
      let state, gains = Prng.For_test.draw_array state ~len:lanes ~limit:32767 in
      let state, shifts = Prng.For_test.draw_array state ~len:lanes ~limit:22 in
      let state, biases = Prng.For_test.draw_array state ~len:lanes ~limit:32767 in
      let state, residual = Prng.For_test.draw_array state ~len:lanes ~limit:32767 in
      ( state
      , { B.sums
        ; norms =
            Array.mapi gains ~f:(fun at q_value ->
              (* a shift is never negative — [Elaboration.create] refuses one — thus the
                 draw takes its magnitude *)
              { Nn_quantized.Constants.q_value; q = Int.abs shifts.(at) })
        ; biases
        ; residual
        } )
    in
    let draw state (_ : int) = draw_row state in
    let (_ : Prng.state), drawn =
      List.fold_map (List.range 0 work_rows) ~init:(Prng.create ~seed:5) ~f:draw
    in
    printf
      "G %d, relu %b, join %b, %d rows: %s\n"
      lanes
      relu
      join
      work_rows
      (B.check drawn ~relu ~join)
  in
  (* the stem and a pair's opening layer *)
  case ~lanes:4 ~relu:true ~join:false ~work_rows:200;
  (* a pair's closing layer *)
  case ~lanes:4 ~relu:false ~join:true ~work_rows:200;
  (* the head *)
  case ~lanes:4 ~relu:false ~join:false ~work_rows:200;
  (* the geometries beside the elected one *)
  case ~lanes:1 ~relu:true ~join:false ~work_rows:60;
  case ~lanes:5 ~relu:false ~join:true ~work_rows:60;
  [%expect
    {|
    G 4, relu true, join false, 200 rows: ok
    G 4, relu false, join true, 200 rows: ok
    G 4, relu false, join false, 200 rows: ok
    G 1, relu true, join false, 60 rows: ok
    G 5, relu false, join true, 60 rows: ok
    |}]
;;

let%expect_test "the second clamp is arithmetic and not hygiene" =
  (* THE DOUBLE CLAMP, ON A ROW THAT SHOWS IT. A convolution that overflows POSITIVE
     meeting a NEGATIVE residual is where the two readings part: the twin clamps the
     convolution to 32767 first, thus the sum is 22767; a circuit that clamped one time
     would carry the whole 40000 into the sum and state 30000. Both stand inside int16,
     thus the second clamp does not catch the difference — it IS the difference. *)
  let module B =
    Bench (struct
      let rows = 48
      let lanes = 1
    end)
  in
  let row =
    { B.sums = [| 40_000 |]
    ; norms = [| { Nn_quantized.Constants.q_value = 1; q = 0 } |]
    ; biases = [| 0 |]
    ; residual = [| -10_000 |]
    }
  in
  let one_clamp =
    let value =
      Nn_quantized.Constants.apply row.norms.(0) row.sums.(0) + row.biases.(0)
    in
    Nn_quantized.clamp16 (Int.max 0 (row.residual.(0) + value))
  in
  printf "the circuit and the twin: %s\n" (B.check [ row ] ~relu:false ~join:true);
  printf
    "the twin states %d; one clamp would state %d\n"
    (B.expected row ~relu:false ~join:true).(0)
    one_clamp;
  [%expect
    {|
    the circuit and the twin: ok
    the twin states 22767; one clamp would state 30000
    |}]
;;
