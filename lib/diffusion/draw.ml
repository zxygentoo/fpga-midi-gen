(* The draw — see draw.mli for the contract and docs/diffusion_rtl.md, "The walk", for its
   place. What stands here is the WHY of each rule.

   Three walks of one column and one uniform. The walks share ONE class counter and ONE
   mux into the logits, thus the width of a cell costs one multiplexer and not three. *)

open Core
open Hardcaml
open Signal
module Nn_quantized = Mgen_nn.Quantized

module type Shape = sig
  val classes : int
end

(* the Q the exp2 unit reads, and the Q the twin's activations carry *)
let exp2_q = 12
let activation_bits = 16

(* the magnitude port of [Exp2]: a wider value saturates into it, and the saturation is
   exact because the unit already gives zero at a magnitude of 16 and above *)
let magnitude_bits = 22

(* the grid of the generator: the uniform is [k * 2 ** -24] *)
let uniform_bits = 24

(* one weight of the pick, in Q15 *)
let weight_bits = 16

module State = struct
  type t =
    | Idle
    | Peak
    | Weigh
    | Threshold
    | Pick
  [@@deriving compare ~localize, enumerate, sexp_of]
end

module Make (Shape : Shape) = struct
  let classes = Shape.classes
  let class_bits = address_bits_for classes

  (* the two table walks count to [classes] and not to [classes] - 1: the last cycle
     retires the weight of the last class, which the table states one cycle behind its
     magnitude *)
  let counter_bits = address_bits_for (classes + 1)

  (* every weight is a Q15 value at most, thus the total of them all needs this many *)
  let total_bits = Int.ceil_log2 ((classes * (1 lsl 15)) + 1)

  (* the peak walk, then the weights and their total, then the pick — and one cycle for
     the threshold between the last two. A table walk takes one cycle more than its
     classes, because the weight of the last class stands one cycle behind its magnitude. *)
  let busy_cycles = classes + (classes + 1) + 1 + (classes + 1)

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a
      ; logits : 'a [@bits classes * activation_bits]
      ; uniform : 'a [@bits uniform_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { busy : 'a
      ; drawn : 'a [@bits class_bits]
      }
    [@@deriving hardcaml]
  end

  let create ~(temper : Nn_quantized.Constants.scale) (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    (* the datapath holds no clear: the caller waits, thus a stale value reaches nothing *)
    let dspec = Reg_spec.create ~clock:i.clock () in
    let open Always in
    let sm = State_machine.create (module State) spec in
    let counter = Variable.reg spec ~width:counter_bits in
    let peak = Variable.reg dspec ~width:activation_bits in
    let total = Variable.reg dspec ~width:total_bits in
    let threshold = Variable.reg dspec ~width:total_bits in
    let running = Variable.reg dspec ~width:total_bits in
    let found = Variable.reg spec ~width:1 in
    let drawn = Variable.reg dspec ~width:class_bits in
    (* THE ONE MUX INTO THE LOGITS. All three walks read the class the counter names, thus
       one [classes]-way multiplexer serves the peak, the weights and the pick. *)
    (* THE ARRAY OWNS THE DSPS AND THIS UNIT TAKES NONE, thus the rule is an attribute and
       not a hope: a 24 by 21 variable product is exactly what the tools infer a DSP48
       for, and the fused rung at G 5 needs all 240 of them for the lanes. draw.mli states
       the reason; this states it to Vivado. *)
    let no_dsp product = add_attribute product (Rtl_attribute.Vivado.use_dsp false) in
    let logit =
      mux
        counter.value
        (List.init classes ~f:(fun at ->
           select
             i.logits
             ~high:((at * activation_bits) + activation_bits - 1)
             ~low:(at * activation_bits)))
    in
    (* The magnitude of one class, the twin's rule in one expression: the difference
       against the peak — never above zero — shifts up to the Q the table reads, takes the
       temper, and NEGATES AFTER THE SCALE. Negating before it parts from the twin by one
       unit wherever the scale does not divide, which era five's head round measured. *)
    let magnitude =
      let difference =
        sresize logit ~width:(activation_bits + 1)
        -: sresize peak.value ~width:(activation_bits + 1)
      in
      let shifted =
        let rise = exp2_q - Quantized.activation_q in
        sll (sresize difference ~width:(activation_bits + 1 + rise)) ~by:rise
      in
      let tempered =
        sra (no_dsp (shifted *+ of_signed_int ~width:18 temper.q_value)) ~by:temper.q
      in
      let wide = negate tempered in
      let ceiling = (1 lsl magnitude_bits) - 1 in
      mux2
        (wide >+ of_signed_int ~width:(width wide) ceiling)
        (of_unsigned_int ~width:magnitude_bits ceiling)
        (select wide ~high:(magnitude_bits - 1) ~low:0)
    in
    let { Exp2.O.e } = Exp2.create { Exp2.I.clock = i.clock; nn = magnitude } in
    let weight = uresize e ~width:total_bits in
    let last_class = counter.value ==:. classes - 1 in
    let walked = counter.value ==:. classes in
    (* the class whose weight stands on the wire: the fork of [Exp2] answers one cycle
       behind the magnitude, thus the retiring class is the counter of the cycle before *)
    let retiring = reg spec counter.value in
    let retires = counter.value <>:. 0 in
    let passes = running.value +: weight >: threshold.value in
    compile
      [ sm.switch
          [ ( State.Idle
            , [ when_
                  i.start
                  [ counter <--. 0
                  ; peak <-- of_signed_int ~width:activation_bits (-32768)
                  ; sm.set_next Peak
                  ]
              ] )
          ; ( Peak
            , [ when_ (logit >+ peak.value) [ peak <-- logit ]
              ; counter <-- counter.value +:. 1
              ; when_ last_class [ counter <--. 0; total <--. 0; sm.set_next Weigh ]
              ] )
          ; ( Weigh
            , [ when_ retires [ total <-- total.value +: weight ]
              ; counter <-- counter.value +:. 1
              ; when_ walked [ sm.set_next Threshold ]
              ] )
          ; ( Threshold
            , [ (* the pick's own rule: the uniform times the total, on the generator's
                   grid. It stands STRICTLY under the total, thus a class always passes
                   and no last-class arm stands anywhere below. *)
                threshold
                <-- select
                      (no_dsp (i.uniform *: total.value))
                      ~high:(uniform_bits + total_bits - 1)
                      ~low:uniform_bits
              ; counter <--. 0
              ; running <--. 0
              ; found <--. 0
              ; sm.set_next Pick
              ] )
          ; ( Pick
            , [ when_
                  retires
                  [ running <-- running.value +: weight
                  ; when_
                      (passes &: ~:(found.value))
                      [ found <-- vdd; drawn <-- sel_bottom retiring ~width:class_bits ]
                  ]
              ; counter <-- counter.value +:. 1
              ; when_ walked [ sm.set_next Idle ]
              ] )
          ]
      ];
    { O.busy = ~:(sm.is Idle); drawn = drawn.value }
  ;;
end

(* ==================================================================== *)
(* The bench *)
(* ==================================================================== *)

(* THE REFERENCE IS THE TWIN, CALLED. [Quantized.draw_cell] is the engine's own draw, thus
   the gate compares the circuit against the arithmetic the walk really takes and never
   against a second reading of it. The twin takes its uniform from [Prng]; the circuit
   takes the 24 bits the walk hands it, thus the bench draws the same three bytes and
   states them both ways. *)
module Bench (Shape : Shape) = struct
  module Drawer = Make (Shape)
  module Sim = Cyclesim.With_interface (Drawer.I) (Drawer.O)

  let classes = Shape.classes

  (* the model the twin's [draw_cell] reads: it takes the temper alone, thus a bench model
     carries no layer *)
  let model temper = { Quantized.Model.layers = [||]; temper }

  (* [run ~temper logits ~uniform] is the class the circuit draws, and the cycles it took
     from [start] to the fall of [busy]. *)
  let run ~temper logits ~uniform =
    let sim = Sim.create (Drawer.create ~temper) in
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs sim in
    inp.logits
    := Bits.concat_lsb (List.map (Array.to_list logits) ~f:(Bits.of_signed_int ~width:16));
    inp.uniform := Bits.of_unsigned_int ~width:24 uniform;
    inp.start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.start := Bits.gnd;
    let cycles = ref 0 in
    while Bits.to_bool !(out.busy) do
      Cyclesim.cycle sim;
      Int.incr cycles
    done;
    Bits.to_unsigned_int !(out.drawn), !cycles
  ;;

  (* [check ~temper logits ~seed] states the twin's class, the circuit's, and whether the
     two agree. The twin's [draw_cell] takes the very uniform the circuit is handed. *)
  let check ~temper logits ~seed =
    let prng = Prng.create ~seed in
    (* the three bytes the twin's own draw would take, read out as the walk reads them *)
    let uniform_bits =
      let state, byte0 = Prng.run Prng.next prng in
      let state, byte1 = Prng.run Prng.next state in
      let (_ : Prng.state), byte2 = Prng.run Prng.next state in
      (byte0 lsl 16) lor (byte1 lsl 8) lor byte2
    in
    let (_ : Prng.state), (_ : float), twin =
      Quantized.For_test.draw_cell (model temper) logits prng
    in
    let circuit, cycles = run ~temper logits ~uniform:uniform_bits in
    twin, circuit, cycles
  ;;
end

let%expect_test "the draw states the class the twin states" =
  (* THE FUZZ, against [Quantized.For_test.draw_cell] itself. The logits run the whole
     int16 both ways, thus the differences reach the saturation the table needs and the
     weights reach zero; and the tempers include ONE THAT DOES NOT DIVIDE — era five's
     head round measured that a scale which divides exactly hides the difference between
     negating before it and negating after, and the twin negates after. *)
  let case ~classes ~temper ~name ~cells =
    let module B =
      Bench (struct
        let classes = classes
      end)
    in
    let take (state, disagree) seed =
      let state, logits = Fuzz.draw_array state ~len:classes ~limit:32767 in
      let twin, circuit, (_ : int) = B.check ~temper logits ~seed in
      state, if twin = circuit then disagree else disagree + 1
    in
    let (_ : Prng.state), disagree =
      List.fold (List.range 1 (cells + 1)) ~init:(Prng.create ~seed:3, 0) ~f:take
    in
    printf "%d classes, %s, %d cells: %d disagree\n" classes name cells disagree
  in
  let one = fst (Nn_quantized.policy ~temperature:1.0 ~min_p:0.0) in
  (* a temper whose shift does not divide its value: the reading that negates before the
     scale parts from the twin by one unit here and nowhere else *)
  let ragged = { Nn_quantized.Constants.q_value = 23637; q = 13 } in
  case ~classes:48 ~temper:one ~name:"the elected temper" ~cells:60;
  case ~classes:48 ~temper:ragged ~name:"a temper that does not divide" ~cells:60;
  case ~classes:8 ~temper:one ~name:"the elected temper" ~cells:40;
  [%expect
    {|
    48 classes, the elected temper, 60 cells: 0 disagree
    48 classes, a temper that does not divide, 60 cells: 0 disagree
    8 classes, the elected temper, 40 cells: 0 disagree
    |}]
;;

let%expect_test "the pick lands by the last class, and costs the cycles it states" =
  (* THE TOP OF THE GRID. A uniform of 2^24 - 1 states the largest threshold the rule can
     make, and the theorem says it still stands STRICTLY under the total — thus the pick
     lands THROUGH the running totals and never on a last-class arm, which this circuit
     does not hold. The cycles are the number [busy_cycles] states, always. *)
  let module B =
    Bench (struct
      let classes = 48
    end)
  in
  let temper = fst (Nn_quantized.policy ~temperature:1.0 ~min_p:0.0) in
  (* the peak stands at class 0 and every other class weighs far less, thus the top of the
     grid must still land inside the totals *)
  let steep = Array.init 48 ~f:(fun at -> if at = 0 then 3000 else -3000) in
  let flat = Array.create ~len:48 100 in
  let show name logits =
    let drawn, cycles = B.run ~temper logits ~uniform:((1 lsl 24) - 1) in
    printf "%s at the top of the grid: class %d in %d cycles\n" name drawn cycles
  in
  show "a steep column" steep;
  show "a flat column" flat;
  printf "busy_cycles states %d\n" B.Drawer.busy_cycles;
  [%expect
    {|
    a steep column at the top of the grid: class 0 in 147 cycles
    a flat column at the top of the grid: class 47 in 147 cycles
    busy_cycles states 147
    |}]
;;
