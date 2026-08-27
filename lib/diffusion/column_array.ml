(* L2 of the diffusion source — see column_array.mli for the contract and
   docs/diffusion_rtl.md, "The dwell" and "The drain", for the design. What stands here is
   the WHY of each rule.

   The unit is one op shape and nothing else: P by G lanes that take one term each cycle,
   and the chain that carries their accumulators out one row a cycle. It knows no layer,
   no memory and no walk — the caller states the terms and the array states the drained
   rows.

   THE THREE-COLUMN WINDOW STANDS OUTSIDE THIS UNIT, with the store it caches. A window is
   a read cache for the column port: what fills it, when a slot is free, and what the zero
   column is beyond the ends of the roll are all questions of the memory and of the walk
   that reads it. Holding the registers here and the policy there would put a load strobe
   on one side of an interface and the dwell it must be timed against on the other. The
   path is the same either way — the caller's window register through its time mux, then
   this unit's pitch mux into the operand register — thus the cut costs no logic and no
   stage. *)

open Core
open Hardcaml
open Signal
module Nn_quantized = Mgen_nn.Quantized

module type Shape = sig
  val rows : int
  val lanes : int
end

(* Cycles from a [term] strobe to the edge that takes its term into the accumulator: the
   operand registers, then the product register. The DSP48E1 packs as A, B, M and P, thus
   the depth is the primitive's own and not a choice. *)
let accumulate_latency = 2

(* Cycles from [term_last] to the first [drained]. The accumulator settles one cycle
   behind its last term, the chain takes it on that cycle's edge, and the bottom stage
   stands one cycle later. *)
let first_row_latency = accumulate_latency + 2

(* the activation format of the stream and the accumulator behind it, read from the twin
   that states them; the weight byte is the quantizer's own rail *)
let activation_bits = Quantized.activation_bits
let accumulator_bits = Quantized.accumulator_bits
let weight_bits = 8

(* THE ARRAY OWNS THE DSPS AND EVERY OTHER UNIT PINS ITS PRODUCTS AWAY FROM THEM. The
   fused rung at G 5 is 48 by 5, which is the device's whole 240, thus a multiply that
   drifted into a DSP anywhere else would have to move at the moment the design is
   tightest. The rule is an attribute and not a hope — and it stands HERE, beside the
   lanes that hold the primitives, rather than once in each unit that obeys it. *)
let no_dsp product = add_attribute product (Rtl_attribute.Vivado.use_dsp false)

(* ONE COPY OF A BROADCAST, KEPT APART FROM ITS SIBLINGS. The rule is an attribute and not
   a hope, as [no_dsp] is, and it stands here for the same reason: this unit is the one
   that holds the primitives, thus the family of Vivado rules the round leans on has one
   home rather than a statement in each unit that obeys it. *)
let replica copy = add_attribute copy (Rtl_attribute.Vivado.dont_touch true)

(* THE REPLICA SLICE — ring 3's rule, and the array's own scale is what imposes it. No net
   of this scale keeps a single driver: a bank stands as one register slice for each
   [slice_rows] rows, [dont_touch] so the tools neither merge the copies nor absorb them
   back into the primitives. The number is a placement fact of the device and not of a
   model, thus the walk that slices a column band reads it here and never states an 8 of
   its own. *)
let slice_rows = 8
let slices_for ~rows = (rows + slice_rows - 1) / slice_rows

module Make (Shape : Shape) = struct
  let rows = Shape.rows
  let lanes = Shape.lanes
  let row_bits = address_bits_for rows

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; term : 'a
      ; term_first : 'a
      ; term_last : 'a
      ; column : 'a [@bits rows * activation_bits]
      ; row_shift : 'a [@bits 2]
      ; weights : 'a [@bits lanes * weight_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { drained : 'a
      ; row : 'a [@bits row_bits]
      ; sums : 'a [@bits lanes * accumulator_bits]
      }
    [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    (* the datapath registers have no clear: what is real is what the strobes mark, and a
       stale value that no strobe reaches touches nothing. The DSP packs best with no
       reset. The control registers clear. *)
    let dspec = Reg_spec.create ~clock:i.clock () in
    let row_of column at =
      select
        column
        ~high:((at * activation_bits) + activation_bits - 1)
        ~low:(at * activation_bits)
    in
    (* The three pitch taps are wire shifts of the registered column: output row [at]
       takes input row [at + dx - 1], and zeros shift in at row 0 and row [rows - 1]. No
       arithmetic stands here — the shift is the wiring. *)
    let silent = zero activation_bits in
    let activation at =
      let below = if at = 0 then silent else row_of i.column (at - 1) in
      let here = row_of i.column at in
      let above = if at = rows - 1 then silent else row_of i.column (at + 1) in
      (* the contract states 0, 1 and 2; the fourth arm is the don't care of a 2-bit
         select and it takes the row itself *)
      mux i.row_shift [ below; here; above; here ]
    in
    let weight lane =
      select
        i.weights
        ~high:((lane * weight_bits) + weight_bits - 1)
        ~low:(lane * weight_bits)
    in
    (* The two broadcast trees: one activation serves the [lanes] of a row and one weight
       serves the [rows] of a channel, thus the operand registers stand once for each row
       and once for each channel and never once for each lane. *)
    let operand_a = Array.init rows ~f:(fun at -> reg dspec (activation at)) in
    (* THE WEIGHT REPLICA BANK — ring 3's second family. The design put a register after
       the ROM and one at the operand, and the tools absorbed the first into the BRAM's
       own output register and the second into each DSP's B port — leaving the broadcast
       net ONE driver of 528 pins with no flop in the fabric to replicate, at 12 ns of
       route. The bank IS the operand register, stated in fabric: one copy for each slice
       of rows, [dont_touch] so the tools neither merge the copies nor absorb them back
       into the primitives. The depth of the pipe does not move — the replica replaces the
       absorbed B register — and the B port runs direct into the multiplier, far inside a
       cycle beside a neighbouring flop. *)
    let slices = slices_for ~rows in
    let operand_b =
      Array.init lanes ~f:(fun lane ->
        Array.init slices ~f:(fun (_ : int) -> replica (reg dspec (weight lane))))
    in
    (* The operand and product registers FREE-RUN and only the sum is gated: that is how
       the DSP is meant to be driven, and a register that captures on a cycle no term
       marks captures nothing that counts. The tags ride the pipe beside the operands —
       the accumulator takes a term two cycles behind its strobe, thus the flags that gate
       it arrive two cycles behind as well. *)
    let taking = pipeline spec ~n:accumulate_latency i.term in
    let opening = pipeline spec ~n:accumulate_latency i.term_first in
    let accumulator =
      Array.init rows ~f:(fun at ->
        let a = operand_a.(at) in
        Array.init lanes ~f:(fun lane ->
          let b = operand_b.(lane).(at / slice_rows) in
          let product = reg dspec (a *+ b) in
          let term = sresize product ~width:accumulator_bits in
          reg_fb dspec ~enable:taking ~width:accumulator_bits ~f:(fun sum ->
            mux2 opening term (sum +: term))))
    in
    (* THE CAPTURE STANDS ON THE EDGE THAT CLOSES THE DWELL. The next dwell may open on
       the cycle behind [term_last] — its first term reaches the accumulator on the very
       edge that loads the chain — and the chain takes the value the register held BEFORE
       that edge, which is the finished sum. Thus the array never waits between two
       dwells. *)
    let pre_capture = pipeline spec ~n:accumulate_latency i.term_last in
    let capture = reg spec pre_capture in
    (* THE CAPTURE BANK — the reserve of ring 1, applied: the capture select reached every
       register of the chain from one flop, fanout 6 019, and its shift enable 6 144. One
       copy of the capture register for each chain stage, [dont_touch] so the equivalent
       registers survive synthesis: 48 drivers of about 128 pins each, laid out beside
       their own stage. The walk of the drain keeps [capture] itself. *)
    let capture_bank =
      Array.init rows ~f:(fun (_ : int) -> replica (reg spec pre_capture))
    in
    let open Always in
    let row = Variable.reg spec ~width:row_bits in
    let draining = Variable.reg spec ~width:1 in
    let last_row = row.value ==:. rows - 1 in
    compile
      [ if_
          capture
          [ draining <-- vdd; row <--. 0 ]
          [ when_
              draining.value
              [ if_ last_row [ draining <-- gnd ] [ row <-- row.value +:. 1 ] ]
          ]
      ];
    (* The chain: one stage for each row, [lanes] accumulators wide. The capture loads
       every stage at one time and each cycle after it every stage takes the stage above,
       thus NO VALUE CROSSES A MUX AND NO REGISTER REACHES FARTHER THAN ITS NEIGHBOUR —
       the regularity the timing risk of this design asks for. The rows leave in row
       order, which is the order the epilogue packs a column in. *)
    let stage_of at above =
      let loaded = concat_lsb (Array.to_list accumulator.(at)) in
      let take = capture_bank.(at) in
      reg
        dspec
        ~enable:(take |: draining.value)
        (mux2 take loaded (Option.value above ~default:loaded))
    in
    (* The chain builds from the top row down, thus each stage names the stage above it as
       a value and no wire stands anywhere in it. The top row has no stage above it and
       takes its own accumulator on both arms, thus its mux collapses. *)
    let chain =
      List.fold
        (List.rev (List.range 0 rows))
        ~init:[]
        ~f:(fun below at -> stage_of at (List.hd below) :: below)
    in
    { O.drained = draining.value; row = row.value; sums = List.hd_exn chain }
  ;;
end

(* ==================================================================== *)
(* The bench *)
(* ==================================================================== *)

(* One shape, the dwells driven into it, and the rows the chain gives back. THE REFERENCE
   IS THE TWIN'S INNER SUM restricted to one (column, group): the array holds no more than
   that, thus the gates need no model of their own. *)
module Bench (Shape : Shape) = struct
  module Lanes = Make (Shape)
  module Sim = Cyclesim.With_interface (Lanes.I) (Lanes.O)

  let rows = Shape.rows
  let lanes = Shape.lanes

  (* one (column, group) of work: the three columns of each input channel, and the weight
     byte of each (input channel, tap) at each lane *)
  type dwell =
    { columns : int array array array (** input channel, then window slot, then row *)
    ; weights : int array array array (** input channel, then tap, then lane *)
    }

  let expected { columns; weights } =
    let inputs = Array.length columns in
    let term ~at ~lane ~cin ~tap =
      let source = at + (tap % 3) - 1 in
      if source < 0 || source >= rows
      then 0
      else columns.(cin).(tap / 3).(source) * weights.(cin).(tap).(lane)
    in
    Array.init rows ~f:(fun at ->
      Array.init lanes ~f:(fun lane ->
        Nn_quantized.sum inputs (fun cin ->
          Nn_quantized.sum 9 (fun tap -> term ~at ~lane ~cin ~tap))))
  ;;

  (* [run dwells] drives the dwells BACK TO BACK — the next one opens on the cycle behind
     the [term_last] of the one before it, which is the hazard the capture rule stands on
     — and gives the drained rows in the order they left the chain. *)
  let run ?(trace = false) dwells =
    let sim =
      Sim.create
        ~config:(if trace then Cyclesim.Config.trace_all else Cyclesim.Config.default)
        Lanes.create
    in
    let waves, sim = Cyclesim.Waveform.create_if ~enabled:trace sim in
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs sim in
    let drained = ref [] in
    let cycle () =
      Cyclesim.cycle sim;
      if Bits.to_bool !(out.drained)
      then (
        let sums = Harness.unpack !(out.sums) ~width:accumulator_bits in
        drained := (Bits.to_unsigned_int !(out.row), sums) :: !drained);
      inp.term := Bits.gnd;
      inp.term_first := Bits.gnd;
      inp.term_last := Bits.gnd
    in
    (* The bench stands where the walk will: it holds the window itself, thus a term names
       its column and the array only shifts it. The term of tap [t] takes the column of
       the time tap [t / 3]. *)
    let multiply ~first ~last ~tap column weights =
      inp.term := Bits.vdd;
      inp.term_first := if first then Bits.vdd else Bits.gnd;
      inp.term_last := if last then Bits.vdd else Bits.gnd;
      inp.column := Harness.pack column ~width:activation_bits;
      inp.row_shift := Bits.of_unsigned_int ~width:2 (tap % 3);
      inp.weights := Harness.pack weights ~width:weight_bits;
      cycle ()
    in
    let channel ~inputs ~cin window weights =
      for tap = 0 to 8 do
        multiply
          ~first:(cin = 0 && tap = 0)
          ~last:(cin = inputs - 1 && tap = 8)
          ~tap
          window.(tap / 3)
          weights.(cin).(tap)
      done
    in
    let dwell { columns; weights } =
      let inputs = Array.length columns in
      Array.iteri columns ~f:(fun cin window -> channel ~inputs ~cin window weights)
    in
    List.iter dwells ~f:dwell;
    (* the chain still holds the last dwell: run it out *)
    for _ = 1 to first_row_latency + rows do
      cycle ()
    done;
    List.rev !drained, waves
  ;;

  let draw_dwell state ~inputs =
    let state, columns =
      Array.fold_map (Array.create ~len:inputs 0) ~init:state ~f:(fun state (_ : int) ->
        Array.fold_map (Array.create ~len:3 0) ~init:state ~f:(fun state (_ : int) ->
          Prng.For_test.draw_array state ~len:rows ~limit:511))
    in
    let state, weights =
      Array.fold_map (Array.create ~len:inputs 0) ~init:state ~f:(fun state (_ : int) ->
        Array.fold_map (Array.create ~len:9 0) ~init:state ~f:(fun state (_ : int) ->
          Prng.For_test.draw_array state ~len:lanes ~limit:127))
    in
    state, { columns; weights }
  ;;

  (* [check dwells] names what disagreed: the rows in the wrong order, the wrong number of
     them, or a value that is not the sum the twin takes. *)
  let check dwells =
    let drained, (_ : Hardcaml_waveterm.Waveform.t option) = run dwells in
    let wanted = List.concat_map dwells ~f:(fun d -> Array.to_list (expected d)) in
    let order = List.concat_map dwells ~f:(fun (_ : dwell) -> List.range 0 rows) in
    let given = List.map drained ~f:snd in
    let counted = List.length given = List.length wanted in
    let complain wrong name = if wrong then Some name else None in
    let complaints =
      List.filter_opt
        [ complain (not (List.equal Int.equal (List.map drained ~f:fst) order)) "order"
        ; complain (not counted) "count"
        ; complain
            (not (counted && List.for_all2_exn given wanted ~f:(Array.equal Int.equal)))
            "sums"
        ]
    in
    if List.is_empty complaints then "ok" else String.concat ~sep:", " complaints
  ;;
end

let%expect_test "the array sums what the twin sums" =
  (* The fuzz: columns inside the range the Q6 clamp allows and int8 weights, over the
     geometries a simulation can hold, over one to six input channels, and at the elected
     P 48 by G 4. THE DWELLS RUN BACK TO BACK — the next opens on the cycle behind the
     [term_last] of the one before — thus every case also holds the capture rule, which is
     the one hazard of the design. *)
  let case ~rows ~lanes ~inputs ~dwells =
    let module B =
      Bench (struct
        let rows = rows
        let lanes = lanes
      end)
    in
    let take (state, held) (_ : int) =
      let state, drawn = B.draw_dwell state ~inputs in
      state, drawn :: held
    in
    let (_ : Prng.state), drawn =
      List.fold (List.range 0 dwells) ~init:(Prng.create ~seed:7, []) ~f:take
    in
    printf
      "P %2d G %d Cin %d, %d dwells: %s\n"
      rows
      lanes
      inputs
      dwells
      (B.check (List.rev drawn))
  in
  case ~rows:3 ~lanes:1 ~inputs:1 ~dwells:1;
  case ~rows:6 ~lanes:3 ~inputs:2 ~dwells:3;
  case ~rows:8 ~lanes:4 ~inputs:3 ~dwells:2;
  case ~rows:48 ~lanes:4 ~inputs:6 ~dwells:2;
  [%expect
    {|
    P  3 G 1 Cin 1, 1 dwells: ok
    P  6 G 3 Cin 2, 3 dwells: ok
    P  8 G 4 Cin 3, 2 dwells: ok
    P 48 G 4 Cin 6, 2 dwells: ok
    |}]
;;

let%expect_test "the waveform of one dwell and its drain" =
  (* Three rows, one lane, one input channel: NINE CYCLES AND NOTHING ELSE — the window
     stands outside, thus a dwell is exactly its terms. [term_first] opens the sum and
     [term_last] closes it; the accumulator takes a term two cycles behind its strobe,
     thus [drained] stands four cycles behind [term_last] and the rows leave in row order,
     which is the order a column is packed in. *)
  let module B =
    Bench (struct
      let rows = 3
      let lanes = 1
    end)
  in
  let column = [| 64; 128; 192 |] in
  let dwell =
    { B.columns = [| [| column; column; column |] |]
    ; weights = [| Array.init 9 ~f:(fun tap -> [| tap + 1 |]) |]
    }
  in
  let drained, waves = B.run ~trace:true [ dwell ] in
  let waves = Option.value_exn waves ~message:"a traced run gives a waveform" in
  (* the picture is the HANDSHAKE and the text below it is the arithmetic: a 32-bit
     accumulator does not fit a wave column, and the rows print exactly *)
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.Bit
          [ "clock"; "term"; "term_first"; "term_last"; "drained" ]
      ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.Unsigned_int
          [ "row" ]
      ]
    ~show_digest:false
    ~wave_width:0
    ~display_width:74
    waves;
  let show (at, sums) = sprintf "row %d = %d" at sums.(0) in
  printf "%s\n" (String.concat ~sep:", " (List.map drained ~f:show));
  printf
    "the twin says %s\n"
    (String.concat
       ~sep:", "
       (List.mapi
          (Array.to_list (B.expected dwell))
          ~f:(fun at sums -> sprintf "row %d = %d" at sums.(0))));
  [%expect
    {|
    ┌Signals─────────┐┌Waves─────────────────────────────────────────────────┐
    │clock           ││┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐│
    │                ││ └┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└│
    │term            ││──────────────────┐                                   │
    │                ││                  └─────────────                      │
    │term_first      ││──┐                                                   │
    │                ││  └─────────────────────────────                      │
    │term_last       ││                ┌─┐                                   │
    │                ││────────────────┘ └─────────────                      │
    │drained         ││                        ┌─────┐                       │
    │                ││────────────────────────┘     └─                      │
    │                ││──────────────────────────┬─┬───                      │
    │row             ││ 0                        │1│2                        │
    │                ││──────────────────────────┴─┴───                      │
    └────────────────┘└──────────────────────────────────────────────────────┘
    row 0 = 3264, row 1 = 6144, row 2 = 4416
    the twin says row 0 = 3264, row 1 = 6144, row 2 = 4416
    |}]
;;

let%expect_test "a dwell shorter than the chain throws rows away" =
  (* THE RULE THE ELABORATION ENFORCES, AND WHAT IT BUYS. A capture reloads every stage of
     the chain, thus a dwell that closes before the drain of the one before it has emptied
     loses the rows still inside. [Elaboration.create] refuses such a layer — a dwell of
     [9 * inputs] cycles against a chain of [rows] stages — and this is the fault it
     refuses: sixteen rows drained by dwells that close twelve cycles apart. *)
  let module B =
    Bench (struct
      let rows = 16
      let lanes = 1
    end)
  in
  let take (state, held) (_ : int) =
    let state, drawn = B.draw_dwell state ~inputs:1 in
    state, drawn :: held
  in
  let (_ : Prng.state), drawn =
    List.fold (List.range 0 2) ~init:(Prng.create ~seed:7, []) ~f:take
  in
  printf "two dwells of one input channel at P 16: %s\n" (B.check drawn);
  [%expect {| two dwells of one input channel at P 16: order, count, sums |}]
;;

let%expect_test "the array agrees over a sweep of shapes" =
  (* THE FUZZ. The named cases above hold the shapes that matter; this holds the ones
     nobody thought of — a shape, its channel count and its data all drawn, over and over.
     A drawn shape takes the input channels the drain rule demands, thus the sweep states
     legal work and never the fault the test below states. One seed reproduces every case
     of it. *)
  let case state =
    let state, rows = Prng.For_test.draw_between state ~low:2 ~high:14 in
    let state, lanes = Prng.For_test.draw_between state ~low:1 ~high:5 in
    let state, drawn = Prng.For_test.draw_between state ~low:1 ~high:6 in
    (* the drain rule: a dwell of [9 * inputs] cycles against a chain of [rows] stages *)
    let inputs = Int.max drawn ((rows + 8) / 9) in
    let state, dwells = Prng.For_test.draw_between state ~low:1 ~high:3 in
    let module B =
      Bench (struct
        let rows = rows
        let lanes = lanes
      end)
    in
    let take (state, held) (_ : int) =
      let state, drawn = B.draw_dwell state ~inputs in
      state, drawn :: held
    in
    let state, work = List.fold (List.range 0 dwells) ~init:(state, []) ~f:take in
    state, (rows * lanes, dwells, B.check work)
  in
  let step (state, lanes, dwells, complaints) (_ : int) =
    let state, (drawn_lanes, drawn_dwells, verdict) = case state in
    ( state
    , lanes + drawn_lanes
    , dwells + drawn_dwells
    , if String.equal verdict "ok" then complaints else verdict :: complaints )
  in
  let cases = 40 in
  let (_ : Prng.state), lanes, dwells, complaints =
    List.fold (List.range 0 cases) ~init:(Prng.create ~seed:11, 0, 0, []) ~f:step
  in
  printf
    "%d shapes, %d dwells, %d lanes in all: %s\n"
    cases
    dwells
    lanes
    (if List.is_empty complaints
     then "none disagree"
     else String.concat ~sep:"; " complaints);
  [%expect {| 40 shapes, 91 dwells, 1042 lanes in all: none disagree |}]
;;

let%expect_test "the accumulator holds the twin's widest layer at its rails" =
  (* THE INT32 ACCUMULATOR IS THE TIGHTEST NUMBER IN THIS UNIT, and a drawn value never
     reaches it. The twin is exact below [Quantized.Model.widest_inputs] channels and the
     array accumulates in 32 bits on that promise; here the promise runs to its edge —
     every activation at the int16 rail, every weight at the quantizer's rail of 127, and
     every sign the same, at the widest layer the twin allows. The printed peak is the
     margin, and it is small: a sum one channel wider would pass the ceiling. *)
  let module B =
    Bench (struct
      let rows = Diffusion.rows
      let lanes = 4
    end)
  in
  let inputs = Quantized.Model.widest_inputs in
  let rail activation weight =
    { B.columns =
        Array.init inputs ~f:(fun (_ : int) ->
          Array.init 3 ~f:(fun (_ : int) -> Array.create ~len:B.rows activation))
    ; weights =
        Array.init inputs ~f:(fun (_ : int) ->
          Array.init 9 ~f:(fun (_ : int) -> Array.create ~len:B.lanes weight))
    }
  in
  let dwells = [ rail 32767 127; rail 32767 (-127); rail (-32768) 127 ] in
  let hottest =
    List.fold dwells ~init:0 ~f:(fun peak dwell ->
      Array.fold (B.expected dwell) ~init:peak ~f:(fun peak row ->
        Array.fold row ~init:peak ~f:(fun peak sum -> Int.max peak (Int.abs sum))))
  in
  let ceiling = (1 lsl 31) - 1 in
  printf "%d input channels at the rails: %s\n" inputs (B.check dwells);
  printf
    "the hottest sum is %d against the int32 ceiling %d, a margin of %.2f percent\n"
    hottest
    ceiling
    (100.0 *. (1.0 -. (Float.of_int hottest /. Float.of_int ceiling)));
  [%expect
    {|
    57 input channels at the rails: ok
    the hottest sum is 2134867968 against the int32 ceiling 2147483647, a margin of 0.59 percent
    |}]
;;
