(* The column array — see column_array.mli, and docs/diffusion_rtl.md for the design.

   THE THREE-COLUMN WINDOW STANDS OUTSIDE THIS UNIT, with the store it caches: holding the
   registers here and the policy there would put a load strobe on one side of an interface
   and the dwell it must be timed against on the other. The path is the same either way,
   thus the cut costs no logic and no stage. *)

open Core
open Hardcaml
open Signal
module Nn_quantized = Mgen_nn.Quantized
module Placement = Mgen_nn.Placement

module type Shape = sig
  val rows : int
  val lanes : int
end

(* the operand registers, then the product register. The DSP48E1 packs as A, B, M and P,
   thus the depth is the primitive's own and not a choice. *)
let accumulate_latency = 2

(* the accumulator settles one cycle behind its last term, the chain takes it on that
   cycle's edge, and the bottom stage stands one cycle later *)
let first_row_latency = accumulate_latency + 2
let activation_bits = Model.activation_bits
let accumulator_bits = Model.accumulator_bits
let weight_bits = 8

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
    (* the datapath takes no clear: a stale value no strobe reaches touches nothing, and
       the DSP packs best with no reset *)
    let dspec = Reg_spec.create ~clock:i.clock () in
    let row_of column at =
      select
        column
        ~high:((at * activation_bits) + activation_bits - 1)
        ~low:(at * activation_bits)
    in
    (* the three pitch taps are wire shifts: no arithmetic stands here *)
    let silent = zero activation_bits in
    let activation at =
      let below = if at = 0 then silent else row_of i.column (at - 1) in
      let here = row_of i.column at in
      let above = if at = rows - 1 then silent else row_of i.column (at + 1) in
      (* the fourth arm is the don't care of a 2-bit select *)
      mux i.row_shift [ below; here; above; here ]
    in
    let weight lane =
      select
        i.weights
        ~high:((lane * weight_bits) + weight_bits - 1)
        ~low:(lane * weight_bits)
    in
    (* the two broadcast trees: the operand registers stand once for each row and once for
       each channel, never once for each lane *)
    let operand_a = Array.init rows ~f:(fun at -> reg dspec (activation at)) in
    (* THE WEIGHT REPLICA BANK. With a plain register the tools absorb it into each DSP's
       B port, leaving the broadcast net ONE driver of 528 pins with no flop in the fabric
       to replicate: 12 ns of route. The bank IS that operand register, stated in fabric,
       one copy for each slice of rows. The depth of the pipe does not move. *)
    let slices = Placement.slices_for ~rows in
    let operand_b =
      Array.init lanes ~f:(fun lane ->
        Array.init slices ~f:(fun (_ : int) ->
          Placement.replica (reg dspec (weight lane))))
    in
    (* The operand and product registers FREE-RUN and only the sum is gated: that is how
       the DSP is meant to be driven. The flags ride the pipe beside the operands. *)
    let taking = pipeline spec ~n:accumulate_latency i.term in
    let opening = pipeline spec ~n:accumulate_latency i.term_first in
    let accumulator =
      Array.init rows ~f:(fun at ->
        let a = operand_a.(at) in
        Array.init lanes ~f:(fun lane ->
          let b = operand_b.(lane).(at / Placement.slice_rows) in
          let product = reg dspec (a *+ b) in
          let term = sresize product ~width:accumulator_bits in
          reg_fb dspec ~enable:taking ~width:accumulator_bits ~f:(fun sum ->
            mux2 opening term (sum +: term))))
    in
    (* THE CAPTURE STANDS ON THE EDGE THAT CLOSES THE DWELL, and the chain takes the value
       the register held BEFORE that edge — the finished sum. Thus two dwells never need a
       gap between them. *)
    let pre_capture = pipeline spec ~n:accumulate_latency i.term_last in
    let capture = reg spec pre_capture in
    (* THE CAPTURE BANK: from one flop the capture select reached every register of the
       chain at fanout 6 019. One copy for each chain stage instead — 48 drivers of about
       128 pins, laid out beside their own stage. The drain walk keeps [capture] itself. *)
    let capture_bank =
      Array.init rows ~f:(fun (_ : int) -> Placement.replica (reg spec pre_capture))
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
    (* The chain: one stage for each row, [lanes] wide. The capture loads every stage at
       one time and each cycle after it every stage takes the stage above, thus NO
       REGISTER REACHES FARTHER THAN ITS NEIGHBOUR. The rows leave in row order, which is
       the order the epilogue packs a column in. *)
    let stage_of at above =
      let loaded = concat_lsb (Array.to_list accumulator.(at)) in
      let take = capture_bank.(at) in
      reg
        dspec
        ~enable:(take |: draining.value)
        (mux2 take loaded (Option.value above ~default:loaded))
    in
    (* built from the top row down, thus each stage names the stage above it as a value
       and no wire stands in the chain. The top row's mux collapses. *)
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

(* THE REFERENCE IS THE TWIN'S INNER SUM restricted to one (column, group): the array
   holds no more than that, thus the gates need no model of their own. *)
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

  (* the dwells BACK TO BACK — the next opens on the cycle behind the [term_last] of the
     one before, which is the hazard the capture rule stands on *)
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
    (* the bench stands where the walk will and holds the window itself, thus a term names
       its column and the array only shifts it *)
    let multiply ~first ~last ~tap column weights =
      inp.term := Bits.vdd;
      inp.term_first := if first then Bits.vdd else Bits.gnd;
      inp.term_last := if last then Bits.vdd else Bits.gnd;
      inp.column := Harness.pack column ~width:activation_bits;
      Harness.set inp.row_shift (tap % 3);
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
    Harness.verdict
      [ "order", not (List.equal Int.equal (List.map drained ~f:fst) order)
      ; "count", not counted
      ; "sums", not (counted && List.for_all2_exn given wanted ~f:(Array.equal Int.equal))
      ]
  ;;
end

let%expect_test "the array sums what the twin sums" =
  (* The fuzz, over the geometries a simulation can hold and at the elected P 48 by G 4.
     THE DWELLS RUN BACK TO BACK, thus every case also holds the capture rule. *)
  let case ~rows ~lanes ~inputs ~dwells =
    let module B =
      Bench (struct
        let rows = rows
        let lanes = lanes
      end)
    in
    let draw state (_ : int) = B.draw_dwell state ~inputs in
    let (_ : Prng.state), drawn =
      List.fold_map (List.range 0 dwells) ~init:(Prng.create ~seed:7) ~f:draw
    in
    printf "P %2d G %d Cin %d, %d dwells: %s\n" rows lanes inputs dwells (B.check drawn)
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
  (* Three rows, one lane, one input channel: NINE CYCLES AND NOTHING ELSE, because the
     window stands outside. [drained] stands four cycles behind [term_last] and the rows
     leave in row order. *)
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
  (* the picture is the HANDSHAKE and the text below it the arithmetic: a 32-bit
     accumulator does not fit a wave column *)
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
  (* THE FUZZ over shapes nobody thought of: the shape, its channel count and its data all
     drawn. A drawn shape takes the input channels the drain rule demands, thus the sweep
     states legal work and never the fault the test below states. *)
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
  (* THE INT32 ACCUMULATOR AT ITS EDGE, which a drawn value never reaches: every
     activation at the int16 rail, every weight at 127, every sign the same, at the widest
     layer the twin allows. The printed margin is small — one channel wider would pass the
     ceiling. *)
  let module B =
    Bench (struct
      let rows = Model.rows
      let lanes = 4
    end)
  in
  let inputs = Model.widest_inputs in
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
