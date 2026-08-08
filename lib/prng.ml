open Base
open Hardcaml
open Signal

(* the software side: the same recurrence in OCaml integers. The mask drops the bits that
   the circuit's shifts drop. *)
type t = int

let mask = 0xFFFF_FFFF

let create ~seed =
  if seed = 0 || seed land mask <> seed
  then invalid_arg "Prng: the seed must fit 32 bits and must not be 0";
  seed
;;

(* A seed from a flag, or from a stream that makes seeds, holds neither rule of [create]:
   it can be 0, and it can be wider than the state. The fold squeezes the whole integer
   into 32 bits, and 0 — no state of the walk — takes the top state instead. A seed
   already inside the range names itself, thus 7 here is the walk of the board's seed 7. *)
let fold_seed seed =
  let folded = seed lxor (seed lsr 32) land mask in
  create ~seed:(if folded = 0 then mask else folded)
;;

let next t =
  let t = t lxor (t lsl 13) land mask in
  let t = t lxor (t lsr 17) in
  let t = t lxor (t lsl 5) land mask in
  t, t land 0xff
;;

(* One step gives eight bits, and three steps make one uniform. The grid of 2 ** -24 keeps
   the tail of a Box-Muller draw, which a single byte would cut at 3.3 sigma. *)
let uniform t =
  let t, high = next t in
  let t, middle = next t in
  let t, low = next t in
  t, Float.of_int ((((high * 256) + middle) * 256) + low) *. 0x1p-24
;;

(* [uniforms t ~count] is [count] draws in the order of the walk. The walk states that
   order: [init] leaves the order of its elements free, thus it cannot carry a state. *)
let uniforms t ~count =
  let rec walk t n draws =
    if n = 0
    then t, draws
    else (
      let t, draw = uniform t in
      walk t (n - 1) (draw :: draws))
  in
  let t, draws = walk t count [] in
  t, Array.of_list_rev draws
;;

let%expect_test "the seed folds, and the uniforms fill the range" =
  Stdio.printf
    "7 -> %x   0 -> %x   wide -> %x\n"
    (fold_seed 7)
    (fold_seed 0)
    (fold_seed 0x3FFF_FFFF_FFFF_FFFF);
  let count = 100_000 in
  let (_ : t), draws = uniforms (fold_seed 1) ~count in
  let outside = Array.count draws ~f:(fun u -> Float.( < ) u 0.0 || Float.( >= ) u 1.0) in
  let mean = Array.fold draws ~init:0.0 ~f:( +. ) /. Float.of_int count in
  Stdio.printf "outside [0, 1): %d   mean %.4f\n" outside mean;
  [%expect
    {|
    7 -> 7   0 -> ffffffff   wide -> c0000000
    outside [0, 1): 0   mean 0.4997
    |}]
;;

module Rtl = struct
  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; load : 'a
      ; seed : 'a [@bits 32]
      ; step : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { value : 'a [@bits 32] } [@@deriving hardcaml]
  end

  (* the three layers of xorshift32; the shifts drop the bits that the OCaml reference
     masks away *)
  let advance state =
    let state = state ^: sll state ~by:13 in
    let state = state ^: srl state ~by:17 in
    state ^: sll state ~by:5
  ;;

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let state = wire 32 in
    let next = mux2 i.load i.seed (mux2 i.step (advance state) state) in
    assign
      state
      (reg spec ~initialize_to:(Bits.of_unsigned_int ~width:32 1) ~clear_to:(one 32) next);
    { O.value = state }
  ;;

  let%expect_test "the waveform of one step" =
    (* The state holds 1 from power-on, takes the seed at [load], advances one time at
       each [step], and holds between the strobes. The last event gives [load] and [step]
       in the same cycle: [load] wins, and the state is the seed again. *)
    let module Sim = Cyclesim.With_interface (I) (O) in
    let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
    let waves, sim = Cyclesim.Waveform.create sim in
    let inp = Cyclesim.inputs sim in
    inp.seed := Bits.of_unsigned_int ~width:32 42;
    let pulse field =
      field := Bits.vdd;
      Cyclesim.cycle sim;
      field := Bits.gnd;
      Cyclesim.cycle sim
    in
    Cyclesim.cycle sim;
    pulse inp.load;
    pulse inp.step;
    inp.load := Bits.vdd;
    inp.step := Bits.vdd;
    Cyclesim.cycle sim;
    inp.load := Bits.gnd;
    inp.step := Bits.gnd;
    Cyclesim.cycle ~n:3 sim;
    let rules =
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Hex)
          [ "load"; "step"; "seed"; "value" ]
      ]
    in
    Hardcaml_waveterm.Waveform.expect
      ~display_rules:rules
      ~show_digest:false
      ~wave_width:2
      waves;
    [%expect
      {|
      ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
      │load           ││      ┌─────┐                 ┌─────┐              │
      │               ││──────┘     └─────────────────┘     └──────────────│
      │step           ││                  ┌─────┐     ┌─────┐              │
      │               ││──────────────────┘     └─────┘     └──────────────│
      │               ││───────────────────────────────────────────────────│
      │seed           ││ 0000002A                                          │
      │               ││───────────────────────────────────────────────────│
      │               ││────────────┬───────────┬───────────┬──────────────│
      │value          ││ 00000001   │0000002A   │00AD4528   │0000002A      │
      │               ││────────────┴───────────┴───────────┴──────────────│
      └───────────────┘└───────────────────────────────────────────────────┘
      |}]
  ;;
end

let%expect_test "the circuit walks with the software" =
  let module Sim = Cyclesim.With_interface (Rtl.I) (Rtl.O) in
  let sim = Sim.create Rtl.create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.load := Bits.vdd;
  inp.seed := Bits.of_unsigned_int ~width:32 1;
  Cyclesim.cycle sim;
  inp.load := Bits.gnd;
  inp.step := Bits.vdd;
  let rec walk reference n agree =
    if n = 0
    then agree
    else (
      Cyclesim.cycle sim;
      let reference, byte = next reference in
      let value = Bits.to_int_trunc !(out.value) in
      if n > 996 then Stdio.printf "%08x %02x\n" value (value land 0xff);
      walk reference (n - 1) (agree && value = reference && byte = value land 0xff))
  in
  let agree = walk (create ~seed:1) 1000 true in
  Stdio.printf "1000 steps agree: %b\n" agree;
  [%expect
    {|
    00042021 21
    04080601 01
    9dcca8c5 c5
    1255994f 4f
    1000 steps agree: true
    |}]
;;
