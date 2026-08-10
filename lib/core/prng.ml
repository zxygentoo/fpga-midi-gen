open Base
open Hardcaml
open Signal

(* A draw carries the state from one draw to the next, thus a draw is a function of the
   state and every function below is one. *)
type 'a t = state -> state * 'a
and state = int

(* an OCaml integer holds 63 bits; the mask drops what the circuit's 32-bit shifts drop *)
let mask = 0xFFFF_FFFF

let create ~seed =
  if seed = 0 || seed land mask <> seed
  then invalid_arg "Prng: the seed must fit 32 bits and must not be 0";
  seed
;;

let create_folded ~seed =
  (* the mask comes after the mix, not inside it: a seed wider than the state must reach
     the low bits *)
  let mixed = seed lxor (seed lsr 32) in
  let folded = mixed land mask in
  create ~seed:(if folded = 0 then mask else folded)
;;

let next state =
  let state = state lxor (state lsl 13) land mask in
  let state = state lxor (state lsr 17) in
  let state = state lxor (state lsl 5) land mask in
  state, state land 0xff
;;

let uniform state =
  let state, high = next state in
  let state, middle = next state in
  let state, low = next state in
  state, Float.of_int ((((high * 256) + middle) * 256) + low) *. 0x1p-24
;;

(* A walk, not an [init]: the order of the elements of [init] is free, thus it cannot
   carry a state. The guard is the choke point of [normals] and [bernoullis]: a negative
   count would walk past 0 and never stop. *)
let uniforms ~count state =
  if count < 0 then invalid_arg "Prng: the count of draws is 0 or more";
  let rec walk state n draws =
    if n = 0
    then state, draws
    else (
      let state, draw = uniform state in
      walk state (n - 1) (draw :: draws))
  in
  let state, draws = walk state count [] in
  state, Array.of_list_rev draws
;;

(* the clamp holds the logarithm finite, because the grid of [uniform] holds 0 *)
let box_muller u1 u2 =
  Float.sqrt (-2. *. Float.log (Float.max 1e-12 u1)) *. Float.cos (2. *. Float.pi *. u2)
;;

(* the uniforms come out first, thus each normal is a function of its index alone and the
   free order of [init] cannot reach the walk *)
let normals ~count ~scale state =
  let state, draws = uniforms ~count:(2 * count) state in
  let normal i = scale *. box_muller draws.(2 * i) draws.((2 * i) + 1) in
  state, Array.init count ~f:normal
;;

let bernoullis ~count ~probability state =
  let state, draws = uniforms ~count state in
  let hit u = if Float.(u < probability) then 1.0 else 0.0 in
  state, Array.map draws ~f:hit
;;

let return value state = state, value

let bind draw ~f state =
  let state, value = draw state in
  (f value) state
;;

let map draw ~f state =
  let state, value = draw state in
  state, f value
;;

let ( let* ) draw f = bind draw ~f
let ( let+ ) draw f = map draw ~f
let all draws state = List.fold_map draws ~init:state ~f:(fun state draw -> draw state)

(* an independent walk, drawn from this one: four steps make its 32 bits. A part of a
   computation takes one of these and then draws on its own, thus it holds no place in the
   order of the parent. *)
let split =
  let* high = next in
  let* second = next in
  let* third = next in
  let+ low = next in
  create_folded ~seed:((((((high * 256) + second) * 256) + third) * 256) + low)
;;

let run draw state = draw state

let%expect_test "the seed folds, and the uniforms fill the range" =
  Stdio.printf
    "7 -> %x   0 -> %x   wide -> %x\n"
    (create_folded ~seed:7)
    (create_folded ~seed:0)
    (create_folded ~seed:0x3FFF_FFFF_FFFF_FFFF);
  let count = 100_000 in
  let (_ : state), draws = uniforms ~count (create_folded ~seed:1) in
  let outside = Array.count draws ~f:(fun u -> Float.(u < 0.0 || u >= 1.0)) in
  let mean = Array.fold draws ~init:0.0 ~f:( +. ) /. Float.of_int count in
  Stdio.printf "outside [0, 1): %d   mean %.4f\n" outside mean;
  (* through [normals], because the guard sits below it and [uniforms] is not exported *)
  (match normals ~count:(-1) ~scale:1.0 (create_folded ~seed:1) with
   | (_ : state * float array) -> ()
   | exception Invalid_argument message -> Stdio.print_endline message);
  [%expect
    {|
    7 -> 7   0 -> ffffffff   wide -> c0000000
    outside [0, 1): 0   mean 0.4997
    Prng: the count of draws is 0 or more
    |}]
;;

let%expect_test "the normals take their scale" =
  let count = 100_000 in
  let (_ : state), draws = normals ~count ~scale:0.02 (create_folded ~seed:1) in
  let total = Array.fold draws ~init:0.0 ~f:( +. ) in
  let mean = total /. Float.of_int count in
  let square acc draw =
    let delta = draw -. mean in
    acc +. (delta *. delta)
  in
  let deviation =
    Float.sqrt (Array.fold draws ~init:0.0 ~f:square /. Float.of_int count)
  in
  Stdio.printf "mean %.4f  deviation %.4f\n" mean deviation;
  [%expect {| mean -0.0000  deviation 0.0201 |}]
;;

let%expect_test "the walk draws what the plain calls draw" =
  (* the same three draws, threaded by hand and threaded by [let*] *)
  let by_hand state =
    let state, a = uniform state in
    let state, b = normals ~count:2 ~scale:0.5 state in
    let state, c = next state in
    state, (a, b, c)
  in
  let by_walk =
    let* a = uniform in
    let* b = normals ~count:2 ~scale:0.5 in
    let+ c = next in
    a, b, c
  in
  let hand_state, (a, b, c) = by_hand (create_folded ~seed:9) in
  let walk_state, (x, y, z) = run by_walk (create_folded ~seed:9) in
  Stdio.printf
    "same state %b   same draws %b\n"
    (hand_state = walk_state)
    (Float.equal a x && Array.equal Float.equal b y && c = z);
  (* [all] must run left to right, or the order of the walk is not the order of the list *)
  let (_ : state), ordered = run (all [ next; next; next ]) (create_folded ~seed:9) in
  let plain =
    let state = create_folded ~seed:9 in
    let state, p = next state in
    let state, q = next state in
    let (_ : state), r = next state in
    [ p; q; r ]
  in
  Stdio.printf "all in order %b\n" (List.equal Int.equal ordered plain);
  [%expect {|
    same state true   same draws true
    all in order true
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

  (* the shifts drop the bits that the OCaml reference masks away *)
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
