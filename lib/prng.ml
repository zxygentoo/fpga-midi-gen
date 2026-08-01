open Base
open Hardcaml
open Signal

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

(* the three layers of xorshift32; the shifts drop the bits that the OCaml reference masks
   away *)
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

let%expect_test "the circuit walks with the reference" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
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
      let reference, byte = Pink.Prng.next reference in
      let value = Bits.to_int_trunc !(out.value) in
      if n > 996 then Stdio.printf "%08x %02x\n" value (value land 0xff);
      walk
        reference
        (n - 1)
        (agree && value = Pink.Prng.state reference && byte = value land 0xff))
  in
  let agree = walk (Pink.Prng.create ~seed:1) 1000 true in
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

let%expect_test "the waveform of one step" =
  (* The state holds 1 from power-on, takes the seed at [load], advances one time at each
     [step], and holds between the strobes. The last event gives [load] and [step] in the
     same cycle: [load] wins, and the state is the seed again. *)
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
