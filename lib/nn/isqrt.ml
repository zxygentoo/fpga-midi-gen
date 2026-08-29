(* The restoring square root — see isqrt.mli for the contract. One radicand bit pair a
   cycle, from the top. *)

open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a
    ; value : 'a [@bits 42]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { root : 'a [@bits 21]
    ; busy : 'a
    }
  [@@deriving hardcaml]
end

(* one cycle for each bit pair of the radicand, thus the width of the root *)
let busy_cycles = 21

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let m = Variable.reg spec ~width:42 in
  let root = Variable.reg spec ~width:21 in
  let r = Variable.reg spec ~width:25 in
  let n = Variable.reg spec ~width:(Int.ceil_log2 (busy_cycles + 1)) in
  let busy = Variable.reg spec ~width:1 in
  compile
    [ if_
        i.start
        [ m <-- i.value; root <--. 0; r <--. 0; n <--. busy_cycles; busy <-- vdd ]
        [ when_
            busy.value
            [ (let r' =
                 sel_bottom (r.value @: select m.value ~high:41 ~low:40) ~width:25
               in
               let trial = uresize (root.value @: of_unsigned_int ~width:2 1) ~width:25 in
               let fits = r' >=: trial in
               proc
                 [ m <-- sll m.value ~by:2
                 ; n <-- n.value -:. 1
                 ; r <-- mux2 fits (r' -: trial) r'
                 ; root <-- sel_bottom (root.value @: fits) ~width:21
                 ; when_ (n.value ==:. 1) [ busy <-- gnd ]
                 ])
            ]
        ]
    ];
  { O.root = root.value; busy = busy.value }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

(* One simulator, and one walk on it: [start], then wait on [busy] and read the root in
   the cycle the wait releases — the contract's own reading order. *)
let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  fun v ->
    inp.value := Bits.of_unsigned_int ~width:42 v;
    inp.start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.start := Bits.gnd;
    while Bits.to_bool !(out.busy) do
      Cyclesim.cycle sim
    done;
    Bits.to_int_trunc !(out.root)
;;

let%expect_test "the isqrt floors, as the reference does" =
  let isqrt = harness () in
  let oracle = Quantized.For_test.isqrt in
  List.iter
    [ 0; 15; 16; 4295; (1 lsl 41) + 12345 ]
    ~f:(fun v -> Stdio.printf "%d -> %d (oracle %d)\n" v (isqrt v) (oracle v));
  [%expect
    {|
    0 -> 0 (oracle 0)
    15 -> 3 (oracle 3)
    16 -> 4 (oracle 4)
    4295 -> 65 (oracle 65)
    2199023267897 -> 1482910 (oracle 1482910)
    |}]
;;

(* the radicand is 42 bits, thus the root is 21 and the top radicand squares just inside
   the root's width: (2^21 - 1)^2 is under 2^42 - 1, and 2^21 squared is over it *)
let value_high = (1 lsl 42) - 1
let root_high = (1 lsl 21) - 1

let%expect_test "the fuzz: the root is the floor, and it squares back under the value" =
  let isqrt = harness () in
  (* The seed is fixed, thus the gate is the same gate on every machine, and the report is
     a verdict and never the drawn values. *)
  let state = Random.State.make [| 20260819 |] in
  let drawn (_ : int) = Random.State.int_incl state 0 value_high in
  (* every perfect square of a power of two, and its two neighbours: the walk turns on the
     compare against the trial subtrahend, thus a square and the value under it are the
     two sides of that decision *)
  let squares =
    List.concat_map (List.range 0 21) ~f:(fun k ->
      let s = 1 lsl k in
      [ (s * s) - 1; s * s; (s * s) + 1 ])
  in
  let edges =
    [ 0; 1; 2; 3; value_high; value_high - 1; root_high * root_high ] @ squares
  in
  let cases = edges @ List.map (List.range 0 1000) ~f:drawn in
  (* the floor states itself: the root squares back under the value, and one more root
     passes it. This holds with no oracle at all, thus the two checks are independent. *)
  let fault v =
    let root = isqrt v in
    let floors = root * root <= v && (root + 1) * (root + 1) > v in
    if floors && root = Quantized.For_test.isqrt v then None else Some (v, root)
  in
  (match List.filter_map cases ~f:fault with
   | [] ->
     Stdio.printf
       "%d walks: every root floors its value and agrees with the reference\n"
       (List.length cases)
   | (v, root) :: (_ : (int * int) list) ->
     Stdio.printf
       "the root of %d read %d, the reference gives %d\n"
       v
       root
       (Quantized.For_test.isqrt v));
  [%expect {| 1070 walks: every root floors its value and agrees with the reference |}]
;;

let%expect_test "the waveform of one walk: a bit pair a cycle, and the root growing" =
  (* The root of 4295 is 65, and the picture is the whole contract: [start] is one cycle,
     [busy] rises after it and stands [busy_cycles], and the root grows a bit at a time
     from the top — thus it is meaningless until the wait releases, and whole in the cycle
     it does. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.value := Bits.of_unsigned_int ~width:42 4295;
  inp.start := Bits.vdd;
  Cyclesim.cycle sim;
  inp.start := Bits.gnd;
  let cycles_waited = ref 0 in
  while Bits.to_bool !(out.busy) do
    Cyclesim.cycle sim;
    Int.incr cycles_waited
  done;
  let landed = Bits.to_int_trunc !(out.root) in
  Cyclesim.cycle sim;
  Cyclesim.cycle sim;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Unsigned_int)
          [ "clock"; "start"; "busy"; "root" ]
      ]
    ~show_digest:false
    ~wave_width:(-1)
    waves;
  Stdio.printf
    "the wait released after %d cycles and the root read %d\n"
    !cycles_waited
    landed;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥│
    │               ││╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨│
    │start          ││─┐                                                 │
    │               ││ └──────────────────────                           │
    │busy           ││ ┌────────────────────┐                            │
    │               ││─┘                    └─                           │
    │               ││────────────────┬┬┬┬┬┬┬─                           │
    │root           ││ 0              │││││││.                           │
    │               ││────────────────┴┴┴┴┴┴┴─                           │
    └───────────────┘└───────────────────────────────────────────────────┘
    the wait released after 21 cycles and the root read 65
    |}]
;;
