(* The exp2 lookup — see exp2.mli for the contract. The table read registers, thus [nn]
   must stand for two cycles and [e] holds on the second. *)

open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; nn : 'a [@bits 22]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { e : 'a [@bits 16] } [@@deriving hardcaml]
end

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock () in
  let data =
    reg
      spec
      (rom
         ~read_addresses:[| select i.nn ~high:11 ~low:4 |]
         Quantized.Constants.exp2_bits).(0)
  in
  let big = select i.nn ~high:21 ~low:16 <>:. 0 in
  let shifted =
    mux (select i.nn ~high:15 ~low:12) (List.init 16 ~f:(fun k -> srl data ~by:k))
  in
  { O.e = mux2 big (zero 16) shifted }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

(* One simulator, and one reading on it. The table read stands in a register, thus [nn]
   holds for two cycles and [e] is whole on the second — the contract's own timing. *)
let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  fun nn ->
    inp.nn := Bits.of_unsigned_int ~width:22 nn;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim;
    Bits.to_int_trunc !(out.e)
;;

(* the unit takes the magnitude of the power, thus the reference's rule reads [-nn] *)
let oracle nn = Quantized.Engine.For_test.exp2_q (-nn)

let%expect_test "the exp2 unit is the table and the shift" =
  let e = harness () in
  List.iter [ 0; 2048; 4096; 8192; 70000 ] ~f:(fun nn ->
    Stdio.printf "%d -> %d (oracle %d)\n" nn (e nn) (oracle nn));
  [%expect
    {|
    0 -> 32768 (oracle 32768)
    2048 -> 23170 (oracle 23170)
    4096 -> 16384 (oracle 16384)
    8192 -> 8192 (oracle 8192)
    70000 -> 0 (oracle 0)
    |}]
;;

let%expect_test "the gate: every reading the unit can make, against the reference" =
  let e = harness () in
  (* Bits 11 to 4 read the table and bits 15 to 12 shift the entry: 256 entries under 16
     shifts is 4096 readings, and that is EVERY distinct reading below the zero floor.
     This gate is therefore exhaustive and not a sample — a fuzz would only rediscover a
     subset of it. *)
  let readings = List.map (List.range 0 4096) ~f:(fun k -> k * 16) in
  (* bits 21 to 16 force 0: an entry shifted 16 places or more is 0, thus every magnitude
     of 16 and above reads 0 *)
  let above_the_floor = List.map (List.range 0 64) ~f:(fun k -> 65536 + (k * 977)) in
  let disagrees nn = e nn <> oracle nn in
  (* bits 3 to 0 fall away, thus a value and the same value with its low bits set are one
     reading. This truncation is a rule of the circuit and the reference holds it too. *)
  let low_bits_move nn = e nn <> e (nn lor 15) in
  Stdio.printf
    "%d readings, every table entry under every shift: %d disagree with the reference\n"
    (List.length readings)
    (List.length (List.filter readings ~f:disagrees));
  Stdio.printf
    "the low four bits fall away: %d readings move when they are set\n"
    (List.length (List.filter readings ~f:low_bits_move));
  Stdio.printf
    "the zero floor at a magnitude of 16: %d of %d readings above it are not 0\n"
    (List.length (List.filter above_the_floor ~f:(fun nn -> e nn <> 0)))
    (List.length above_the_floor);
  Stdio.printf
    "and above the floor the reference agrees: %d disagree\n"
    (List.length (List.filter above_the_floor ~f:disagrees));
  [%expect
    {|
    4096 readings, every table entry under every shift: 0 disagree with the reference
    the low four bits fall away: 0 readings move when they are set
    the zero floor at a magnitude of 16: 0 of 64 readings above it are not 0
    and above the floor the reference agrees: 0 disagree
    |}]
;;

let%expect_test "the waveform of the read: [nn] stands two cycles and [e] lands on the \
                 second"
  =
  (* The one timing rule of this unit. The table read stands in a register, thus the entry
     is a cycle behind [nn] while the shift is not: the cycle after a change carries the
     OLD entry under the NEW shift, and [e] is whole only on the second cycle of a held
     [nn]. That stale cycle is why the contract asks the caller to hold.

     The three values are chosen to show it. Each stands two cycles, and no two neighbours
     share a table entry — a pair that shared one would hide the fault, because the stale
     entry would give the right answer by accident:

     - 0 is entry 0 under no shift, thus 1.0 as 32768
     - 2048 is entry 128 under no shift, thus 1/sqrt 2 as 23170
     - 4096 is entry 0 under one shift, thus 16384

     Read the stale cycles in the picture: [e] holds 32768 into the first cycle of 2048,
     and reads 11585 — entry 128 shifted once, which belongs to neither — into the first
     cycle of 4096. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  List.iter [ 0; 2048; 4096 ] ~f:(fun nn ->
    inp.nn := Bits.of_unsigned_int ~width:22 nn;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim);
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Unsigned_int)
          [ "clock"; "nn"; "e" ]
      ]
    ~show_digest:false
    ~wave_width:2
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──│
    │               ││   └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  │
    │               ││────────────┬───────────┬───────────               │
    │nn             ││ 0          │2048       │4096                      │
    │               ││────────────┴───────────┴───────────               │
    │               ││──────┬───────────┬─────┬─────┬─────               │
    │e              ││ 0    │32768      │23170│11585│16384               │
    │               ││──────┴───────────┴─────┴─────┴─────               │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
