(* The sigmoid lookup — see sigmoid.mli for the contract. The table read registers, thus
   [v] must stand for two cycles and [s] holds on the second. It is the idiom of [Exp2],
   with a simpler address: the row is the top eight bits of the input with the sign bit
   flipped, thus there is no arithmetic on the address path at all. *)

open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; v : 'a [@bits 16] (** the input, signed Q12 *)
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { s : 'a [@bits 16] (** the sigmoid, Q15 *) } [@@deriving hardcaml]
end

(* The row of a signed Q12 value. Adding 128 to the top eight bits inverts the sign bit
   and leaves the other seven, thus the whole address is one inverter. *)
let row v = ~:(msb v) @: select v ~high:14 ~low:8

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock () in
  { O.s =
      reg spec (rom ~read_addresses:[| row i.v |] Quantized.Constants.sigmoid_bits).(0)
  }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  fun v ->
    inp.v := Bits.of_signed_int ~width:16 v;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim;
    Bits.to_int_trunc !(out.s)
;;

let%expect_test "the gate: EVERY reading the unit can make, against the reference" =
  let s = harness () in
  (* The row is eight bits of a sixteen-bit input, thus 256 rows cover the whole domain
     and the low eight bits change nothing. This walks one input in each row and one with
     every low bit set, which is every DISTINCT reading the unit has and the truncation
     rule under it. A fuzz would rediscover a subset of this. *)
  let rows = List.range 0 256 in
  let at row = (row - 128) * 256 in
  let disagrees v = s v <> Quantized.Engine.For_test.sigmoid_q v in
  Stdio.printf
    "%d rows, %d disagree with the reference at the low end, %d at the high\n"
    (List.length rows)
    (List.count rows ~f:(fun row -> disagrees (at row)))
    (List.count rows ~f:(fun row -> disagrees (at row + 255)));
  Stdio.printf
    "the low eight bits fall away: %d rows move when they are set\n"
    (List.count rows ~f:(fun row -> s (at row) <> s (at row + 255)));
  (* the ends and the middle, as numbers a reader can check by hand *)
  List.iter [ -32768; -4096; 0; 4096; 32767 ] ~f:(fun v ->
    Stdio.printf "  %6d -> %5d\n" v (s v));
  [%expect
    {|
    256 rows, 0 disagree with the reference at the low end, 0 at the high
    the low eight bits fall away: 0 rows move when they are set
      -32768 ->    11
       -4096 ->  9015
           0 -> 16640
        4096 -> 24155
       32767 -> 32757
    |}]
;;

let%expect_test "the waveform of the read: [v] stands two cycles and [s] lands on the \
                 second"
  =
  (* The one timing rule of this unit, and the reason the contract asks the caller to
     hold: the table read stands in a register, thus [s] is the row of the PREVIOUS
     cycle's [v]. Nothing bypasses that register — [Exp2] shifts its entry combinationally
     and can show a value that belongs to neither input, and this unit cannot — thus the
     lag is clean and the picture is [s] one whole cycle behind [v]. The three values
     below sit in three different rows, thus the lag cannot hide behind a shared entry. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  List.iter [ 0; -4096; 4096 ] ~f:(fun v ->
    inp.v := Bits.of_signed_int ~width:16 v;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim);
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Unsigned_int)
          [ "clock"; "v"; "s" ]
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
    │v              ││ 0          │61440      │4096                      │
    │               ││────────────┴───────────┴───────────               │
    │               ││──────┬───────────┬───────────┬─────               │
    │s              ││ 0    │16640      │9015       │24155               │
    │               ││──────┴───────────┴───────────┴─────               │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
