(* The softplus correction lookup — see softplus.mli for the contract. The table read
   registers, thus [v] must stand for two cycles and [c] holds on the second. The unit
   gives the CORRECTION alone; the caller adds the ramp, which is exact and costs a mux. *)

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
  type 'a t = { c : 'a [@bits 16] (** ln(1 + exp(-|v|)), Q12 *) } [@@deriving hardcaml]
end

(* the magnitude of a signed Q12 value, 16 bits: -32768 negates to itself, which is the
   one value the row clamp below catches *)
let magnitude v = mux2 (msb v) (negate v) v

(* The row of a magnitude: bits 14 down to 7, and the clamp for the single magnitude
   32768, whose bit 15 stands alone. *)
let row v =
  let m = magnitude v in
  mux2 (msb m) (ones 8) (select m ~high:14 ~low:7)
;;

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock () in
  { O.c =
      reg spec (rom ~read_addresses:[| row i.v |] Quantized.Constants.softplus_bits).(0)
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
    Bits.to_int_trunc !(out.c)
;;

let%expect_test "the gate: EVERY reading the unit can make, against the reference" =
  let c = harness () in
  (* The row is the magnitude's bits 14 to 7, thus 256 rows cover every input and the low
     seven bits change nothing. This walks a positive and a negative input in each row —
     the two must read the same row, because the table takes a magnitude — and the low
     bits of each. The whole domain is covered; a fuzz would rediscover a subset. *)
  let rows = List.range 0 256 in
  let at row = row * 128 in
  (* The oracle is the table and the row rule, both of them [Quantized]'s. Taking the ramp
     off the whole softplus would NOT do: that function clamps its sum to the format, thus
     at the top of the range it would ask this unit to answer 0 for a correction that is
     really 1. The clamp belongs to the caller that adds the ramp, and it is tested where
     it lives. *)
  let want v =
    Hardcaml.Bits.to_int_trunc
      Quantized.Constants.softplus_bits.(Quantized.Constants.softplus_index v)
  in
  let disagrees v = c v <> want v in
  Stdio.printf
    "%d rows: %d disagree positive, %d disagree negative, %d disagree with the low bits \
     set\n"
    (List.length rows)
    (List.count rows ~f:(fun row -> disagrees (at row)))
    (List.count rows ~f:(fun row -> disagrees (-at row)))
    (List.count rows ~f:(fun row -> disagrees (at row + 127)));
  (* a magnitude and its negative read one row: the table takes |v| *)
  Stdio.printf
    "the sign falls away: %d rows read differently at -v\n"
    (List.count rows ~f:(fun row -> c (at row) <> c (-at row)));
  (* the two ends, and the one input whose magnitude does not fit the rows *)
  List.iter [ -32768; -4096; 0; 4096; 32767 ] ~f:(fun v ->
    Stdio.printf "  %6d -> %5d (reference %d)\n" v (c v) (want v));
  [%expect
    {|
    256 rows: 0 disagree positive, 0 disagree negative, 0 disagree with the low bits set
    the sign falls away: 0 rows read differently at -v
      -32768 ->     1 (reference 1)
       -4096 ->  1266 (reference 1266)
           0 ->  2807 (reference 2807)
        4096 ->  1266 (reference 1266)
       32767 ->     1 (reference 1)
    |}]
;;

let%expect_test "the waveform of the read: [v] stands two cycles and [c] lands on the \
                 second"
  =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  List.iter [ 0; 4096; -16384 ] ~f:(fun v ->
    inp.v := Bits.of_signed_int ~width:16 v;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim);
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Unsigned_int)
          [ "clock"; "v"; "c" ]
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
    │v              ││ 0          │4096       │49152                     │
    │               ││────────────┴───────────┴───────────               │
    │               ││──────┬───────────┬───────────┬─────               │
    │c              ││ 0    │2807       │1266       │73                  │
    │               ││──────┴───────────┴───────────┴─────               │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
