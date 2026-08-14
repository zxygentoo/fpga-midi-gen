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

let%expect_test "the exp2 unit is the table and the shift" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let e nn =
    inp.nn := Bits.of_unsigned_int ~width:22 nn;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim;
    Bits.to_int_trunc !(out.e)
  in
  (* the unit takes the negated exponent, thus the reference's rule reads [-nn] *)
  let oracle nn = Quantized.Engine.For_test.exp2_q (-nn) in
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
