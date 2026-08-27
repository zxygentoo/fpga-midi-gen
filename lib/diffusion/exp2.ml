(* Era six's fork of [Mgen_nn.Exp2] — see exp2.mli for the contract and for the backport
   question. What changes is three registers; what does not is the table and its rules. *)

open Base
open Hardcaml
open Signal
module Nn_quantized = Mgen_nn.Quantized

let latency = 2

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
  (* THE ADDRESS REGISTER STANDS BEFORE THE MEMORY — era four's rule, which the fork first
     broke: ring 3 read the caller's whole magnitude cone on the table's address pins, the
     worst path of the machine. The register takes the magnitude WHOLE, thus the entry,
     the shift and the zero test all derive from one registered value. *)
  let nn = reg spec i.nn in
  let data =
    reg
      spec
      (rom
         ~read_addresses:[| select nn ~high:11 ~low:4 |]
         Nn_quantized.Constants.exp2_bits).(0)
  in
  (* THE SHIFT AND THE ZERO TEST RIDE THE REGISTER THE ENTRY RIDES. In the shared unit
     they read [nn] as it stands, thus the entry of one cycle meets the shift of the next
     and a caller must hold its magnitude. Here all three name one magnitude, and a
     magnitude may change every cycle. *)
  let integer = reg spec (select nn ~high:15 ~low:12) in
  let big = reg spec (select nn ~high:21 ~low:16 <>:. 0) in
  let shifted = mux integer (List.init 16 ~f:(fun k -> srl data ~by:k)) in
  { O.e = mux2 big (zero 16) shifted }
;;

let%expect_test "the fork answers what the shared unit answers" =
  (* THE EVIDENCE FOR THE BACKPORT. Driven the shared unit's way — a magnitude held for
     two cycles — the fork states the same weight by the held read, and both state the
     software rule; the fork stands one cycle deeper, the address register of ring 3's
     round. That is what makes the change safe for era four and era five, and it is
     measured here rather than argued. *)
  let module Shared = Cyclesim.With_interface (Mgen_nn.Exp2.I) (Mgen_nn.Exp2.O) in
  let module Fork = Cyclesim.With_interface (I) (O) in
  let magnitudes = [ 0; 1; 300; 1000; 4095; 4096; 5000; 20000; 65535; 65536; 100000 ] in
  let held sim inp out =
    List.map magnitudes ~f:(fun nn ->
      inp := Bits.of_unsigned_int ~width:22 nn;
      Cyclesim.cycle sim;
      Cyclesim.cycle sim;
      Bits.to_unsigned_int !out)
  in
  let shared = Shared.create Mgen_nn.Exp2.create in
  let fork = Fork.create create in
  let shared_held = held shared (Cyclesim.inputs shared).nn (Cyclesim.outputs shared).e in
  let fork_held = held fork (Cyclesim.inputs fork).nn (Cyclesim.outputs fork).e in
  let wanted = List.map magnitudes ~f:Nn_quantized.exp2_of_magnitude in
  Stdio.printf
    "held two cycles: the shared unit and the fork agree %b, and both state the rule %b\n"
    (List.equal Int.equal shared_held fork_held)
    (List.equal Int.equal fork_held wanted);
  (* THE ONE-A-CYCLE CLAIM IS NOT HERE, and deliberately. A simulator's read after a clock
     edge sees the registers and the inputs of one value, thus it hides the very skew the
     shared unit's contract exists for: an entry from the cycle before under a shift from
     this one. The draw's fuzz is where that claim is really made — it walks 48 classes a
     magnitude a cycle against the twin, and it read 58 disagreements of 60 on the shared
     unit before this fork stood. *)
  [%expect
    {| held two cycles: the shared unit and the fork agree true, and both state the rule true |}]
;;
