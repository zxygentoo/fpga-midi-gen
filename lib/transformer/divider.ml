(* The restoring divider — see divider.mli for the contract. The walk holds the magnitude
   alone and the sign lands at the output, thus the quotient truncates toward zero at
   every sign — the one division rule of the circuit. *)

open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a
    ; numerator : 'a [@bits 40]
    ; denominator : 'a [@bits 24]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { quotient : 'a [@bits 40]
    ; busy : 'a
    }
  [@@deriving hardcaml]
end

(* one cycle for each bit of the quotient, thus the width of the numerator *)
let busy_cycles = 40

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let m = Variable.reg spec ~width:40 in
  let d = Variable.reg spec ~width:24 in
  let q = Variable.reg spec ~width:40 in
  let r = Variable.reg spec ~width:25 in
  let n = Variable.reg spec ~width:(Int.ceil_log2 (busy_cycles + 1)) in
  let sign = Variable.reg spec ~width:1 in
  let busy = Variable.reg spec ~width:1 in
  compile
    [ if_
        i.start
        [ m <-- mux2 (i.numerator <+ zero 40) (negate i.numerator) i.numerator
        ; sign <-- (i.numerator <+ zero 40)
        ; d <-- i.denominator
        ; q <--. 0
        ; r <--. 0
        ; n <--. busy_cycles
        ; busy <-- vdd
        ]
        [ when_
            busy.value
            [ (let r' = sel_bottom (r.value @: msb m.value) ~width:25 in
               let fits = r' >=: uresize d.value ~width:25 in
               proc
                 [ m <-- sll m.value ~by:1
                 ; n <-- n.value -:. 1
                 ; r <-- mux2 fits (r' -: uresize d.value ~width:25) r'
                 ; q <-- sel_bottom (q.value @: fits) ~width:40
                 ; when_ (n.value ==:. 1) [ busy <-- gnd ]
                 ])
            ]
        ]
    ];
  { O.quotient = mux2 sign.value (negate q.value) q.value; busy = busy.value }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the divider is the reference division, toward zero" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let divide n d =
    inp.numerator := Bits.of_signed_int ~width:40 n;
    inp.denominator := Bits.of_unsigned_int ~width:24 d;
    inp.start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.start := Bits.gnd;
    while Bits.to_bool !(out.busy) do
      Cyclesim.cycle sim
    done;
    Bits.to_signed_int !(out.quotient)
  in
  List.iter
    [ 100, 7; -100, 7; 0, 5; 1234567, 89; -(1 lsl 38), 3 ]
    ~f:(fun (n, d) ->
      Stdio.printf "%d / %d = %d, the reference gives %d\n" n d (divide n d) (n / d));
  [%expect
    {|
    100 / 7 = 14, the reference gives 14
    -100 / 7 = -14, the reference gives -14
    0 / 5 = 0, the reference gives 0
    1234567 / 89 = 13871, the reference gives 13871
    -274877906944 / 3 = -91625968981, the reference gives -91625968981
    |}]
;;
