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

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let m = Variable.reg spec ~width:42 in
  let root = Variable.reg spec ~width:21 in
  let r = Variable.reg spec ~width:25 in
  let n = Variable.reg spec ~width:5 in
  let busy = Variable.reg spec ~width:1 in
  compile
    [ if_
        i.start
        [ m <-- i.value; root <--. 0; r <--. 0; n <--. 21; busy <-- vdd ]
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

let%expect_test "the isqrt floors, as the reference does" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let isqrt v =
    inp.value := Bits.of_unsigned_int ~width:42 v;
    inp.start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.start := Bits.gnd;
    while Bits.to_bool !(out.busy) do
      Cyclesim.cycle sim
    done;
    Bits.to_int_trunc !(out.root)
  in
  (* the oracle of the reference: floor of the square root *)
  let floor_sqrt v =
    let rec shrink g = if g * g > v then shrink (g - 1) else g in
    shrink (Float.to_int (Float.sqrt (Float.of_int v)) + 1)
  in
  List.iter
    [ 0; 15; 16; 4295; (1 lsl 41) + 12345 ]
    ~f:(fun v -> Stdio.printf "%d -> %d (oracle %d)\n" v (isqrt v) (floor_sqrt v));
  [%expect
    {|
    0 -> 0 (oracle 0)
    15 -> 3 (oracle 3)
    16 -> 4 (oracle 4)
    4295 -> 65 (oracle 65)
    2199023267897 -> 1482910 (oracle 1482910)
    |}]
;;
